"""本地分类头适配器（冻结主干特征 → 五头预测）。

严格镜像 ``starlife/starlife_scripts/eval_compare_cloud.py:129-196 local_predictions``
的推理路径，并沿用 ``train_heads.main`` 的特征拼装：

- change/type 覆盖全部 slot；
- relation/rel_pos/attrs 覆盖 ``build_relation_data`` 的行（**含负样本**，
  负样本追加在末尾；评测时 rel_pos/attrs 只取非 none 行）；
- pair 特征是否拼 9 维 bbox 几何由 ``relation.pt`` 的 ``d_in`` 唯一决定
  （v2 外观版 d_in=3D+1，v2g 几何版 d_in=3D+10），不做 z-score；
- 设备自动选择（``torch.cuda.is_available()``），无 GPU 也无硬依赖。

延迟语义：``HeadPrediction.latency_s`` 是 **per-pair 头 forward 实测均值（秒）**；
详细分布见 :meth:`LocalHeadsAdapter.runtime_stats`（ms 的 mean/p50/max）。
这里只测本地头推理，不含 27B 主干/渲染等整机指标（那部分由 report 引用
``bench/bench_pipeline.json`` 的真实测量值）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..dataset.negative_sampling import build_relation_data
from ..dataset.pair_data import (
    build_pair_feat,
    build_rel_pos_feat,
    build_slot_inputs,
    relation_onehot,
)
from ..types import CHANGE_LABELS, DatasetManifest, FeatureBundle, HeadPrediction, PairPrediction, RelationPrediction, RuntimeStats
from .heads import HeadCheckpoint, load_heads


@dataclass(frozen=True)
class HeadDims:
    """从特征数组推导出的、五头期望的输入维度（不含任何硬编码常量）。"""

    feature_dim: int
    slot_feat_dim: int
    type_feat_dim: int
    pair_base_dim: int
    pair_geo_dim: int
    rel_onehot_dim: int
    slot_geom_dim: int = 0

    @classmethod
    def from_features(cls, features: FeatureBundle, slot_geom_dim: int = 0) -> "HeadDims":
        d = int(features["slot_after"].shape[1])
        n_rel_classes = len(features.rel_types) + 1
        pair_base = 3 * d + 1
        geom_dim = int(slot_geom_dim)
        return cls(
            feature_dim=d,
            slot_feat_dim=3 * d + 1 + geom_dim,
            type_feat_dim=2 * d,
            pair_base_dim=pair_base,
            pair_geo_dim=pair_base + 9,
            rel_onehot_dim=n_rel_classes,
            slot_geom_dim=geom_dim,
        )


class LocalHeadsAdapter:
    """加载 ``ckpt_dir`` 下 5 个头，在给定 :class:`FeatureBundle` 上推理。

    Parameters
    ----------
    ckpt_dir:
        包含 ``change.pt / type.pt / relation.pt / rel_pos.pt / attrs.pt`` 的目录。
    features:
        :func:`starlife_eval.dataset.features.load_features` 的产物。
    manifest:
        dataset manifest（geometry 与 per-pair 转换必需；纯 v2 外观版也建议传）。
    device:
        覆盖默认设备；默认 CPU，有 CUDA 则 CUDA（不硬依赖）。
    slot_geometry:
        ``(N_slot, 13)`` slot 级几何特征（``build_slot_geometry.py`` 产物），给
        v3 的 change 头补位移；按训练脚本同一口径做 z-score（全量 mean/std），
        拼接在 ``[before, after, |diff|, mask]`` 之后。``None`` 表示 v2/v2g 外观版。
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        features: FeatureBundle,
        manifest: DatasetManifest | None = None,
        device: str | torch.device | None = None,
        slot_geometry: np.ndarray | None = None,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.features = features
        self.manifest = manifest
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.heads: dict[str, HeadCheckpoint] = load_heads(self.ckpt_dir)
        self._move_to_device()

        self.slot_geometry = self._prepare_slot_geometry(slot_geometry)
        self.dims = HeadDims.from_features(
            features, slot_geom_dim=0 if self.slot_geometry is None else int(self.slot_geometry.shape[1])
        )
        self._validate_heads()
        self._relation_data: dict[str, Any] | None = None
        self.latency_per_pair_s: list[float] = []

    def _prepare_slot_geometry(self, slot_geometry: np.ndarray | None) -> np.ndarray | None:
        """v3 slot 几何：校验行数 + 按训练脚本口径 z-score（全量 mean/std）。"""
        if slot_geometry is None:
            return None
        geom = np.asarray(slot_geometry, dtype=np.float32)
        n_slot = int(self.features["slot_after"].shape[0])
        if geom.ndim != 2 or geom.shape[0] != n_slot:
            raise ValueError(
                f"slot 几何特征形状 {geom.shape} 与 slot 数 {n_slot} 不一致（期望 (N_slot, 13)）"
            )
        mu = geom.mean(0, keepdims=True)
        sd = geom.std(0, keepdims=True) + 1e-6
        return ((geom - mu) / sd).astype(np.float32)
        self.last_inference_seconds: float | None = None

    # ------------------------------------------------------------------ setup
    def _move_to_device(self) -> None:
        for ckpt in self.heads.values():
            ckpt.model.to(self.device)

    @property
    def geometry_used(self) -> bool:
        """relation 头是否吃 9 维 bbox 几何（由 checkpoint d_in 反推，v2/v2g 通用）。"""
        return self.heads["relation"].d_in == self.dims.pair_geo_dim

    def _validate_heads(self) -> None:
        d = self.dims
        checks = {
            "change": (d.slot_feat_dim, len(CHANGE_LABELS)),
            "type": (d.type_feat_dim, len(self.features.type_keys)),
            "relation": (
                d.pair_geo_dim if self.geometry_used else d.pair_base_dim,
                len(self.features.rel_types) + 1,
            ),
            "rel_pos": (
                (d.pair_geo_dim if self.geometry_used else d.pair_base_dim) + d.rel_onehot_dim,
                len(self.features.rel_pos_keys),
            ),
            "attrs": ((d.pair_geo_dim if self.geometry_used else d.pair_base_dim), 2),
        }
        for name, (exp_d_in, exp_n_out) in checks.items():
            ckpt = self.heads[name]
            if ckpt.d_in != exp_d_in:
                raise ValueError(
                    f"{ckpt.path}: d_in={ckpt.d_in} 与特征维度 {exp_d_in} 不一致"
                    f"（feature_dim={d.feature_dim}, geometry_used={self.geometry_used}）"
                )
            if ckpt.n_classes != exp_n_out:
                raise ValueError(f"{ckpt.path}: n_classes={ckpt.n_classes} 与期望类别数 {exp_n_out} 不一致")
        if self.geometry_used and self.manifest is None:
            raise ValueError(
                "relation 头需要 9 维 bbox 几何（d_in=3D+10），但构造时未提供 DatasetManifest；"
                "请传入 manifest 或改用 v2 外观版 checkpoint"
            )

    @property
    def relation_data(self) -> dict[str, Any]:
        """``train_heads.build_relation_data`` 的结果（含负样本），惰性计算一次。"""
        if self._relation_data is None:
            self._relation_data = build_relation_data(self.features, self.features.meta)
        return self._relation_data

    # -------------------------------------------------------------- inference
    def _build_inputs(self) -> dict[str, np.ndarray]:
        rel = self.relation_data
        slot = build_slot_inputs(self.features)
        slot_feat = slot["slot_feat"]
        if self.slot_geometry is not None:
            slot_feat = np.concatenate([slot_feat, self.slot_geometry], axis=1)
        rel_d_in = self.heads["relation"].d_in
        pair_feat = build_pair_feat(self.features, rel, d_in=rel_d_in, manifest=self.manifest)
        onehot = relation_onehot(rel["label"], n_classes=self.dims.rel_onehot_dim)
        rel_pos_feat = build_rel_pos_feat(pair_feat, onehot)
        return {
            "slot_feat": slot_feat,
            "type_feat": slot["type_feat"],
            "pair_feat": pair_feat,
            "onehot": onehot,
            "rel_pos_feat": rel_pos_feat,
        }

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

    def predict(self, measure_latency_pairs: Sequence[int] | None = None) -> HeadPrediction:
        """跑完五个头，返回与 npz 行对齐的 :class:`HeadPrediction`。

        ``measure_latency_pairs``: 逐 pair 实测头 forward 延迟的 pair 行号；
        默认取 manifest 的 ``test`` split（无 manifest 时取 ``pair_split == "test"``，
        再退化为全部 pair）。传 ``[]`` 可跳过延迟测量。
        """
        inputs = self._build_inputs()

        t_slot = self._to_tensor(inputs["slot_feat"])
        t_type = self._to_tensor(inputs["type_feat"])
        t_pair = self._to_tensor(inputs["pair_feat"])
        t_pos = self._to_tensor(inputs["rel_pos_feat"])

        t0 = time.perf_counter()
        with torch.no_grad():
            change_p = self.heads["change"].model(t_slot).argmax(1).cpu().numpy()
            type_p = self.heads["type"].model(t_type).argmax(1).cpu().numpy()
            rel_p = self.heads["relation"].model(t_pair).argmax(1).cpu().numpy()
            pos_p = self.heads["rel_pos"].model(t_pos).argmax(1).cpu().numpy()
            attr_p = (self.heads["attrs"].model(t_pair).sigmoid() > 0.5).cpu().numpy().astype(np.int64)
        self.last_inference_seconds = time.perf_counter() - t0

        rel = self.relation_data
        n_pos = int(np.sum(~rel["is_none"]))
        n_rows = int(len(rel["frm"]))
        if n_pos != int(len(self.features["rel_from"])):
            raise ValueError(
                f"负采样正样本数 {n_pos} 与 npz rel_from 长度 {len(self.features['rel_from'])} 不一致"
            )
        head = HeadPrediction(
            change=np.asarray(change_p, dtype=np.int64),
            type=np.asarray(type_p, dtype=np.int64),
            relation=np.asarray(rel_p, dtype=np.int64),
            rel_pos=np.asarray(pos_p, dtype=np.int64),
            attrs=np.asarray(attr_p, dtype=np.int64),
            n_relation_positives=n_pos,
            latency_s=None,
        )
        if len(head.relation) != n_rows:
            raise ValueError(f"relation 预测长度 {len(head.relation)} != 负采样行数 {n_rows}")

        self.latency_per_pair_s = self._measure_latency(inputs, measure_latency_pairs)
        if self.latency_per_pair_s:
            head = HeadPrediction(
                change=head.change,
                type=head.type,
                relation=head.relation,
                rel_pos=head.rel_pos,
                attrs=head.attrs,
                n_relation_positives=head.n_relation_positives,
                latency_s=float(np.mean(self.latency_per_pair_s)),
            )
        return head

    # --------------------------------------------------------------- latency
    def default_latency_pairs(self) -> list[int]:
        if self.manifest is not None:
            test_idx = self.manifest.split_indices("test")
            if test_idx:
                return test_idx
        pair_split = self.features["pair_split"]
        idx = [int(i) for i in np.where(pair_split == "test")[0]]
        return idx if idx else list(range(len(pair_split)))

    def _measure_latency(
        self, inputs: Mapping[str, np.ndarray], pairs: Sequence[int] | None
    ) -> list[float]:
        """逐 pair 实测五个头 forward 的墙钟耗时（秒）；不含 numpy 拼装。"""
        pair_list = self.default_latency_pairs() if pairs is None else [int(p) for p in pairs]
        if not pair_list:
            return []
        slot_pair = self.features["slot_pair"]
        rel = self.relation_data
        rel_pair = np.asarray(slot_pair[rel["frm"]]) if len(rel["frm"]) else np.zeros((0,), dtype=np.int64)

        latencies: list[float] = []
        model_change = self.heads["change"].model
        model_type = self.heads["type"].model
        model_rel = self.heads["relation"].model
        model_pos = self.heads["rel_pos"].model
        model_attr = self.heads["attrs"].model
        with torch.no_grad():
            for pid in pair_list:
                srows = np.where(slot_pair == pid)[0]
                rrows = np.where(rel_pair == pid)[0]
                ts = self._to_tensor(inputs["slot_feat"][srows]) if len(srows) else None
                tt = self._to_tensor(inputs["type_feat"][srows]) if len(srows) else None
                tp = self._to_tensor(inputs["pair_feat"][rrows]) if len(rrows) else None
                tpos = self._to_tensor(inputs["rel_pos_feat"][rrows]) if len(rrows) else None
                t0 = time.perf_counter()
                if ts is not None:
                    model_change(ts)
                    model_type(tt)
                if tp is not None:
                    model_rel(tp)
                    model_pos(tpos)
                    model_attr(tp)
                latencies.append(time.perf_counter() - t0)
        return latencies

    def runtime_stats(self) -> RuntimeStats:
        """本地头推理延迟（ms）真实测量值；成本为自有硬件 0（口径见 ARCHITECTURE §5.5）。"""
        if not self.latency_per_pair_s:
            return RuntimeStats(source="measured", cost_usd=0.0, cost_basis="self-hosted local heads; no external billing")
        arr = np.asarray(self.latency_per_pair_s, dtype=np.float64) * 1000.0
        return RuntimeStats(
            latency_ms_mean=float(arr.mean()),
            latency_ms_p50=float(np.percentile(arr, 50)),
            latency_ms_max=float(arr.max()),
            cost_usd=0.0,
            cost_basis="self-hosted local heads (own hardware); no external billing",
            n_calls=len(self.latency_per_pair_s),
            source="measured",
        )

    # ------------------------------------------------------- per-pair conversion
    def predict_pairs(self, sample_indices: Sequence[int] | None = None) -> dict[int, PairPrediction]:
        """跑头 + 转成 60 对协议用的 per-pair 结构（镜像 eval_compare_cloud local_map）。"""
        if self.manifest is None:
            raise ValueError("predict_pairs 需要 DatasetManifest（per-pair slot 映射）")
        head = self.predict()
        return head_predictions_to_pairs(
            head, self.features, self.manifest, sample_indices, relation_data=self.relation_data
        )


def head_predictions_to_pairs(
    head: HeadPrediction,
    features: FeatureBundle,
    manifest: DatasetManifest,
    sample_indices: Iterable[int] | None = None,
    *,
    relation_data: Mapping[str, Any] | None = None,
) -> dict[int, PairPrediction]:
    """全局 slot 下标 → per-pair slot 编号，只保留采样 pair。

    ported from eval_compare_cloud.py:289-305（``local_map`` 构造）：
    ``idx2slot = {offset + k: e["slot"]}``；changes/types 覆盖该 pair 的所有 slot；
    relations 只保留非 none 的真关系，from/to 换成 pair 内 slot 编号。
    """
    rel = relation_data if relation_data is not None else build_relation_data(features, features.meta)
    n_pos = int(np.sum(~rel["is_none"]))  # 正样本前 n_pos 行（build_relation_data 保证正样本在前）
    n_rows = int(len(rel["frm"]))
    if int(head.n_relation_positives) != n_pos:
        raise ValueError(
            f"HeadPrediction.n_relation_positives={head.n_relation_positives} 与 relation_data 正样本数 {n_pos} 不一致"
        )
    if len(head.relation) != n_rows:
        raise ValueError(f"relation 预测长度 {len(head.relation)} != 负采样行数 {n_rows}")
    if len(head.change) != int(features["slot_pair"].shape[0]):
        raise ValueError(
            f"change 预测长度 {len(head.change)} != N_slot {features['slot_pair'].shape[0]}"
        )

    if sample_indices is None:
        sample = list(range(len(manifest.pairs)))
    else:
        sample = sorted({int(i) for i in sample_indices})

    out: dict[int, PairPrediction] = {i: PairPrediction(pair_index=i) for i in sample}
    slot_pair = features["slot_pair"]
    type_keys = features.type_keys
    rel_types = features.rel_types
    rel_pos_keys = features.rel_pos_keys

    for pid in sample:
        pair = manifest.pairs[pid]
        base = manifest.slot_offsets[pid]
        idx2slot = {base + e.slot: e.slot for e in pair.entities}
        pred = out[pid]
        for gslot in range(base, manifest.slot_offsets[pid + 1]):
            slot = idx2slot[gslot]
            pred.changes[slot] = CHANGE_LABELS[int(head.change[gslot])]
            pred.types[slot] = type_keys[int(head.type[gslot])]

    for k in range(n_pos):
        frm, to = int(rel["frm"][k]), int(rel["to"][k])
        pid = int(slot_pair[frm])
        if pid not in out or bool(rel["is_none"][k]):
            continue
        base = manifest.slot_offsets[pid]
        rel_label = int(head.relation[k])
        pred = out[pid]
        pred.relations.append(
            RelationPrediction(
                from_slot=frm - base,
                to_slot=to - base,
                type=rel_types[rel_label] if rel_label < len(rel_types) else "none",
                rel_pos=rel_pos_keys[int(head.rel_pos[k])],
                contact=bool(int(head.attrs[k, 0])),
                support=bool(int(head.attrs[k, 1])),
            )
        )
    return out
