"""头输入特征拼装（numpy；不依赖 torch）。

逐行对齐两处产线脚本，任何顺序/拼接位置变化都会让 v2/v2g 的验收数字偏掉：

- ``train_heads.py:224-238`` (main, slot_feat / type_feat)
- ``train_heads.py:259-270`` (main, pair_feat / onehot / rel_pos 输入)
- ``eval_compare_cloud.py:140-160`` (local_predictions 同一套拼装 + 依据 ckpt d_in 决定几何)

维度约定（D = 主干特征维度，取自 npz 形状；不硬编码）：

===============  ==========================  ==========================================
数组             形状                        出处
===============  ==========================  ==========================================
slot_feat        (N_slot, 3D+1)              [slot_before, slot_after, |diff|, mask]
type_feat        (N_slot, 2D)                [slot_after, |diff|]
pair_base        (N_rel, 3D+1)               [after[frm], after[to], |after[frm]-after[to]|, dist]
pair_feat        (N_rel, 3D+1) / (N_rel, 3D+10)  pair_base (+9 维 bbox 几何，由 ckpt d_in 决定)
onehot           (N_rel, len(rel_types)+1)   关系 GT label 的 one-hot（none→最后一维）
rel_pos 输入     (N_rel, pair_dim + onehot_dim)  concat([pair_feat, onehot])
===============  ==========================  ==========================================
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..types import DatasetManifest
from .geometry import GEOMETRY_DIM, bbox_map_from_manifest, pair_geometry


def _arr(z: Any, key: str) -> np.ndarray:
    return z[key]  # type: ignore[index]


def _meta_of(z: Any, meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if meta is not None:
        return meta
    inner = getattr(z, "meta", None)
    if inner is None:
        raise TypeError("需要 meta（或带 .meta 的 FeatureBundle）")
    return inner


def build_slot_inputs(z: Any) -> dict[str, np.ndarray]:
    """slot 级输入（change/type 头共用），一次算好 diff 避免重复大数组减法。"""
    slot_after = _arr(z, "slot_after").astype(np.float32)
    slot_before = _arr(z, "slot_before").astype(np.float32)
    diff = np.abs(slot_after - slot_before)
    slot_feat = np.concatenate(
        [slot_before, slot_after, diff, _arr(z, "slot_mask").astype(np.float32)[:, None]], axis=1
    )
    type_feat = np.concatenate([slot_after, diff], axis=1)
    return {
        "slot_after": slot_after,
        "slot_before": slot_before,
        "diff": diff,
        "slot_feat": slot_feat,
        "type_feat": type_feat,
        "slot_mask": _arr(z, "slot_mask").astype(np.float32),
        "slot_change": _arr(z, "slot_change").astype(np.int64),
        "slot_type": _arr(z, "slot_type").astype(np.int64),
        "slot_pair": _arr(z, "slot_pair"),
        "slot_split": _arr(z, "pair_split")[_arr(z, "slot_pair")],
    }


def build_slot_feat(z: Any) -> np.ndarray:
    """change 头输入 ``[slot_before, slot_after, |diff|, mask]``。

    ported from train_heads.py:225-229
    """
    return build_slot_inputs(z)["slot_feat"]


def build_type_feat(z: Any) -> np.ndarray:
    """type 头输入 ``[slot_after, |diff|]``。

    ported from train_heads.py:252
    """
    return build_slot_inputs(z)["type_feat"]


def build_pair_base(z: Any, rel: Mapping[str, Any]) -> np.ndarray:
    """relation/attrs 基础输入（不含 bbox 几何）。

    ported from train_heads.py:261-263 / eval_compare_cloud.py:148-150
    """
    slot_after = _arr(z, "slot_after").astype(np.float32)
    frm, to = rel["frm"], rel["to"]
    return np.concatenate(
        [
            slot_after[frm],
            slot_after[to],
            np.abs(slot_after[frm] - slot_after[to]),
            rel["dist"][:, None],
        ],
        axis=1,
    )


def build_pair_feat(
    z: Any,
    rel: Mapping[str, Any],
    d_in: int,
    manifest: DatasetManifest | None = None,
    bbox_map: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    """pair 特征；是否拼接 9 维 bbox 几何由 checkpoint 的 ``d_in`` 决定（唯一权威）。

    - ``d_in == pair_base_dim``          → 不拼几何（v2 外观版）
    - ``d_in == pair_base_dim + 9``      → 拼 ``pair_geometry``（v2g 几何版）
    - 其他                               → ValueError（绝不静默 pad / 截断）

    ported from eval_compare_cloud.py:151-158（依据 relation.pt 的 d_in 条件拼接）
    """
    base = build_pair_base(z, rel)
    base_dim = int(base.shape[1])
    d_in = int(d_in)
    if d_in == base_dim:
        return base
    if d_in == base_dim + GEOMETRY_DIM:
        if bbox_map is None:
            if manifest is None:
                raise ValueError(
                    f"relation 头 d_in={d_in} 需要 {GEOMETRY_DIM} 维 bbox 几何，但未提供 manifest/bbox_map"
                )
            bbox_map = bbox_map_from_manifest(manifest)
        geo = pair_geometry(rel, bbox_map)
        if geo.shape[0] != base.shape[0]:
            raise ValueError(f"bbox 几何行数 {geo.shape[0]} != 关系行数 {base.shape[0]}")
        return np.concatenate([base, geo], axis=1)
    raise ValueError(
        f"relation 头 d_in={d_in} 与特征维度不匹配：pair_base={base_dim}，"
        f"pair_base+{GEOMETRY_DIM}={base_dim + GEOMETRY_DIM}"
    )


def relation_onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """关系 GT label 的 one-hot，``n_classes = len(rel_types)+1``（none→最后一维）。

    ported from train_heads.py:268-269 / eval_compare_cloud.py:159-160
    ``onehot[arange, minimum(label, len(types))] = 1``
    """
    labels = np.asarray(labels)
    n_classes = int(n_classes)
    if n_classes <= 0:
        raise ValueError(f"n_classes 必须 > 0，实际 {n_classes}")
    onehot = np.zeros((len(labels), n_classes), dtype=np.float32)
    if len(labels):
        onehot[np.arange(len(labels)), np.minimum(labels, n_classes - 1)] = 1
    return onehot


def build_rel_pos_feat(pair_feat: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """rel_pos 头输入 = ``[pair_feat, onehot]``。port train_heads.py:270。"""
    if len(pair_feat) != len(onehot):
        raise ValueError(f"pair_feat 行数 {len(pair_feat)} != onehot 行数 {len(onehot)}")
    return np.concatenate([pair_feat, onehot], axis=1)


def build_pair_inputs(
    z: Any,
    rel: Mapping[str, Any],
    d_in: int,
    manifest: DatasetManifest | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """一次产出 relation / attrs / rel_pos 三个头的输入（顺序与 train_heads.main 一致）。"""
    meta = _meta_of(z, meta)
    pair_feat = build_pair_feat(z, rel, d_in=d_in, manifest=manifest)
    onehot = relation_onehot(rel["label"], n_classes=len(meta["rel_types"]) + 1)
    rel_pos_feat = build_rel_pos_feat(pair_feat, onehot)
    return {"pair_feat": pair_feat, "onehot": onehot, "rel_pos_feat": rel_pos_feat}


def head_input_dims(bundle: Any) -> dict[str, int]:
    """从特征数组推导五个头的期望 d_in（用于校验 checkpoint，不读死维度）。"""
    d = int(_arr(bundle, "slot_after").shape[1])
    n_rel_classes = len(_meta_of(bundle, None)["rel_types"]) + 1
    pair_dim = 3 * d + 1
    return {
        "change": 3 * d + 1,
        "type": 2 * d,
        "relation_base": pair_dim,
        "relation_geo": pair_dim + GEOMETRY_DIM,
        "rel_pos_base": pair_dim + n_rel_classes,
        "rel_pos_geo": pair_dim + GEOMETRY_DIM + n_rel_classes,
        "attrs_base": pair_dim,
        "attrs_geo": pair_dim + GEOMETRY_DIM,
    }
