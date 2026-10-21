"""60 对云端对照协议（L2-b）：严格镜像 ``eval_compare_cloud.py::main/evaluate``。

口径来源（**只读**移植，不 import 原脚本）：
``starlife/starlife_scripts/eval_compare_cloud.py`` 第 199-324 行：

* 抽样：``test_idx`` 上 ``random.Random(20261020).sample(..., n=60)`` 后 **sorted**
  （第 211-213 行）；
* ``change``：遍历每对 **所有 entities 的 slot**，GT ``changes`` 缺省为 ``none``，
  预测缺省 ``none``；得到 ``change_slots / change_top1 / change_macro_f1``，
  其中 ``change_macro_f1`` 用 :func:`starlife_eval.verifiers.metrics.macro_f1_compare`
  （**跳过 tp==0 的类**，第 31-41 行；云端 0.7781、本地 0.6409 依赖此口径）；
* ``relations``：按 ``(from_slot, to_slot)`` 匹配（原脚本 dict 推导，重复键后者覆盖），
  在 GT 关系对上算 ``relation_type_acc``；
* ``rel_pos_acc``：GT 关系上、类型命中且 GT 带 ``rel_pos`` 的样本；
* ``attrs_acc``：GT 关系上、类型命中时 contact/support **各算 1 项**。
* **所有指标 round(,4)**（原脚本如此）；要全精度请直接用
  :mod:`starlife_eval.verifiers.metrics` 的原语。

注意：本模块不依赖 torch / dataset / models，``manifest`` 是
:class:`starlife_eval.types.DatasetManifest`，``preds`` 是
``dict[pair_index, PairPrediction]``（云端预测的 ``from_slot/to_slot`` 是 pair 内 slot 编号）。
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import (
    CHANGE_INDEX,
    CHANGE_LABELS,
    DatasetManifest,
    HeadPrediction,
    PairPrediction,
    RelationPrediction,
)
from .metrics import macro_f1_compare

__all__ = ["sample_pairs", "evaluate_pair_protocol", "local_pair_predictions"]


def sample_pairs(
    manifest: DatasetManifest,
    split: str = "test",
    n: int = 60,
    seed: int = 20261020,
) -> list[int]:
    """在 ``split`` 的 pair 行号上抽 ``n`` 对，返回**升序**的 manifest 行号。

    逐行对齐 ``eval_compare_cloud.main`` 第 211-213 行::

        test_idx = [i for i, r in enumerate(rows) if r["split"] == split]
        rng = random.Random(seed)
        sample = sorted(rng.sample(test_idx, min(n, len(test_idx))))

    ``n`` 大于该 split 的对数时取全部（与原脚本 ``min`` 行为一致）。
    """
    if n < 0:
        raise ValueError(f"n 必须 >= 0，实际 {n}")
    candidates = list(manifest.split_indices(split))
    if not candidates:
        raise ValueError(f"split={split!r} 没有任何 pair，无法抽样")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, min(int(n), len(candidates))))


def _pair_by_index(manifest: DatasetManifest) -> dict[int, Any]:
    return {p.index: p for p in manifest.pairs}


def _gt_has_rel_pos(pair: Any, rel_idx: int) -> bool:
    """GT 关系是否带 ``rel_pos`` 字段（原脚本 ``"rel_pos" in g``）。

    优先用 manifest 原始行判断字段是否存在；拿不到 raw 时退回
    ``Relation.rel_pos is not None``（二者在真实产物上等价）。
    """
    raw = pair.raw.get("relations") if isinstance(pair.raw, Mapping) else None
    if isinstance(raw, (list, tuple)) and len(raw) == len(pair.relations):
        item = raw[rel_idx]
        if isinstance(item, Mapping):
            return "rel_pos" in item
    return pair.relations[rel_idx].rel_pos is not None


def _empty_prediction(pair_index: int) -> PairPrediction:
    return PairPrediction(pair_index=pair_index)


def evaluate_pair_protocol(
    manifest: DatasetManifest,
    sample_indices: Sequence[int],
    preds: Mapping[int, PairPrediction],
    *,
    strict: bool = True,
) -> dict[str, int | float]:
    """在抽样 pair 上计算云端可比协议指标（镜像 eval_compare_cloud.evaluate）。

    Parameters
    ----------
    manifest:
        数据集 manifest（``PairSample.relations`` 是 after 帧 GT）。
    sample_indices:
        要评测的 pair 行号（通常来自 :func:`sample_pairs`）。
    preds:
        ``{pair_index: PairPrediction}``；预测的 ``changes/types`` 用 pair 内 slot 编号，
        ``relations`` 用 ``(from_slot, to_slot)``。
    strict:
        ``True``（默认）时缺预测立即 ``KeyError``，避免静默按全错计分；
        ``False`` 时按"空预测"计（等价原脚本 ``cloud_pred`` 解析失败的兜底）。

    Returns
    -------
    dict
        ``{change_slots, change_top1, change_macro_f1, relation_pairs,
        relation_type_acc, rel_pos_acc, attrs_acc}``，数值 ceil 到 4 位小数
        （与原脚本一致，便于与 reference 逐位对齐）。
    """
    by_index = _pair_by_index(manifest)

    ch_true: list[str] = []
    ch_pred: list[str] = []
    rel_total = rel_hit = 0
    pos_total = pos_hit = 0
    attr_total = attr_hit = 0

    for i in sample_indices:
        pair = by_index.get(i)
        if pair is None:
            raise KeyError(f"manifest 中不存在 pair_index={i}")
        pred = preds.get(i)
        if pred is None:
            if strict:
                raise KeyError(
                    f"缺失 pair 预测: pair_index={i}（preds 里没有该 pair；"
                    "回放缓存不完整时应报错而不是按全错计分）"
                )
            pred = _empty_prediction(i)

        # ---------------- change：所有 entity slot 逐槽比较 -------------------
        gt_change: Mapping[int, str] = pair.changes
        pch: Mapping[int, str] = pred.changes
        for entity in pair.entities:
            slot = entity.slot
            ch_true.append(gt_change.get(slot, "none"))
            ch_pred.append(pch.get(slot, "none"))

        # ---------------- relations：按 (from_slot,to_slot) 匹配 --------------
        # 原脚本用 dict 推导，重复键保留最后一条
        p_rel = {
            (r.from_slot, r.to_slot): r for r in pred.relations
        }
        for rel_idx, g in enumerate(pair.relations):
            key = (g.from_slot, g.to_slot)
            rel_total += 1
            cand = p_rel.get(key)
            cand_type = getattr(cand, "type", None)
            if cand_type == g.type:
                rel_hit += 1
                if _gt_has_rel_pos(pair, rel_idx):
                    pos_total += 1
                    if getattr(cand, "rel_pos", None) == g.rel_pos:
                        pos_hit += 1
                attr_total += 2
                attr_hit += int(bool(getattr(cand, "contact", None)) == bool(g.contact))
                attr_hit += int(bool(getattr(cand, "support", None)) == bool(g.support))

    # change 的字符串 -> 类别下标；非法预测标签用 -1 表示（原脚本 CHANGE.index 会崩，
    # 这里改为按"错误"计入 top1 且不参与 macro 的类，保证一个坏模型输出不中断整套评测）
    true_idx = [CHANGE_INDEX.get(x, -1) for x in ch_true]
    pred_idx = [CHANGE_INDEX.get(x, -1) for x in ch_pred]

    return {
        "change_slots": len(ch_true),
        "change_top1": round(
            float(np.mean([a == b for a, b in zip(ch_true, ch_pred)])) if ch_true else 0, 4
        ),
        "change_macro_f1": round(macro_f1_compare(true_idx, pred_idx, len(CHANGE_LABELS)), 4),
        "relation_pairs": rel_total,
        "relation_type_acc": round(rel_hit / rel_total, 4) if rel_total else 0,
        "rel_pos_acc": round(pos_hit / pos_total, 4) if pos_total else 0,
        "attrs_acc": round(attr_hit / attr_total, 4) if attr_total else 0,
    }


# ------------------------------------------------- local head -> per-pair preds


def _pair_of_global_slot(manifest: DatasetManifest, global_slot: int) -> int:
    """npz 全局 slot 下标 -> manifest pair 行号（``slot_offsets`` 二分）。"""
    offsets = np.asarray(manifest.slot_offsets, dtype=np.int64).reshape(-1)
    if not (0 <= global_slot < int(offsets[-1])):
        raise ValueError(f"global slot {global_slot} 越界（总 slot 数 {int(offsets[-1])}）")
    return int(np.searchsorted(offsets, global_slot, side="right")) - 1


def local_pair_predictions(
    manifest: DatasetManifest,
    head: HeadPrediction,
    relation: Mapping[str, Any],
    sample_indices: Sequence[int],
    meta: Mapping[str, Any] | None = None,
) -> dict[int, PairPrediction]:
    """本地头 ``HeadPrediction``（全局行对齐）-> 逐 pair ``PairPrediction``。

    逐行镜像 ``eval_compare_cloud.local_predictions`` + ``main`` 里 ``local_map`` 的
    ``idx2slot`` 映射（第 177-196、289-305 行），供 Lead 的 runner 直接把本地头接进
    60 对协议 / 事件翻译：

    * ``changes`` 覆盖 sample pair 的全部 entity slot（``global = slot_offsets[pair] + pos``，
      再映射回 per-pair ``entities[pos].slot``）；
    * ``relations`` 只取**非 none 的真关系**（负样本丢弃），``from/to`` 转成 per-pair slot；
    * 关系类型名/方位名优先用 relation dict 的 ``types`` / ``rel_pos_keys``，
      缺省用 ``meta["rel_types"]`` / ``meta["rel_pos_keys"]``；
    * ``PairPrediction.types`` 在 ``meta`` 可用时一并填（协议不用，事件/状态回放可用）。

    Parameters
    ----------
    manifest:
        真实 manifest；``slot_offsets`` 必须是**全局** slot 偏移（真实加载的 manifest 满足）。
    head:
        本地头预测（change/type 按全局 slot 行对齐，relation/rel_pos/attrs 按 relation
        采样行对齐，含负样本）。
    relation:
        ``train_heads.build_relation_data`` 的 dict（dataset 层产出，含负样本）。
    sample_indices:
        需要转成逐 pair 预测的 pair 行号（通常是 60 对抽样）。
    meta:
        特征元信息；给了就同时填 types，缺省只填 changes/relations。
    """
    offsets = np.asarray(manifest.slot_offsets, dtype=np.int64).reshape(-1)
    total_slots = int(offsets[-1])
    if len(head.change) != total_slots:
        raise ValueError(
            f"head.change 长度 {len(head.change)} != manifest 总 slot 数 {total_slots}"
        )

    rel_types: list[str] | None = None
    if "types" in relation:
        rel_types = list(relation["types"])
    elif meta is not None and "rel_types" in meta:
        rel_types = list(meta["rel_types"])
    rel_pos_keys: list[str] | None = None
    if "rel_pos_keys" in relation:
        rel_pos_keys = list(relation["rel_pos_keys"])
    elif meta is not None and "rel_pos_keys" in meta:
        rel_pos_keys = list(meta["rel_pos_keys"])
    type_keys = list(meta["type_keys"]) if meta is not None and "type_keys" in meta else None

    by_index = _pair_by_index(manifest)
    sample = [int(i) for i in sample_indices]
    sample_set = set(sample)
    idx2slot: dict[int, int] = {}
    out: dict[int, PairPrediction] = {}

    for pair_index in sample:
        pair = by_index.get(pair_index)
        if pair is None:
            raise KeyError(f"manifest 中不存在 pair_index={pair_index}")
        base = int(offsets[pair_index])
        changes: dict[int, str] = {}
        types: dict[int, str] = {}
        for pos, entity in enumerate(pair.entities):
            global_slot = base + pos
            idx2slot[global_slot] = int(entity.slot)
            change_idx = int(head.change[global_slot])
            if not (0 <= change_idx < len(CHANGE_LABELS)):
                raise ValueError(
                    f"head.change[{global_slot}]={change_idx} 不是合法类别 "
                    f"(0..{len(CHANGE_LABELS) - 1})"
                )
            changes[int(entity.slot)] = CHANGE_LABELS[change_idx]
            if type_keys is not None:
                type_idx = int(head.type[global_slot])
                if 0 <= type_idx < len(type_keys):
                    types[int(entity.slot)] = type_keys[type_idx]
        out[pair_index] = PairPrediction(
            pair_index=pair_index, changes=changes, types=types, relations=[]
        )

    frm = np.asarray(relation["frm"]).reshape(-1)
    to = np.asarray(relation["to"]).reshape(-1)
    is_none = np.asarray(relation["is_none"]).astype(bool).reshape(-1)
    n_rel = int(frm.shape[0])
    if len(head.relation) != n_rel or len(head.rel_pos) != n_rel or len(head.attrs) != n_rel:
        raise ValueError(
            "head 的关系数组长度与 relation 采样结果不一致: "
            f"frm={n_rel} relation={len(head.relation)} rel_pos={len(head.rel_pos)} "
            f"attrs={len(head.attrs)}"
        )
    if rel_types is None or rel_pos_keys is None:
        raise ValueError("缺少关系类型/rel_pos 词表：请传 meta 或带 types/rel_pos_keys 的 relation")
    attrs = np.asarray(head.attrs)

    for k in range(n_rel):
        if is_none[k]:
            continue
        pair_index = _pair_of_global_slot(manifest, int(frm[k]))
        if pair_index not in sample_set:
            continue
        from_global, to_global = int(frm[k]), int(to[k])
        type_idx = int(head.relation[k])
        pos_idx = int(head.rel_pos[k])
        out[pair_index].relations.append(
            RelationPrediction(
                from_slot=idx2slot.get(from_global, from_global),
                to_slot=idx2slot.get(to_global, to_global),
                type=rel_types[type_idx] if type_idx < len(rel_types) else "none",
                rel_pos=rel_pos_keys[pos_idx] if 0 <= pos_idx < len(rel_pos_keys) else None,
                contact=bool(attrs[k, 0]),
                support=bool(attrs[k, 1]),
            )
        )
    return out
