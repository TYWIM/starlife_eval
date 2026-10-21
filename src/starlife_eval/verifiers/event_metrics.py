"""L3 事件级指标：预测事件流 vs GT 事件流的逐类 P/R/F1 + macro-F1。

口径来源：`starlife_eval/ARCHITECTURE.md` §5.3、`测试套件设计.md` §3「event_F1」。
事件由 `starlife_eval.translator` 按 `WorldEvent.cs` 四类事件契约翻译得到。

匹配单位（per-pair slot 编号，ID 分配顺序无关）：
- `EntityCreated`  : (pair_index, slot, type)
- `StateChanged`   : (pair_index, slot, change_type)
- `RelationAdded`  : (pair_index, from_slot, to_slot, relation_type)
- `RelationRemoved`: (pair_index, from_slot, to_slot, relation_type)

**诚实的 N/A 约定（v2 数据现实）**：
- `RelationRemoved` 恒为 N/A：v2 数据集只标注 after 帧关系，没有 before 关系标注，
  构造不出 GT；`before_relations_available=True` 仅供 L1 合成的多帧用例显式开启。
- `EntityCreated` 的 type 维度只有本地头有；云端零样本预测不返回实体类型，
  此时 `EntityCreated` 不可比，标 N/A（不给 0 分）。
- `id_switch_rate`（跨帧 ID 连续）在 v2 上不可测（无连续帧序列），恒 N/A。
- 某类事件在 GT 与预测里都为 0 条时，该类的 P/R/F1 无定义，标 N/A，不虚构成 1.0 或 0.0。
- `macro_f1` 只对 **available 的类** 等权平均；N/A 类不参与（也不计 0 分）。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..translator import (
    EVENT_KINDS,
    ID_SWITCH_NA_REASON,
    RELATION_REMOVED_NA_REASON,
    EventTranslator,
    WorldEvent,
    as_prediction_map,
    select_pairs,
    translate_gt,
)
from ..types import DatasetManifest, PairPrediction, PairSample

#: 云端零样本预测没有实体类型，EntityCreated(type) 维度不可比时的 N/A 原因。
ENTITY_CREATED_TYPE_NA_REASON: str = (
    "预测侧没有实体类型（云端零样本预测只返回 changes/relations，不返回 type）；"
    "EntityCreated 的 type 维度仅本地头可测，故该事件类按 N/A 处理，不以 0 分计入 macro-F1。"
)


# --------------------------------------------------------------------- matching


def _event_key(ev: WorldEvent) -> tuple[Any, ...]:
    """事件匹配键；缺少 per-pair slot 元数据的事件不能参与事件级匹配。"""
    if ev.kind == "StateChanged":
        if ev.pair_index is None or ev.slot is None:
            raise ValueError("StateChanged 缺少 pair_index/slot 元数据，无法做事件匹配")
        return (int(ev.pair_index), int(ev.slot), str(ev.properties["change"]))
    if ev.kind == "EntityCreated":
        if ev.pair_index is None or ev.slot is None:
            raise ValueError("EntityCreated 缺少 pair_index/slot 元数据，无法做事件匹配")
        return (int(ev.pair_index), int(ev.slot), str(ev.entity_type))
    if ev.kind in ("RelationAdded", "RelationRemoved"):
        if ev.pair_index is None or ev.from_slot is None or ev.to_slot is None:
            raise ValueError(f"{ev.kind} 缺少 pair_index/from_slot/to_slot 元数据，无法做事件匹配")
        return (
            int(ev.pair_index),
            int(ev.from_slot),
            int(ev.to_slot),
            str(ev.relation_type),
        )
    raise ValueError(f"非法事件类型 {ev.kind!r}")


def _bucket(events: Sequence[WorldEvent]) -> dict[str, set[tuple[Any, ...]]]:
    out: dict[str, set[tuple[Any, ...]]] = {kind: set() for kind in EVENT_KINDS}
    for ev in events:
        if not isinstance(ev, WorldEvent):
            raise TypeError(f"events 元素不是 WorldEvent: {type(ev)!r}")
        out[ev.kind].add(_event_key(ev))
    return out


def _prf(support: int, predicted: int, matched: int) -> tuple[float, float, float]:
    """由集合匹配数算 P/R/F1；分母为 0 时按 0.0 处理（该情形由调用方决定是否 N/A）。"""
    precision = matched / predicted if predicted else 0.0
    recall = matched / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def _na_entry(
    reason: str, support: int, predicted: int, matched: int
) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "precision": None,
        "recall": None,
        "f1": None,
        "support": int(support),
        "predicted": int(predicted),
        "matched": int(matched),
    }


# --------------------------------------------------------------------- metrics


def evaluate_events(
    gt_events: Sequence[WorldEvent],
    pred_events: Sequence[WorldEvent],
    *,
    pred_has_types: bool | None = None,
    before_relations_available: bool = False,
) -> dict[str, Any]:
    """按四类事件分别算 P/R/F1，再对 available 类等权平均得 event macro-F1。

    Args:
        gt_events: GT 事件流（`translate_gt(manifest, ...)`）。
        pred_events: 预测事件流（`EventTranslator().translate(preds, ...)`）。
        pred_has_types: 预测侧是否有实体类型。`False` 时 EntityCreated 维度标 N/A；
            `None` 视为有类型（不做 N/A 判定，仅显式 False 才 N/A，避免把"漏检"
            误判成"不可测"）。
        before_relations_available: 是否存在 before 关系标注（v2 为 False → RelationRemoved 恒 N/A）。
    """
    gt_bucket = _bucket(gt_events)
    pred_bucket = _bucket(pred_events)
    per_kind: dict[str, dict[str, Any]] = {}
    available_kinds: list[str] = []
    na: dict[str, str] = {"id_switch_rate": ID_SWITCH_NA_REASON}

    for kind in EVENT_KINDS:
        gt_keys = gt_bucket[kind]
        pred_keys = pred_bucket[kind]
        support = len(gt_keys)
        predicted = len(pred_keys)
        matched = len(gt_keys & pred_keys)

        if kind == "RelationRemoved" and not before_relations_available:
            entry = _na_entry(RELATION_REMOVED_NA_REASON, support, predicted, matched)
        elif kind == "EntityCreated" and pred_has_types is False:
            entry = _na_entry(
                ENTITY_CREATED_TYPE_NA_REASON, support, predicted, matched
            )
        elif support == 0 and predicted == 0:
            entry = _na_entry(
                f"该子集既无 GT {kind} 事件也无预测 {kind} 事件，P/R/F1 无定义，按 N/A 处理",
                support,
                predicted,
                matched,
            )
        else:
            precision, recall, f1 = _prf(support, predicted, matched)
            entry = {
                "available": True,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted": predicted,
                "matched": matched,
            }
        per_kind[kind] = entry
        if entry["available"]:
            available_kinds.append(kind)
        else:
            na[kind] = entry["reason"]

    if available_kinds:
        macro_f1: float | None = float(
            sum(per_kind[k]["f1"] for k in available_kinds) / len(available_kinds)
        )
    else:
        macro_f1 = None

    return {
        "per_kind": per_kind,
        "macro_f1": macro_f1,
        "matched_total": int(sum(per_kind[k]["matched"] for k in available_kinds)),
        "available_kinds": available_kinds,
        "n_gt_events": int(len(gt_events)),
        "n_pred_events": int(len(pred_events)),
        "na": na,
    }


def evaluate_event_metrics(
    manifest: DatasetManifest | Sequence[PairSample],
    preds: Mapping[int, PairPrediction] | Iterable[PairPrediction],
    *,
    split: str | None = None,
    sample_indices: Iterable[int] | None = None,
    translator: EventTranslator | None = None,
    pred_has_types: bool | None = None,
    before_relations_available: bool = False,
) -> dict[str, Any]:
    """便捷入口：manifest + 预测 → GT/预测事件流 → 事件级指标（L3 报告用）。

    只评估 `split`/`sample_indices` 选中的 pair；选中 pair 缺少预测时明确报错
    （不静默按空预测算），保证指标可追溯。
    """
    pred_map = as_prediction_map(preds)
    pairs = select_pairs(manifest, split=split, sample_indices=sample_indices)
    indices = [p.index for p in pairs]
    missing = [i for i in indices if i not in pred_map]
    if missing:
        raise ValueError(f"缺少 pair 预测：{missing}（事件级指标不做缺省填充）")

    gt_events = translate_gt(
        manifest,
        split=split,
        sample_indices=indices,
        translator=EventTranslator(),
    )
    pred_events = (translator or EventTranslator()).translate(
        pred_map, pair_indices=indices
    )

    if pred_has_types is None:
        pred_has_types = any(bool(pred_map[i].types) for i in indices)

    result = evaluate_events(
        gt_events,
        pred_events,
        pred_has_types=pred_has_types,
        before_relations_available=before_relations_available,
    )
    result["n_pairs"] = len(indices)
    return result


__all__ = [
    "ENTITY_CREATED_TYPE_NA_REASON",
    "evaluate_events",
    "evaluate_event_metrics",
]
