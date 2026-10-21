"""L4 世界状态回放：把预测事件回放进最小 `WorldStore`，与 GT after 场景比较。

口径来源：`starlife_eval/ARCHITECTURE.md` §5.4、`测试套件设计.md` §3「state_acc」与 L4。
（只读参考 `starlife_store/Core/{WorldEvent,WorldStore,WorldGraph}.cs` 的语义，
用 Python 最小实现，不 import .NET 代码。）

回放语义（与 v2 数据现实对齐）：
- 每个 pair 从 **before 帧世界状态** 开始回放（v2 manifest 只有 after 的 `present_after`，
  before 存在性由 `present_after + change` 语义还原：appeared 的实体 before 不存在，
  disappeared/其它实体 before 存在）；
- 然后顺序 `WorldStore.apply` 该 pair 的预测事件（`translator.EventTranslator`），
  得到 after 快照；
- GT after 快照直接由 manifest 构造（present_after 实体集合、type_key、changes、
  relations 集合/类型、rel_pos/contact/support）。

五个维度按“逐项一致率”计算（与头级/协议指标同量级，避免逐 pair 全等被实体数放大惩罚），
先在选中 pair 上汇总命中/总数再相除，state_acc = 可用维度等权平均：
- `objects`  : 每个 manifest slot 的 after 存在性是否与 GT `present_after` 一致（逐 slot 命中率）；
- `types`    : after 存在实体的 type 命中率（**仅当预测侧有类型时可用**）；
- `changes`  : 逐 slot change 标签（none/appeared/displaced/disappeared）命中率；
- `relations`: (from_slot,to_slot,type) 集合的 F1 `2*matched/(|GT|+|pred|)`（多报/漏报都扣）；
- `attrs`    : GT 关系上的 rel_pos/contact/support 逐项命中率（额外预测关系不在此维扣分）。

`state_acc` = 可用维度等权平均；不可用维度（types/attrs/relations 视数据而定）进 `na`
并写明原因，绝不填 0 分或编造。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..translator import (
    EventTranslator,
    WorldEvent,
    as_prediction_map,
    select_pairs,
)
from ..types import DatasetManifest, PairPrediction, PairSample

#: 云端零样本预测没有实体类型 → types 维度不可测。
TYPES_NA_REASON: str = (
    "预测侧没有实体类型（云端零样本预测只返回 changes/relations，不返回 type）；"
    "types 维度仅本地头可测，按 N/A 处理，不参与 state_acc。"
)
#: after 帧没有 GT 关系 → attrs 维度没有可比对象。
ATTRS_NA_REASON: str = (
    "该子集 after 帧没有 GT 关系，attrs(rel_pos/contact/support) 没有可比对象，"
    "按 N/A 处理，不参与 state_acc。"
)
#: GT 与预测都没有关系 → relations 维度无定义（不虚构成 1.0）。
RELATIONS_NA_REASON: str = (
    "该子集 after 帧没有 GT 关系且预测也没有关系，relations 维度没有可比对象，"
    "按 N/A 处理，不参与 state_acc。"
)
#: GT after 没有存在实体 → types 维度没有可比对象。
EMPTY_TYPES_NA_REASON: str = (
    "该子集 GT after 没有存在实体（present_after 全为 false），types 维度没有可比对象，"
    "按 N/A 处理，不参与 state_acc。"
)
#: 某个维度在 GT 与预测里都没有可比对象（如没有实体 slot）。
EMPTY_ITEMS_NA_REASON: str = (
    "该子集 {dim} 维度没有可比对象，按 N/A 处理，不参与 state_acc。"
)
#: 过滤后没有 pair。
EMPTY_NA_REASON: str = "没有可评估的 pair（split/sample_indices 过滤后为空），所有维度 N/A。"

DIMENSIONS: tuple[str, ...] = ("objects", "types", "changes", "relations", "attrs")


# --------------------------------------------------------------------- WorldStore


@dataclass
class EntityRecord:
    """WorldStore 中的实体投影（present 对应 StateChanged 的 properties.present）。"""

    present: bool = True
    entity_type: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


class WorldStore:
    """最小世界状态存储：只做内存投影，`apply(event)` 顺序回放。"""

    def __init__(self) -> None:
        self.entities: dict[int, EntityRecord] = {}
        self.relations: set[tuple[int, int, str]] = set()
        self.relation_attrs: dict[tuple[int, int, str], dict[str, Any]] = {}

    # -- 回放前 seeding（before 帧世界）---------------------------------
    def seed(
        self,
        entity_id: int,
        *,
        present: bool,
        entity_type: str | None = None,
    ) -> None:
        record = self.entities.get(entity_id)
        if record is None:
            self.entities[entity_id] = EntityRecord(
                present=bool(present), entity_type=entity_type
            )
        else:
            record.present = bool(present)
            if entity_type is not None:
                record.entity_type = entity_type

    # -- 事件投影 --------------------------------------------------------
    def apply(self, event: WorldEvent) -> None:
        if event.kind == "EntityCreated":
            assert event.entity_id is not None
            record = self.entities.get(event.entity_id)
            if record is None:
                record = EntityRecord(present=True)
                self.entities[event.entity_id] = record
            # 实体注册即存在；后续 StateChanged{present:...} 可改写
            record.present = True
            record.entity_type = event.entity_type
            return
        if event.kind == "StateChanged":
            assert event.entity_id is not None
            record = self.entities.get(event.entity_id)
            if record is None:
                record = EntityRecord(present=False)
                self.entities[event.entity_id] = record
            props = dict(event.properties)
            record.properties.update(props)
            if "present" in props:
                record.present = bool(props["present"])
            return
        if event.kind == "RelationAdded":
            assert event.from_id is not None and event.to_id is not None
            key = (event.from_id, event.to_id, str(event.relation_type))
            self.relations.add(key)
            self.relation_attrs[key] = {
                k: event.properties[k]
                for k in ("rel_pos", "contact", "support")
                if k in event.properties
            }
            return
        if event.kind == "RelationRemoved":
            assert event.from_id is not None and event.to_id is not None
            key = (event.from_id, event.to_id, str(event.relation_type))
            self.relations.discard(key)
            self.relation_attrs.pop(key, None)
            return
        raise ValueError(f"非法事件类型 {event.kind!r}")

    # -- 查询 ------------------------------------------------------------
    def present_entity_ids(self) -> set[int]:
        return {eid for eid, rec in self.entities.items() if rec.present}

    def snapshot(self) -> dict[str, Any]:
        return {
            "entities": {
                eid: {
                    "present": rec.present,
                    "type": rec.entity_type,
                    "properties": dict(rec.properties),
                }
                for eid, rec in sorted(self.entities.items())
            },
            "relations": sorted(self.relations),
            "relation_attrs": {
                key: dict(value) for key, value in sorted(self.relation_attrs.items())
            },
        }


# --------------------------------------------------------------------- helpers


def _before_present(pair: PairSample, slot: int) -> bool:
    """v2 manifest 只有 after 的 present_after；按 change 语义还原 before 存在性。"""
    change = pair.changes.get(slot)
    if change == "appeared":
        return False
    if change == "disappeared":
        return True
    # none / displaced：after 与 before 都存在
    return bool(pair.entity(slot).present_after)


def _pred_changes_norm(pred: PairPrediction, slots: Sequence[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for slot in slots:
        value = pred.changes.get(slot)
        out[int(slot)] = "none" if value is None else str(value)
    return out


def _gt_changes(pair: PairSample) -> dict[int, str]:
    return {int(e.slot): str(pair.changes.get(e.slot, "none")) for e in pair.entities}


def _gt_snapshot(pair: PairSample) -> dict[str, Any]:
    present_slots = {int(e.slot) for e in pair.entities if e.present_after}
    return {
        "objects": present_slots,
        "types": {int(e.slot): e.type_key for e in pair.entities if e.present_after},
        "changes": _gt_changes(pair),
        "relations": {
            (int(r.from_slot), int(r.to_slot), str(r.type)) for r in pair.relations
        },
        "attrs": {
            (int(r.from_slot), int(r.to_slot)): (
                r.rel_pos,
                bool(r.contact),
                bool(r.support),
            )
            for r in pair.relations
        },
    }


def _pair_stats(
    pair: PairSample,
    pred: PairPrediction,
    translator: EventTranslator,
) -> dict[str, int | bool]:
    """回放一个 pair 的预测事件，返回五维的“命中/总数”计数（供全局一致率汇总）。"""
    pid = int(pair.index)
    slots = [int(e.slot) for e in pair.entities]
    gt = _gt_snapshot(pair)
    gt_present = {int(e.slot): bool(e.present_after) for e in pair.entities}

    # 1) before 世界 seeding + 顺序回放预测事件
    store = WorldStore()
    for entity in pair.entities:
        slot = int(entity.slot)
        store.seed(
            translator.entity_id(pid, slot),
            present=_before_present(pair, slot),
            entity_type=None,  # GT 类型不能泄漏到预测侧；类型由 EntityCreated 事件给出
        )
    for event in translator.translate_pair(pred, pair_index=pid):
        store.apply(event)

    # 2) 预测 after 快照（存在集合 / 关系 / 关系属性）
    pred_present = {
        slot: store.entities.get(
            translator.entity_id(pid, slot), EntityRecord(present=False)
        ).present
        for slot in slots
    }
    pred_relations: set[tuple[int, int, str]] = set()
    pred_attrs: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
    for from_id, to_id, relation_type in store.relations:
        frm = translator.entity_slot(from_id)[1]
        to = translator.entity_slot(to_id)[1]
        pred_relations.add((frm, to, relation_type))
        attrs = store.relation_attrs.get((from_id, to_id, relation_type), {})
        pred_attrs[(frm, to)] = (
            attrs.get("rel_pos"),
            attrs.get("contact"),
            attrs.get("support"),
        )

    # 3) 五维逐项统计（与 test split 上对应头级指标同量级的“逐项一致率”）
    objects_ok = sum(pred_present[slot] == gt_present[slot] for slot in slots)
    objects_total = len(slots)

    pred_changes = _pred_changes_norm(pred, slots)
    changes_ok = sum(pred_changes[slot] == gt["changes"][slot] for slot in slots)
    changes_total = len(slots)

    # types：只比较 after 存在实体的类型（消失实体不参与）
    types_ok = sum(
        pred.types.get(slot) == pair.entity(slot).type_key for slot in gt["types"]
    )
    types_total = len(gt["types"])

    # relations：(from_slot,to_slot,type) 集合的匹配数；多报/漏报都进 P/R
    relations_matched = len(gt["relations"] & pred_relations)
    relations_gt = len(gt["relations"])
    relations_pred = len(pred_relations)

    # attrs：只看 GT 关系上的 rel_pos/contact/support；缺失关系该关系的属性全记错
    attrs_ok = 0
    attrs_total = 0
    for relation in pair.relations:
        key = (int(relation.from_slot), int(relation.to_slot))
        pred_attr = pred_attrs.get(key)
        pred_rel_pos, pred_contact, pred_support = pred_attr or (None, None, None)
        if relation.rel_pos is not None:
            attrs_total += 1
            attrs_ok += int(pred_rel_pos == relation.rel_pos)
        attrs_total += 2
        attrs_ok += int(pred_contact is not None and bool(pred_contact) == bool(relation.contact))
        attrs_ok += int(pred_support is not None and bool(pred_support) == bool(relation.support))

    return {
        "objects_ok": int(objects_ok),
        "objects_total": int(objects_total),
        "types_ok": int(types_ok),
        "types_total": int(types_total),
        "changes_ok": int(changes_ok),
        "changes_total": int(changes_total),
        "relations_matched": int(relations_matched),
        "relations_gt": int(relations_gt),
        "relations_pred": int(relations_pred),
        "attrs_ok": int(attrs_ok),
        "attrs_total": int(attrs_total),
        "has_pred_types": bool(pred.types),
    }


# ----------------------------------------------------------------------- replay


def replay(
    preds: Mapping[int, PairPrediction] | Iterable[PairPrediction],
    manifest: DatasetManifest | Sequence[PairSample],
    *,
    split: str | None = None,
    sample_indices: Iterable[int] | None = None,
    translator: EventTranslator | None = None,
    pred_has_types: bool | None = None,
) -> dict[str, Any]:
    """L4 世界状态回放：返回五维一致率 + state_acc + N/A 说明。

    Args:
        preds: `dict[pair_index, PairPrediction]` 或可迭代预测；选中 pair 缺失时明确报错。
        manifest: `DatasetManifest` 或 `Sequence[PairSample]`。
        split / sample_indices: 只评估该 split / 这些 pair 下标（与事件级指标一致）。
        translator: 复用外部翻译器（ID/时间戳基准）；缺省新建。
        pred_has_types: 预测侧是否有实体类型；`None` 时按预测数据是否为空自动判定。
    """
    pred_map = as_prediction_map(preds)
    pairs = select_pairs(manifest, split=split, sample_indices=sample_indices)
    indices = [p.index for p in pairs]
    missing = [i for i in indices if i not in pred_map]
    if missing:
        raise ValueError(f"缺少 pair 预测：{missing}（状态回放不做缺省填充）")

    if not pairs:
        return {
            **{dim: None for dim in DIMENSIONS},
            "state_acc": None,
            "na": {dim: EMPTY_NA_REASON for dim in DIMENSIONS},
            "n_pairs": 0,
            "n_ok": {},
        }

    tr = translator if translator is not None else EventTranslator()
    stats = [_pair_stats(pair, pred_map[pair.index], tr) for pair in pairs]

    objects_ok = sum(s["objects_ok"] for s in stats)
    objects_total = sum(s["objects_total"] for s in stats)
    types_ok = sum(s["types_ok"] for s in stats)
    types_total = sum(s["types_total"] for s in stats)
    changes_ok = sum(s["changes_ok"] for s in stats)
    changes_total = sum(s["changes_total"] for s in stats)
    relations_matched = sum(s["relations_matched"] for s in stats)
    relations_gt = sum(s["relations_gt"] for s in stats)
    relations_pred = sum(s["relations_pred"] for s in stats)
    attrs_ok = sum(s["attrs_ok"] for s in stats)
    attrs_total = sum(s["attrs_total"] for s in stats)

    if pred_has_types is None:
        pred_has_types = any(pred_map[i].types for i in indices)

    available: dict[str, bool] = {
        "objects": objects_total > 0,
        "types": bool(pred_has_types) and types_total > 0,
        "changes": changes_total > 0,
        "relations": (relations_gt + relations_pred) > 0,
        "attrs": attrs_total > 0,
    }
    na: dict[str, str] = {}
    if not available["types"]:
        na["types"] = (
            TYPES_NA_REASON if not pred_has_types else EMPTY_TYPES_NA_REASON
        )
    if not available["relations"]:
        na["relations"] = RELATIONS_NA_REASON
    if not available["attrs"]:
        na["attrs"] = ATTRS_NA_REASON
    for dim in ("objects", "changes"):
        if not available[dim]:
            na[dim] = EMPTY_ITEMS_NA_REASON.format(dim=dim)

    n_pairs = len(pairs)
    relation_denominator = relations_gt + relations_pred
    values: dict[str, float | None] = {
        "objects": objects_ok / objects_total if available["objects"] else None,
        "types": types_ok / types_total if available["types"] else None,
        "changes": changes_ok / changes_total if available["changes"] else None,
        "relations": (
            2.0 * relations_matched / relation_denominator
            if available["relations"]
            else None
        ),
        "attrs": attrs_ok / attrs_total if available["attrs"] else None,
    }
    n_ok = {
        "objects": int(objects_ok),
        "types": int(types_ok),
        "changes": int(changes_ok),
        "relations": int(relations_matched),
        "attrs": int(attrs_ok),
    }
    n_total = {
        "objects": int(objects_total),
        "types": int(types_total),
        "changes": int(changes_total),
        "relations": int(relation_denominator),
        "attrs": int(attrs_total),
    }

    available_values = [values[dim] for dim in DIMENSIONS if values[dim] is not None]
    state_acc = (
        float(sum(available_values) / len(available_values))
        if available_values
        else None
    )

    return {
        **values,
        "state_acc": state_acc,
        "na": na,
        "n_pairs": n_pairs,
        "n_ok": n_ok,
        "n_total": n_total,
    }


def replay_and_compare(
    manifest: DatasetManifest | Sequence[PairSample],
    sample_indices: Iterable[int],
    preds: Mapping[int, PairPrediction] | Iterable[PairPrediction],
    *,
    split: str | None = None,
    translator: EventTranslator | None = None,
    pred_has_types: bool | None = None,
) -> dict[str, Any]:
    """runner 接口（与 `verifiers/event_metrics.evaluate_event_metrics` 顺序一致）。

    参数顺序为 (manifest, sample_indices, preds)；仅 `sample_indices` 里的 pair 参与，
    返回扁平五维键 + state_acc + na + n_pairs。
    """
    return replay(
        preds,
        manifest,
        split=split,
        sample_indices=sample_indices,
        translator=translator,
        pred_has_types=pred_has_types,
    )


__all__ = [
    "TYPES_NA_REASON",
    "ATTRS_NA_REASON",
    "RELATIONS_NA_REASON",
    "EMPTY_NA_REASON",
    "EMPTY_TYPES_NA_REASON",
    "EMPTY_ITEMS_NA_REASON",
    "DIMENSIONS",
    "EntityRecord",
    "WorldStore",
    "replay",
    "replay_and_compare",
]
