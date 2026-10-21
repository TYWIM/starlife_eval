"""L1 事件契约翻译器单测：用构造张量断言事件条数/类型/时间戳/ID 复用/去重/N/A 语义。"""

from __future__ import annotations

from pathlib import Path

import pytest

from starlife_eval.translator import (
    DEFAULT_TS_START_MS,
    DEFAULT_TS_STEP_MS,
    ENTITY_ID_BASE,
    EventTranslator,
    WorldEvent,
    entity_created,
    relation_added,
    state_changed,
    translate_gt,
    validate_event_stream,
)
from starlife_eval.types import (
    DatasetManifest,
    Entity,
    PairPrediction,
    PairSample,
    Relation,
    RelationPrediction,
)


# --------------------------------------------------------------------- fixtures


def make_manifest(pairs: list[PairSample]) -> DatasetManifest:
    offsets = [0]
    for pair in pairs:
        offsets.append(offsets[-1] + len(pair.entities))
    return DatasetManifest(
        root=Path("."), pairs=tuple(pairs), meta={"type_keys": [], "rel_types": []},
        slot_offsets=tuple(offsets),
    )


def pair0() -> PairSample:
    """test pair 0：slot1 位移，slot2 出现；两条 after 关系。"""
    return PairSample(
        index=0,
        pair_id="000001",
        split="test",
        before="frames/000001_before.jpg",
        after="frames/000001_after.jpg",
        entities=(
            Entity(0, (0.0, 0.0, 100.0, 100.0), "table", True),
            Entity(1, (100.0, 100.0, 200.0, 200.0), "cup", True),
            Entity(2, (200.0, 200.0, 300.0, 300.0), "bowl", True),
        ),
        changes={1: "displaced", 2: "appeared"},
        relations=(
            Relation(0, 1, "on", "top", True, True, 0.1),
            Relation(2, 0, "near", "left", False, False, 0.4),
        ),
    )


def pair1() -> PairSample:
    """test pair 1：slot0 出现，slot1 消失；一条 after 关系。"""
    return PairSample(
        index=1,
        pair_id="000002",
        split="test",
        before="frames/000002_before.jpg",
        after="frames/000002_after.jpg",
        entities=(
            Entity(0, (0.0, 0.0, 50.0, 50.0), "box", True),
            Entity(1, (50.0, 50.0, 80.0, 80.0), "pen", False),
        ),
        changes={0: "appeared", 1: "disappeared"},
        relations=(Relation(0, 1, "near", "front", False, False, 0.3),),
    )


def make_manifest_2pairs() -> DatasetManifest:
    return make_manifest([pair0(), pair1()])


def pred0() -> PairPrediction:
    """与 pair0 GT 完全一致的预测（含类型）。"""
    return PairPrediction(
        pair_index=0,
        changes={1: "displaced", 2: "appeared"},
        types={0: "table", 1: "cup", 2: "bowl"},
        relations=[
            RelationPrediction(0, 1, "on", "top", True, True),
            RelationPrediction(2, 0, "near", "left", False, False),
        ],
    )


def pred1() -> PairPrediction:
    return PairPrediction(
        pair_index=1,
        changes={0: "appeared", 1: "disappeared"},
        types={0: "box", 1: "pen"},
        relations=[RelationPrediction(0, 1, "near", "front", False, False)],
    )


def signature(ev: WorldEvent) -> tuple:
    """事件的可比语义签名（ID 分配顺序无关，用于 GT/预测对称性断言）。"""
    if ev.kind == "StateChanged":
        return ("StateChanged", ev.pair_index, ev.slot, ev.properties["change"])
    if ev.kind == "EntityCreated":
        return ("EntityCreated", ev.pair_index, ev.slot, ev.entity_type)
    return (ev.kind, ev.pair_index, ev.from_slot, ev.to_slot, ev.relation_type)


def kinds(events: list[WorldEvent]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ev in events:
        out[ev.kind] = out.get(ev.kind, 0) + 1
    return out


# ------------------------------------------------------------------- ID 分配


def test_entity_id_allocation_starts_at_base_and_is_stable() -> None:
    tr = EventTranslator()
    eid_0_0 = tr.entity_id(0, 0)
    assert eid_0_0 == ENTITY_ID_BASE == 1_000_000
    # 同一 (pair, slot) 稳定复用
    assert tr.entity_id(0, 0) == eid_0_0
    assert tr.entity_id(0, 1) == ENTITY_ID_BASE + 1
    assert tr.entity_id(0, 0) == eid_0_0
    # pair 之间不混用
    eid_1_0 = tr.entity_id(1, 0)
    assert eid_1_0 == ENTITY_ID_BASE + 2
    assert eid_1_0 != eid_0_0
    # 反向映射
    assert tr.entity_slot(eid_1_0) == (1, 0)
    # reset 后从 base 重新开始
    tr.reset()
    assert tr.entity_id(0, 0) == ENTITY_ID_BASE


# ------------------------------------------------------------- 预测 → 事件流


def test_translate_pair_event_counts_types_and_timestamps() -> None:
    tr = EventTranslator()
    events = tr.translate([pred0(), pred1()])

    # pair0: slot1 displaced(1) + slot2 appeared(created+state=2) + 2 relation = 5
    # pair1: slot0 appeared(2) + slot1 disappeared(1) + 1 relation = 4
    assert len(events) == 9
    assert kinds(events) == {
        "EntityCreated": 2,
        "StateChanged": 4,
        "RelationAdded": 3,
    }
    assert kinds(events).get("RelationRemoved", 0) == 0

    # 时间戳：pair0 同一帧、pair1 后一帧（Unix ms 单调不减）
    ts0 = [ev.ts_ms for ev in events if ev.pair_index == 0]
    ts1 = [ev.ts_ms for ev in events if ev.pair_index == 1]
    assert set(ts0) == {DEFAULT_TS_START_MS}
    assert set(ts1) == {DEFAULT_TS_START_MS + DEFAULT_TS_STEP_MS}
    assert ts0[0] < ts1[0]
    validate_event_stream(events)  # 不抛异常即通过

    created = [ev for ev in events if ev.kind == "EntityCreated"]
    assert {(ev.pair_index, ev.slot, ev.entity_type) for ev in created} == {
        (0, 2, "bowl"),
        (1, 0, "box"),
    }
    # EntityCreated payload 严格 {entity_id, type}，且不编造坐标
    for ev in created:
        payload = ev.to_payload()
        assert set(payload) == {"entity_id", "type"}
        assert isinstance(ev.ts_ms, int)
    # StateChanged payload 严格 {entity_id, valid_from, properties}
    for ev in (e for e in events if e.kind == "StateChanged"):
        payload = ev.to_payload()
        assert set(payload) == {"entity_id", "valid_from", "properties"}
        assert payload["valid_from"] == ev.ts_ms
        assert isinstance(payload["properties"]["present"], bool)
        assert set(payload) >= {"entity_id", "valid_from", "properties"}
    # RelationAdded payload 严格 {from,to,relation_type,valid_from,properties}
    for ev in (e for e in events if e.kind == "RelationAdded"):
        payload = ev.to_payload()
        assert set(payload) == {"from", "to", "relation_type", "valid_from", "properties"}
        assert payload["valid_from"] == ev.ts_ms


def test_change_semantics_present_flag() -> None:
    tr = EventTranslator()
    pred = PairPrediction(
        pair_index=7,
        changes={0: "appeared", 1: "displaced", 2: "disappeared"},
        types={0: "x"},
        relations=[],
    )
    events = tr.translate_pair(pred)
    assert kinds(events) == {"EntityCreated": 1, "StateChanged": 3}
    by_slot = {ev.slot: ev for ev in events if ev.kind == "StateChanged"}
    assert by_slot[0].properties == {"present": True, "change": "appeared"}
    assert by_slot[1].properties == {"present": True, "change": "displaced"}
    assert by_slot[2].properties == {"present": False, "change": "disappeared"}


def test_same_frame_relation_dedupe() -> None:
    tr = EventTranslator()
    pred = PairPrediction(
        pair_index=0,
        changes={},
        types={},
        relations=[
            RelationPrediction(0, 1, "on", "top", True, True),
            RelationPrediction(0, 1, "on", "top", True, True),  # 同帧重复
            RelationPrediction(0, 1, "near", "left", False, False),
        ],
    )
    events = tr.translate_pair(pred)
    assert kinds(events) == {"RelationAdded": 2}


def test_relation_removed_requires_previous_snapshot() -> None:
    # v2 单帧对（无前序快照）→ 不产生 RelationRemoved
    tr = EventTranslator()
    first = tr.translate_pair(pred0())
    assert kinds(first).get("RelationRemoved", 0) == 0

    # 同一 pair 的“下一帧快照”少了 (2,0,near) → 产生 RelationRemoved
    next_pred = PairPrediction(
        pair_index=0,
        changes={},
        types={},
        relations=[RelationPrediction(0, 1, "on", "top", True, True)],
    )
    second = tr.translate_pair(next_pred)
    removed = [ev for ev in second if ev.kind == "RelationRemoved"]
    assert len(removed) == 1
    ev = removed[0]
    assert (ev.from_slot, ev.to_slot, ev.relation_type) == (2, 0, "near")
    assert ev.ts_ms == DEFAULT_TS_START_MS + DEFAULT_TS_STEP_MS  # 下一帧
    assert set(ev.to_payload()) == {"from", "to", "relation_type", "valid_at"}
    assert ev.to_payload()["valid_at"] == ev.ts_ms

    # 显式 previous_relations 也能构造（L1 契约单测用）
    tr2 = EventTranslator()
    events = tr2.translate_pair(
        next_pred, previous_relations=[(0, 1, "on"), (2, 0, "near")]
    )
    assert kinds(events)["RelationRemoved"] == 1


# ------------------------------------------------------------------ GT 对称性


def test_translate_gt_is_symmetric_with_prediction() -> None:
    manifest = make_manifest_2pairs()
    gt_events = translate_gt(manifest, split="test", translator=EventTranslator())
    pred_events = EventTranslator().translate([pred0(), pred1()])

    assert kinds(gt_events) == kinds(pred_events) == {
        "EntityCreated": 2,
        "StateChanged": 4,
        "RelationAdded": 3,
    }
    assert [signature(ev) for ev in gt_events] == [signature(ev) for ev in pred_events]
    # v2 GT 没有 before 关系 → 永远没有 RelationRemoved
    assert not [ev for ev in gt_events if ev.kind == "RelationRemoved"]
    assert [ev.ts_ms for ev in gt_events] == [ev.ts_ms for ev in pred_events]


def test_translate_gt_only_lists_non_none_changes() -> None:
    manifest = make_manifest_2pairs()
    gt_events = translate_gt(manifest, split="test", translator=EventTranslator())
    changed_slots = {
        (ev.pair_index, ev.slot)
        for ev in gt_events
        if ev.kind in ("StateChanged", "EntityCreated")
    }
    # pair0: slot1 displaced, slot2 appeared; pair1: slot0 appeared, slot1 disappeared
    assert changed_slots == {(0, 1), (0, 2), (1, 0), (1, 1)}
    # 未变化的 slot（pair0 slot0 / pair1 无）不发事件
    assert (0, 0) not in changed_slots


def test_translate_gt_relation_removed_from_previous_snapshot() -> None:
    manifest = make_manifest_2pairs()
    gt_events = translate_gt(
        manifest,
        split="test",
        translator=EventTranslator(),
        previous_relations={0: [(1, 2, "held_by")], 1: []},
    )
    removed = [ev for ev in gt_events if ev.kind == "RelationRemoved"]
    assert len(removed) == 1
    assert (removed[0].pair_index, removed[0].from_slot, removed[0].to_slot) == (0, 1, 2)
    assert removed[0].relation_type == "held_by"


# ------------------------------------------------------------ 流级校验与错误


def test_validate_event_stream_rejects_timestamp_regression() -> None:
    events = [
        state_changed(1_000_000, "appeared", 2_000, present=True, pair_index=0, slot=0),
        state_changed(1_000_000, "displaced", 1_999, present=True, pair_index=0, slot=1),
    ]
    with pytest.raises(ValueError, match="时间戳回退"):
        validate_event_stream(events)


def test_validate_event_stream_rejects_duplicate_same_frame() -> None:
    events = [
        state_changed(1_000_000, "appeared", 1_000, present=True, pair_index=0, slot=0),
        state_changed(1_000_000, "appeared", 1_000, present=True, pair_index=0, slot=0),
    ]
    with pytest.raises(ValueError, match="同帧重复"):
        validate_event_stream(events)


def test_translate_rejects_missing_pair_prediction() -> None:
    tr = EventTranslator()
    with pytest.raises(ValueError, match="缺少 pair 预测"):
        tr.translate({0: pred0()}, pair_indices=[0, 1])


def test_world_event_rejects_invalid_change_label() -> None:
    with pytest.raises(ValueError, match="change"):
        state_changed(
            1_000_000, "none", 1_000, present=True, pair_index=0, slot=0
        )


def test_relation_added_and_entity_created_factories_payload() -> None:
    ev = entity_created(1_000_000, "cup", 1_000, pair_index=0, slot=1)
    assert ev.to_payload() == {"entity_id": 1_000_000, "type": "cup"}
    rel = relation_added(
        1_000_000, 1_000_001, "on", 1_000,
        properties={"rel_pos": "top", "contact": True},
        pair_index=0, from_slot=0, to_slot=1,
    )
    assert rel.to_payload() == {
        "from": 1_000_000,
        "to": 1_000_001,
        "relation_type": "on",
        "valid_from": 1_000,
        "properties": {"rel_pos": "top", "contact": True},
    }
