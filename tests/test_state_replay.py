"""L4 世界状态回放单测：WorldStore 语义 + 五维一致率精确数值 + N/A 约定。"""

from __future__ import annotations

from pathlib import Path

import pytest

from starlife_eval.translator import (
    entity_created,
    relation_added,
    relation_removed,
    state_changed,
)
from starlife_eval.types import (
    DatasetManifest,
    Entity,
    PairPrediction,
    PairSample,
    Relation,
    RelationPrediction,
)
from starlife_eval.verifiers.state_replay import (
    ATTRS_NA_REASON,
    DIMENSIONS,
    EMPTY_NA_REASON,
    RELATIONS_NA_REASON,
    TYPES_NA_REASON,
    WorldStore,
    replay_and_compare,
)


# --------------------------------------------------------------------- fixtures


def make_manifest_pairs() -> DatasetManifest:
    pair0 = PairSample(
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
    pair1 = PairSample(
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
    pair2_train = PairSample(
        index=2,
        pair_id="000003",
        split="train",
        before="frames/000003_before.jpg",
        after="frames/000003_after.jpg",
        entities=(Entity(0, (0.0, 0.0, 10.0, 10.0), "lamp", True),),
        changes={},
        relations=(),
    )
    offsets = [0]
    for pair in (pair0, pair1, pair2_train):
        offsets.append(offsets[-1] + len(pair.entities))
    return DatasetManifest(
        root=Path("."),
        pairs=(pair0, pair1, pair2_train),
        meta={"type_keys": [], "rel_types": []},
        slot_offsets=tuple(offsets),
    )


def preds_all_correct(with_types: bool = True) -> dict[int, PairPrediction]:
    return {
        0: PairPrediction(
            pair_index=0,
            changes={1: "displaced", 2: "appeared"},
            types={0: "table", 1: "cup", 2: "bowl"} if with_types else {},
            relations=[
                RelationPrediction(0, 1, "on", "top", True, True),
                RelationPrediction(2, 0, "near", "left", False, False),
            ],
        ),
        1: PairPrediction(
            pair_index=1,
            changes={0: "appeared", 1: "disappeared"},
            types={0: "box", 1: "pen"} if with_types else {},
            relations=[RelationPrediction(0, 1, "near", "front", False, False)],
        ),
    }


# ------------------------------------------------------------------ WorldStore


def test_world_store_apply_semantics() -> None:
    store = WorldStore()
    store.apply(entity_created(1_000_000, "cup", 1_000, pair_index=0, slot=0))
    assert store.entities[1_000_000].present is True
    assert store.entities[1_000_000].entity_type == "cup"

    store.apply(
        state_changed(1_000_000, "disappeared", 1_000, present=False, pair_index=0, slot=0)
    )
    assert store.entities[1_000_000].present is False
    assert store.entities[1_000_000].properties["change"] == "disappeared"

    store.apply(
        relation_added(
            1_000_000, 1_000_001, "on", 1_000,
            properties={"rel_pos": "top", "contact": True, "support": False},
            pair_index=0, from_slot=0, to_slot=1,
        )
    )
    assert (1_000_000, 1_000_001, "on") in store.relations
    assert store.relation_attrs[(1_000_000, 1_000_001, "on")] == {
        "rel_pos": "top",
        "contact": True,
        "support": False,
    }
    store.apply(
        relation_removed(
            1_000_000, 1_000_001, "on", 2_000, pair_index=0, from_slot=0, to_slot=1
        )
    )
    assert store.relations == set()
    assert store.relation_attrs == {}

    snapshot = store.snapshot()
    assert set(snapshot) == {"entities", "relations", "relation_attrs"}
    assert snapshot["entities"][1_000_000]["present"] is False


def test_world_store_state_changed_on_unknown_entity() -> None:
    store = WorldStore()
    store.apply(
        state_changed(1_000_005, "disappeared", 1_000, present=False, pair_index=0, slot=5)
    )
    assert store.entities[1_000_005].present is False
    assert store.entities[1_000_005].entity_type is None


# ---------------------------------------------------------- 五维一致率（数值）


def test_replay_all_correct_state_acc_is_one() -> None:
    manifest = make_manifest_pairs()
    result = replay_and_compare(
        manifest, [0, 1], preds_all_correct(), split="test", pred_has_types=True
    )
    assert result["n_pairs"] == 2
    for dim in DIMENSIONS:
        assert result[dim] == pytest.approx(1.0), dim
    assert result["state_acc"] == pytest.approx(1.0)
    assert result["na"] == {}
    # 逐项汇总：objects 5 个 slot、types 4 个 after 存在实体、changes 5 个 slot、
    # relations GT/pred 各 3 条、attrs 9 项（每个 GT 关系 rel_pos+contact+support）
    assert result["n_ok"] == {
        "objects": 5,
        "types": 4,
        "changes": 5,
        "relations": 3,
        "attrs": 9,
    }
    assert result["n_total"] == {
        "objects": 5,
        "types": 4,
        "changes": 5,
        "relations": 6,
        "attrs": 9,
    }


def test_replay_extra_relation_hurts_relations_only() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    # pair0 预测多一条关系 (1,2,on)：GT 3 条、预测 4 条、命中 3 条 -> F1=2*3/(3+4)=6/7
    preds[0].relations.append(RelationPrediction(1, 2, "on", "right", False, False))
    result = replay_and_compare(manifest, [0, 1], preds, split="test", pred_has_types=True)

    assert result["relations"] == pytest.approx(6 / 7)
    # attrs 只对 GT 关系比较，额外预测关系不在此维扣分
    assert result["attrs"] == pytest.approx(1.0)
    assert result["objects"] == pytest.approx(1.0)
    assert result["changes"] == pytest.approx(1.0)
    assert result["types"] == pytest.approx(1.0)
    # state_acc = (1+1+1+6/7+1)/5 = 34/35
    assert result["state_acc"] == pytest.approx((1.0 + 1.0 + 1.0 + 6 / 7 + 1.0) / 5)
    assert result["state_acc"] == pytest.approx(34 / 35)


def test_replay_wrong_attrs_hurts_attrs_only() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    # pair0 第一条关系 rel_pos 猜错（top -> front）：关系集合不变，仅 attrs 1/9 项错
    preds[0].relations[0] = RelationPrediction(0, 1, "on", "front", True, True)
    result = replay_and_compare(manifest, [0, 1], preds, split="test", pred_has_types=True)

    assert result["relations"] == pytest.approx(1.0)
    assert result["attrs"] == pytest.approx(8 / 9)
    assert result["objects"] == pytest.approx(1.0)
    # state_acc = (1+1+1+1+8/9)/5 = 44/45
    assert result["state_acc"] == pytest.approx(44 / 45)


def test_replay_wrong_type_hurts_types_only() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    preds[0].types[2] = "cup"  # pair0 slot2 GT 是 bowl；after 存在实体共 4 个，错 1 个
    result = replay_and_compare(manifest, [0, 1], preds, split="test", pred_has_types=True)

    assert result["types"] == pytest.approx(3 / 4)
    assert result["objects"] == pytest.approx(1.0)
    assert result["changes"] == pytest.approx(1.0)
    assert result["relations"] == pytest.approx(1.0)
    assert result["attrs"] == pytest.approx(1.0)
    assert result["state_acc"] == pytest.approx(0.95)


def test_replay_missed_disappeared_hurts_objects_and_changes() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    # pair1 slot1 漏掉 disappeared：回放后对象仍存在（objects 错），change 也错
    preds[1].changes = {0: "appeared"}
    result = replay_and_compare(manifest, [0, 1], preds, split="test", pred_has_types=True)

    # objects：5 个 slot 中 pair1 slot1 错 -> 4/5；changes 同理 4/5
    assert result["objects"] == pytest.approx(4 / 5)
    assert result["changes"] == pytest.approx(4 / 5)
    assert result["types"] == pytest.approx(1.0)
    assert result["relations"] == pytest.approx(1.0)
    assert result["attrs"] == pytest.approx(1.0)
    # state_acc = (0.8+1+0.8+1+1)/5 = 0.92
    assert result["state_acc"] == pytest.approx(0.92)


# --------------------------------------------------------------- N/A 与过滤


def test_replay_cloud_without_types_marks_types_na() -> None:
    manifest = make_manifest_pairs()
    result = replay_and_compare(
        manifest,
        [0, 1],
        preds_all_correct(with_types=False),
        split="test",
        pred_has_types=False,
    )
    assert result["types"] is None
    assert result["na"]["types"] == TYPES_NA_REASON
    assert "云端" in result["na"]["types"]
    # 其余四维全对 -> state_acc 只均分 4 个可用维度
    assert result["objects"] == pytest.approx(1.0)
    assert result["changes"] == pytest.approx(1.0)
    assert result["relations"] == pytest.approx(1.0)
    assert result["attrs"] == pytest.approx(1.0)
    assert result["state_acc"] == pytest.approx(1.0)
    assert result["n_ok"] == {
        "objects": 5,
        "types": 0,  # 预测无 type，types 维度 N/A（n_ok 仍记录 0 以便审计）
        "changes": 5,
        "relations": 3,
        "attrs": 9,
    }


def test_replay_no_gt_relations_marks_relations_and_attrs_na() -> None:
    pair = PairSample(
        index=0,
        pair_id="000010",
        split="test",
        before="frames/000010_before.jpg",
        after="frames/000010_after.jpg",
        entities=(Entity(0, (0.0, 0.0, 10.0, 10.0), "lamp", True),),
        changes={},
        relations=(),
    )
    manifest = DatasetManifest(
        root=Path("."), pairs=(pair,), meta={}, slot_offsets=(0, 1)
    )
    preds = {
        0: PairPrediction(pair_index=0, changes={}, types={0: "lamp"}, relations=[])
    }
    result = replay_and_compare(manifest, [0], preds, split="test", pred_has_types=True)

    assert result["objects"] == pytest.approx(1.0)
    assert result["types"] == pytest.approx(1.0)
    assert result["changes"] == pytest.approx(1.0)
    assert result["relations"] is None
    assert result["attrs"] is None
    assert result["na"]["relations"] == RELATIONS_NA_REASON
    assert result["na"]["attrs"] == ATTRS_NA_REASON
    # state_acc 只均分 objects/types/changes
    assert result["state_acc"] == pytest.approx(1.0)


def test_replay_empty_selection_is_all_na() -> None:
    manifest = make_manifest_pairs()
    result = replay_and_compare(
        manifest, [], preds_all_correct(), split="test", pred_has_types=True
    )
    assert result["n_pairs"] == 0
    assert result["state_acc"] is None
    for dim in DIMENSIONS:
        assert result[dim] is None
        assert result["na"][dim] == EMPTY_NA_REASON


def test_replay_only_sample_indices_participate() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    # 额外给 train pair 一个错误预测：不应参与（sample 只含 0/1）
    preds[2] = PairPrediction(
        pair_index=2, changes={0: "disappeared"}, types={0: "wrong"}, relations=[]
    )
    result = replay_and_compare(
        manifest, [0, 1], preds, split="test", pred_has_types=True
    )
    assert result["n_pairs"] == 2
    assert result["state_acc"] == pytest.approx(1.0)


def test_replay_missing_prediction_raises() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    del preds[1]
    with pytest.raises(ValueError, match="缺少 pair 预测"):
        replay_and_compare(manifest, [0, 1], preds, split="test", pred_has_types=True)
