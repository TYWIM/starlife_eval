"""L3 事件级指标单测：用手算 P/R/F1 精确断言，并验证 N/A 约定。"""

from __future__ import annotations

from pathlib import Path

import pytest

from starlife_eval.translator import (
    entity_created,
    relation_added,
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
from starlife_eval.verifiers.event_metrics import (
    ENTITY_CREATED_TYPE_NA_REASON,
    evaluate_event_metrics,
    evaluate_events,
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
    types0 = {0: "table", 1: "cup", 2: "bowl"} if with_types else {}
    types1 = {0: "box", 1: "pen"} if with_types else {}
    return {
        0: PairPrediction(
            pair_index=0,
            changes={1: "displaced", 2: "appeared"},
            types=dict(types0),
            relations=[
                RelationPrediction(0, 1, "on", "top", True, True),
                RelationPrediction(2, 0, "near", "left", False, False),
            ],
        ),
        1: PairPrediction(
            pair_index=1,
            changes={0: "appeared", 1: "disappeared"},
            types=dict(types1),
            relations=[RelationPrediction(0, 1, "near", "front", False, False)],
        ),
    }


# --------------------------------------------------- 手算 P/R/F1（合成事件流）


def _synthetic_events() -> tuple[list, list]:
    ts = 1_000
    gt = [
        entity_created(1_000_000, "cup", ts, pair_index=0, slot=0),
        state_changed(1_000_000, "appeared", ts, present=True, pair_index=0, slot=0),
        state_changed(1_000_001, "displaced", ts, present=True, pair_index=0, slot=1),
        relation_added(
            1_000_000, 1_000_001, "on", ts, pair_index=0, from_slot=0, to_slot=1
        ),
        relation_added(
            1_000_001, 1_000_000, "near", ts, pair_index=0, from_slot=1, to_slot=0
        ),
    ]
    # 预测多了一条关系 (2,0,near) 且其余全对：
    #   RelationAdded: support=2, predicted=3, matched=2 -> P=2/3, R=1, F1=0.8
    pred = gt + [
        relation_added(
            1_000_002, 1_000_000, "near", ts, pair_index=0, from_slot=2, to_slot=0
        )
    ]
    return gt, pred


def test_hand_computed_event_prf_and_macro() -> None:
    gt, pred = _synthetic_events()
    result = evaluate_events(gt, pred, pred_has_types=True)

    rel = result["per_kind"]["RelationAdded"]
    assert rel["available"] is True
    assert rel["support"] == 2
    assert rel["predicted"] == 3
    assert rel["matched"] == 2
    assert rel["precision"] == pytest.approx(2 / 3)
    assert rel["recall"] == pytest.approx(1.0)
    assert rel["f1"] == pytest.approx(0.8)

    # StateChanged / EntityCreated 全对
    assert result["per_kind"]["StateChanged"]["f1"] == pytest.approx(1.0)
    assert result["per_kind"]["EntityCreated"]["f1"] == pytest.approx(1.0)

    # v2 无 before 关系标注 -> RelationRemoved 恒 N/A，且不参与 macro
    removed = result["per_kind"]["RelationRemoved"]
    assert removed["available"] is False
    assert removed["precision"] is None and removed["f1"] is None
    assert "before 关系标注" in removed["reason"]
    assert "RelationRemoved" not in result["available_kinds"]
    assert result["available_kinds"] == [
        "EntityCreated",
        "StateChanged",
        "RelationAdded",
    ]

    # macro 只对 available 类等权平均：(1 + 1 + 0.8) / 3
    assert result["macro_f1"] == pytest.approx((1.0 + 1.0 + 0.8) / 3)
    assert result["matched_total"] == 5
    # id_switch_rate 不可测 -> N/A 原因
    assert "id_switch_rate" in result["na"]
    assert "连续帧" in result["na"]["id_switch_rate"]
    assert result["na"]["RelationRemoved"] == removed["reason"]


def test_change_label_mismatch_precision_recall() -> None:
    ts = 1_000
    gt = [
        state_changed(1_000_000, "appeared", ts, present=True, pair_index=0, slot=0),
        state_changed(1_000_001, "displaced", ts, present=True, pair_index=0, slot=1),
        state_changed(1_000_002, "disappeared", ts, present=False, pair_index=0, slot=2),
    ]
    # slot2 猜成 displaced：support=3, predicted=3, matched=2 -> P=R=2/3, F1=2/3
    pred = [
        gt[0],
        gt[1],
        state_changed(1_000_002, "displaced", ts, present=True, pair_index=0, slot=2),
    ]
    result = evaluate_events(gt, pred, pred_has_types=True)
    sc = result["per_kind"]["StateChanged"]
    assert (sc["support"], sc["predicted"], sc["matched"]) == (3, 3, 2)
    assert sc["precision"] == pytest.approx(2 / 3)
    assert sc["recall"] == pytest.approx(2 / 3)
    assert sc["f1"] == pytest.approx(2 / 3)


def test_relation_type_mismatch_uses_type_in_match_key() -> None:
    ts = 1_000
    gt = [
        relation_added(
            1_000_000, 1_000_001, "on", ts, pair_index=0, from_slot=0, to_slot=1
        )
    ]
    pred = [
        relation_added(
            1_000_000, 1_000_001, "near", ts, pair_index=0, from_slot=0, to_slot=1
        )
    ]
    result = evaluate_events(gt, pred, pred_has_types=True)
    rel = result["per_kind"]["RelationAdded"]
    assert (rel["support"], rel["predicted"], rel["matched"]) == (1, 1, 0)
    assert rel["f1"] == 0.0
    # GT 无 EntityCreated/StateChanged、预测也没有 -> N/A；RelationAdded 有事件 -> available
    assert rel["available"] is True
    assert result["macro_f1"] == pytest.approx(0.0)


def test_no_predictions_gives_zero_but_available_kind() -> None:
    gt, _ = _synthetic_events()
    result = evaluate_events(gt, [], pred_has_types=True)
    rel = result["per_kind"]["RelationAdded"]
    assert rel["available"] is True
    assert (rel["predicted"], rel["matched"]) == (0, 0)
    assert rel["precision"] == 0.0 and rel["f1"] == 0.0


def test_entity_created_na_when_prediction_has_no_types() -> None:
    gt, pred = _synthetic_events()
    result = evaluate_events(gt, pred, pred_has_types=False)
    created = result["per_kind"]["EntityCreated"]
    assert created["available"] is False
    assert created["reason"] == ENTITY_CREATED_TYPE_NA_REASON
    assert "云端" in created["reason"]
    assert result["na"]["EntityCreated"] == ENTITY_CREATED_TYPE_NA_REASON
    # macro 只均分 StateChanged(1.0) 与 RelationAdded(0.8)
    assert result["macro_f1"] == pytest.approx((1.0 + 0.8) / 2)
    assert result["available_kinds"] == ["StateChanged", "RelationAdded"]


def test_all_empty_kinds_are_na_not_fake_perfect() -> None:
    result = evaluate_events([], [], pred_has_types=True, before_relations_available=True)
    assert result["available_kinds"] == []
    assert result["macro_f1"] is None
    for kind, entry in result["per_kind"].items():
        assert entry["available"] is False
        assert entry["reason"]
        assert entry["f1"] is None
    assert "为空" not in result["per_kind"]["EntityCreated"]["reason"]  # 措辞是"无定义"
    assert "P/R/F1 无定义" in result["per_kind"]["EntityCreated"]["reason"]


# ------------------------------------------------- 真链路：manifest + 预测


def test_event_metrics_pipeline_all_correct_is_one() -> None:
    manifest = make_manifest_pairs()
    result = evaluate_event_metrics(
        manifest, preds_all_correct(), split="test"
    )
    assert result["n_pairs"] == 2  # train pair 被 split 过滤掉了
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["available_kinds"] == [
        "EntityCreated",
        "StateChanged",
        "RelationAdded",
    ]
    for kind in result["available_kinds"]:
        entry = result["per_kind"][kind]
        assert entry["f1"] == pytest.approx(1.0)
        assert entry["support"] == entry["predicted"] == entry["matched"]
    # RelationRemoved 恒 N/A
    assert result["per_kind"]["RelationRemoved"]["available"] is False


def test_event_metrics_pipeline_relation_type_mismatch() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    # pair1 唯一关系类型猜错：near -> on
    preds[1].relations = [RelationPrediction(0, 1, "on", "front", False, False)]

    result = evaluate_event_metrics(manifest, preds, split="test")
    rel = result["per_kind"]["RelationAdded"]
    # GT 3 条（pair0 两条 + pair1 一条），预测 3 条，命中 2 条
    assert (rel["support"], rel["predicted"], rel["matched"]) == (3, 3, 2)
    assert rel["precision"] == pytest.approx(2 / 3)
    assert rel["recall"] == pytest.approx(2 / 3)
    assert rel["f1"] == pytest.approx(2 / 3)
    # EntityCreated / StateChanged 全对；macro = (1 + 1 + 2/3) / 3
    assert result["per_kind"]["EntityCreated"]["f1"] == pytest.approx(1.0)
    assert result["per_kind"]["StateChanged"]["f1"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx((1.0 + 1.0 + 2 / 3) / 3)


def test_event_metrics_cloud_predictions_without_types() -> None:
    """云端零样本只有 changes/relations：EntityCreated 标 N/A，macro 不被拉低。"""
    manifest = make_manifest_pairs()
    result = evaluate_event_metrics(
        manifest, preds_all_correct(with_types=False), split="test"
    )
    assert result["per_kind"]["EntityCreated"]["available"] is False
    assert "EntityCreated" in result["na"]
    assert result["available_kinds"] == ["StateChanged", "RelationAdded"]
    assert result["macro_f1"] == pytest.approx(1.0)


def test_event_metrics_missing_prediction_raises() -> None:
    manifest = make_manifest_pairs()
    preds = preds_all_correct()
    del preds[1]
    with pytest.raises(ValueError, match="缺少 pair 预测"):
        evaluate_event_metrics(manifest, preds, split="test")
