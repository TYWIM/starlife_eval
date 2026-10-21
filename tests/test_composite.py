"""综合能力分：口径、缺值剔除、等权与排序。

这些断言是手算的已知数值，用来钉住"综合分 = 7 维等权平均"这个契约，
防止以后有人偷偷改权重或把 N/A 当 0 分。
"""

from __future__ import annotations

import pytest

from starlife_eval.report.composite import (
    COMPOSITE_DIMS,
    add_composite_score,
    composite_score,
    state_acc_common,
)


def _model(*, pair: dict | None = None, **over) -> dict:
    base_pair = {
        "change_top1": 0.90,
        "change_macro_f1": 0.80,
        "relation_type_acc": 0.70,
        "rel_pos_acc": 0.60,
        "attrs_acc": 0.50,
    }
    base_pair.update(pair or {})
    base = {
        "pair_protocol": base_pair,
        "event_metrics": {"macro_f1_common": 0.40},
        "state_replay": {"objects": 1.00, "changes": 0.90, "relations": 0.80, "attrs": 0.70,
                         "types": 0.10, "state_acc": 0.70},
    }
    base.update(over)
    return base


def test_composite_is_equal_weight_mean_of_seven_dims():
    out = composite_score(_model())
    # (90 + 80 + 70 + 60 + 50 + 40 + 85) / 7 = 475 / 7 = 67.857 -> 67.9
    assert out["score"] == pytest.approx(67.9, abs=1e-9)
    assert out["available"] == 7
    assert out["total"] == 7
    assert out["missing"] == []
    assert len(out["dims"]) == 7
    assert sum(out["weights"]) == pytest.approx(1.0, abs=1e-4)


def test_types_excluded_from_state_consistency():
    # 4 维均值 (1.00+0.90+0.80+0.70)/4 = 0.85；types=0.10 不得拉低
    assert state_acc_common(_model()["state_replay"]) == pytest.approx(0.85)
    out = composite_score(_model())
    dims = {d["key"]: d for d in out["dims"]}
    assert dims["state_consistency"]["value"] == pytest.approx(85.0)
    assert out["state_dims_used"] == ["objects", "changes", "relations", "attrs"]


def test_missing_dims_are_dropped_not_zeroed():
    m = _model(pair={"rel_pos_acc": None}, state_replay=None)
    out = composite_score(m)
    # 剩下 90,80,70,50,40 -> 330/5 = 66.0
    assert out["score"] == pytest.approx(66.0)
    assert out["available"] == 5
    assert set(out["missing"]) == {"rel_pos", "state_consistency"}
    dims = {d["key"]: d for d in out["dims"]}
    assert dims["rel_pos"]["available"] is False and dims["rel_pos"]["value"] is None


def test_all_missing_gives_none_not_zero():
    out = composite_score({})
    assert out["score"] is None
    assert out["available"] == 0
    assert len(out["missing"]) == len(COMPOSITE_DIMS)


def test_add_composite_score_is_idempotent_and_sets_meta():
    report = {"models": {"a": _model(), "b": _model()}, "meta": {}}
    add_composite_score(report)
    first = {k: dict(v["composite"]) for k, v in report["models"].items()}
    add_composite_score(report)  # 幂等
    assert {k: dict(v["composite"]) for k, v in report["models"].items()} == first
    assert report["meta"]["composite"]["method"] == "equal-weight"
    assert report["meta"]["composite"]["n_dims"] == 7
    assert report["models"]["a"]["composite_score"] == pytest.approx(67.9)


def test_leaderboard_ranks_by_composite_not_legacy_score():
    from starlife_eval.report.report_json import build_leaderboard_rows

    good_legacy = _model(event_metrics={"macro_f1_common": 0.95}, state_replay=None,
                         pair_protocol={"change_top1": 0.10, "change_macro_f1": 0.10,
                                        "relation_type_acc": 0.10, "rel_pos_acc": 0.10,
                                        "attrs_acc": 0.10})
    balanced = _model()
    report = {"models": {"legacy": good_legacy, "balanced": balanced}}
    add_composite_score(report)
    rows = build_leaderboard_rows(report)
    assert [r["id"] for r in rows] == ["balanced", "legacy"]
    assert rows[0]["isBest"] is True and rows[1]["isBest"] is False
    # 旧 score 仍然保留用于对照
    assert rows[0]["score"] == pytest.approx(40.0)
