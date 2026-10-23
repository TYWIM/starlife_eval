"""OpenWSV 结构化协议单测（fixture 假响应，不联网、不依赖 dataset 帧文件）。

覆盖：
* WSV ``/predict`` 响应 -> ``PairPrediction`` 的解析；
* 方向匹配：GT 正序用 forward 响应，逆序用 reversed 响应；
* 复用 60 对协议指标（change_top1 / macro_f1 / relation / rel_pos / attrs）；
* bootstrap CI 的确定性、覆盖率口径（缺一次响应 = 未覆盖，不按错题计）。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

# 允许在未 pip install 的源码树里直接跑 `python -m pytest`（CI/沙箱常用）
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from starlife_eval.wsv.state import (
    build_manifest,
    evaluate_pairs,
    pair_prediction,
    slot_id,
)


# ------------------------------------------------------------------ fixtures


def _task_a():
    return {
        "task_id": "pairs-A",
        "suite": "pairs",
        "question": "q",
        "images": ["frames/A_before.jpg", "frames/A_after.jpg"],
        "choices": [],
        "answer": None,
        "answers": [],
        "meta": {
            "pair_id": "A",
            "dataset": "dataset_spatial_v3",
            "split": "test",
            "before": "frames/A_before.jpg",
            "after": "frames/A_after.jpg",
            "entities": [
                {"slot": 0, "type_key": "table", "bbox": [0, 0, 100, 100], "present_after": True},
                {"slot": 1, "type_key": "mug", "bbox": [10, 10, 20, 20], "present_after": True},
                {"slot": 2, "type_key": "lamp", "bbox": [30, 30, 40, 40], "present_after": False},
            ],
            "changes": {2: "disappeared"},
            "relations": [
                {
                    "from_slot": 0, "to_slot": 1, "type": "above", "rel_pos": "top",
                    "rel_pos_present": True, "contact": False, "support": False, "dist_m": None,
                },
                {
                    "from_slot": 2, "to_slot": 0, "type": "part_of", "rel_pos": "front",
                    "rel_pos_present": True, "contact": True, "support": False, "dist_m": None,
                },
            ],
            "negative": False,
            "raw_meta": {},
        },
    }


def _task_b():
    return {
        "task_id": "pairs-B",
        "suite": "pairs",
        "question": "q",
        "images": ["frames/B_before.jpg", "frames/B_after.jpg"],
        "choices": [],
        "answer": None,
        "answers": [],
        "meta": {
            "pair_id": "B",
            "dataset": "dataset_spatial_v3",
            "split": "test",
            "before": "frames/B_before.jpg",
            "after": "frames/B_after.jpg",
            "entities": [
                {"slot": 0, "type_key": "desk", "bbox": [0, 0, 200, 200], "present_after": True},
                {"slot": 1, "type_key": "cup", "bbox": [10, 10, 30, 30], "present_after": True},
                {"slot": 2, "type_key": "book", "bbox": [40, 40, 60, 60], "present_after": True},
                {"slot": 3, "type_key": "pen", "bbox": [70, 70, 90, 90], "present_after": True},
            ],
            "changes": {1: "displaced"},
            "relations": [
                {
                    "from_slot": 3, "to_slot": 1, "type": "next_to", "rel_pos": "right",
                    "rel_pos_present": True, "contact": False, "support": True, "dist_m": None,
                },
            ],
            "negative": False,
            "raw_meta": {},
        },
    }


def _forward_a():
    return {
        "slots": [
            {"id": "slot0", "change": "none"},
            {"id": "slot1", "change": "none"},
            {"id": "slot2", "change": "appeared"},
        ],
        "pairs": [
            {"from": "slot0", "to": "slot1", "relation": "above", "rel_pos": "top",
             "contact": 0, "support": 0},
            {"from": "slot0", "to": "slot2", "relation": "below", "rel_pos": "top",
             "contact": 0, "support": 0},
            {"from": "slot1", "to": "slot2", "relation": "none", "rel_pos": "top",
             "contact": 0, "support": 0},
        ],
    }


def _reversed_a():
    return {
        "slots": [
            {"id": "slot0", "change": "none"},
            {"id": "slot1", "change": "none"},
            {"id": "slot2", "change": "disappeared"},
        ],
        "pairs": [
            {"from": "slot2", "to": "slot1", "relation": "next_to", "rel_pos": "front",
             "contact": 0, "support": 0},
            {"from": "slot2", "to": "slot0", "relation": "part_of", "rel_pos": "front",
             "contact": 1, "support": 0},
            {"from": "slot1", "to": "slot0", "relation": "below", "rel_pos": "top",
             "contact": 0, "support": 0},
        ],
    }


def _forward_b():
    return {
        "slots": [
            {"id": "slot0", "change": "none"},
            {"id": "slot1", "change": "displaced"},
            {"id": "slot2", "change": "none"},
            {"id": "slot3", "change": "none"},
        ],
        "pairs": [
            # 正序枚举里 (3,1) 不存在，只有 (1,3) 且关系错误
            {"from": "slot1", "to": "slot3", "relation": "below", "rel_pos": "left",
             "contact": 0, "support": 0},
        ],
    }


def _reversed_b():
    return {
        "slots": [
            {"id": "slot0", "change": "none"},
            {"id": "slot1", "change": "displaced"},
            {"id": "slot2", "change": "none"},
            {"id": "slot3", "change": "none"},
        ],
        "pairs": [
            {"from": "slot3", "to": "slot1", "relation": "next_to", "rel_pos": "right",
             "contact": 0, "support": 1},
        ],
    }


def _responses():
    return {
        "pairs-A": {"forward": _forward_a(), "reversed": _reversed_a(), "error": None},
        "pairs-B": {"forward": _forward_b(), "reversed": _reversed_b(), "error": None},
    }


# --------------------------------------------------------------- slot parsing


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), ("0", 0), ("slot7", 7), ("slot:slot7", 7), ("slot_7", 7)],
)
def test_slot_id(value, expected):
    assert slot_id(value) == expected


# ------------------------------------------------------------ response -> pred


def test_pair_prediction_direction_matching():
    pred = pair_prediction(_task_a(), _forward_a(), _reversed_a())
    assert pred.changes == {0: "none", 1: "none", 2: "appeared"}
    by_key = {(r.from_slot, r.to_slot): r for r in pred.relations}
    # 正序 GT (0,1) 用 forward 响应
    assert by_key[(0, 1)].type == "above"
    assert by_key[(0, 1)].rel_pos == "top"
    # 逆序 GT (2,0) 用 reversed 响应（forward 里没有该方向）
    assert by_key[(2, 0)].type == "part_of"
    assert by_key[(2, 0)].rel_pos == "front"
    assert by_key[(2, 0)].contact is True and by_key[(2, 0)].support is False


def test_pair_prediction_missing_relation_is_skipped():
    task = _task_a()
    # 逆序 GT 关系 (2,0) 在两个响应里都不存在 -> 该关系没有预测（记为不命中）
    reversed_response = copy.deepcopy(_reversed_a())
    reversed_response["pairs"] = [p for p in reversed_response["pairs"] if p["from"] != "slot2"]
    pred = pair_prediction(task, _forward_a(), reversed_response)
    assert [(r.from_slot, r.to_slot) for r in pred.relations] == [(0, 1)]


def test_adapter_wrapped_response_is_accepted():
    wrapped = {"response": _forward_a(), "latency_s": 1.0, "error": None}
    pred = pair_prediction(_task_a(), wrapped, {"response": _reversed_a(), "error": None})
    assert pred.changes[2] == "appeared"


# ------------------------------------------------------------------- manifest


def test_build_manifest_reconstructs_ground_truth():
    manifest = build_manifest([_task_a(), _task_b()])
    assert len(manifest.pairs) == 2
    assert manifest.slot_offsets == (0, 3, 7)
    pair = manifest.pairs[0]
    assert pair.pair_id == "A"
    assert pair.changes == {2: "disappeared"}
    assert [e.slot for e in pair.entities] == [0, 1, 2]
    assert [r.type for r in pair.relations] == ["above", "part_of"]
    assert "rel_pos" in pair.raw["relations"][0]


# ----------------------------------------------------------------- metrics


def test_evaluate_pairs_exact_metrics():
    tasks = [_task_a(), _task_b()]
    result = evaluate_pairs(tasks, _responses(), n_boot=200, seed=20261022)

    assert result["n_pairs"] == 2
    assert result["n_covered"] == 2
    assert result["coverage"] == pytest.approx(1.0)
    metrics = result["metrics"]
    assert metrics["change_slots"] == 7
    assert metrics["change_top1"] == pytest.approx(6 / 7, abs=1e-4)
    assert metrics["change_macro_f1"] == pytest.approx(1.0)
    assert metrics["relation_pairs"] == 3
    assert metrics["relation_type_acc"] == pytest.approx(1.0)
    assert metrics["rel_pos_acc"] == pytest.approx(1.0)
    assert metrics["attrs_acc"] == pytest.approx(1.0)

    ci = result["ci95"]
    assert ci["change_top1"] is not None
    assert ci["change_top1"][0] <= metrics["change_top1"] <= ci["change_top1"][1]
    assert ci["relation_type_acc"] == [1.0, 1.0]
    assert ci["attrs_acc"] == [1.0, 1.0]

    per_pair = {row["task_id"]: row for row in result["per_pair"]}
    assert per_pair["pairs-A"]["change_correct"] == 2
    assert per_pair["pairs-B"]["change_correct"] == 4
    assert per_pair["pairs-B"]["relation_type_correct"] == 1


def test_evaluate_pairs_is_deterministic():
    tasks = [_task_a(), _task_b()]
    first = evaluate_pairs(tasks, _responses(), n_boot=100, seed=20261022)
    second = evaluate_pairs(tasks, _responses(), n_boot=100, seed=20261022)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_missing_reversed_response_is_uncovered_not_wrong():
    responses = _responses()
    responses["pairs-B"] = {
        "forward": _forward_b(), "reversed": None, "error": None,
    }
    result = evaluate_pairs([_task_a(), _task_b()], responses, n_boot=100, seed=20261022)
    assert result["n_covered"] == 1
    assert result["coverage"] == pytest.approx(0.5)
    # 只在覆盖的 pair A 上算指标
    assert result["metrics"]["change_top1"] == pytest.approx(2 / 3, abs=1e-4)
    per_pair = {row["task_id"]: row for row in result["per_pair"]}
    assert per_pair["pairs-B"]["covered"] is False
    assert per_pair["pairs-B"]["error"] == "incomplete forward/reversed response"


def test_adapter_error_is_uncovered():
    responses = _responses()
    responses["pairs-A"] = {"forward": None, "reversed": None, "error": "HTTP 500: boom"}
    result = evaluate_pairs([_task_a(), _task_b()], responses, n_boot=100, seed=20261022)
    assert result["n_covered"] == 1
    per_pair = {row["task_id"]: row for row in result["per_pair"]}
    assert per_pair["pairs-A"]["error"] == "HTTP 500: boom"
    assert per_pair["pairs-A"]["covered"] is False


def test_evaluate_pairs_empty():
    result = evaluate_pairs([], {}, n_boot=10)
    assert result["n_pairs"] == 0 and result["n_covered"] == 0
    assert result["coverage"] == 0.0
    assert result["metrics"]["change_top1"] == 0.0
    assert all(value is None for value in result["ci95"].values())