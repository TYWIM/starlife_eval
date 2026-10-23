"""OpenWSV 判分/统计单测：严格选项解析、覆盖率口径、bootstrap、McNemar。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# 允许在未 pip install 的源码树里直接跑 `python -m pytest`（CI/沙箱常用）
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from starlife_eval.wsv.scoring import (
    bootstrap_ci,
    mcnemar,
    parse_choice,
    score_choices,
)


# ------------------------------------------------------------------ parse


@pytest.mark.parametrize(
    ("text", "n", "expected"),
    [
        ("A", 4, 0),
        ("b", 4, 1),
        ("(C)", 4, 2),
        ("D.", 4, 3),
        ("Answer: D", 4, 3),
        ("answer is (c)", 4, 2),
        ("Final answer: B", 4, 1),
        ("选 A", 4, 0),
        ("选择：B", 4, 1),
        ("答案：C", 4, 2),
        ("选项 B", 6, 1),
        ("Option D", 6, 3),
        ("**B**", 4, 1),
        ("```\nC\n```", 4, 2),
        (" A \n", 4, 0),
    ],
)
def test_parse_choice_accepts_explicit_forms(text, n, expected):
    assert parse_choice(text, n) == expected


@pytest.mark.parametrize(
    ("text", "n"),
    [
        (None, 4),
        ("", 4),
        ("   ", 4),
        ("I don't know", 4),
        ("The table is on the left", 4),
        ("AB", 4),
        ("Z", 4),  # 越界字母
        ("Answer: Z", 4),
        ("Yes.", 4),
        (42, 4),
    ],
)
def test_parse_choice_rejects_unparsable(text, n):
    assert parse_choice(text, n) is None


def test_parse_choice_never_guesses_from_prose():
    # 自由文本里出现多个独立字母，但没有任何 answer 关键词 -> 解析失败（未覆盖 ≠ 错）
    assert parse_choice("The answer might relate table A to mug B.", 4) is None


# --------------------------------------------------------------- bootstrap


def test_bootstrap_ci_deterministic_and_bounds():
    values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    first = bootstrap_ci(values, n_boot=300, seed=20261022)
    second = bootstrap_ci(values, n_boot=300, seed=20261022)
    assert first == second
    point, lo, hi = first
    assert math.isclose(point, 0.6, abs_tol=1e-12)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([1, 1, 1], n_boot=50, seed=1) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([0, 0], n_boot=50, seed=1) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_seed_changes_draws_but_not_point():
    a = bootstrap_ci([1, 0, 1, 0, 1], n_boot=100, seed=1)
    b = bootstrap_ci([1, 0, 1, 0, 1], n_boot=100, seed=2)
    assert math.isclose(a[0], b[0])
    assert a[1] != b[1] or a[2] != b[2]  # 不同种子给出不同重采样（几乎必然）


# ------------------------------------------------------------------ mcnemar


def test_mcnemar_no_discordant_pairs():
    result = mcnemar([True, False], [True, False])
    assert result == {
        "b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "note": "no discordant pairs"
    }


def test_mcnemar_counts_and_exact_p_value():
    # b=1 (A错B对), c=1 (A对B错) -> n=2, p = 2 * (C(2,0)+C(2,1)) / 4 = 1.0
    result = mcnemar([True, True, False, False], [True, False, False, True])
    assert (result["b"], result["c"], result["n_discordant"]) == (1, 1, 2)
    assert result["p_value"] == 1.0

    # b=4, c=0 -> p = 2 * C(4,0) / 16 = 0.125
    result = mcnemar([False] * 4, [True] * 4)
    assert (result["b"], result["c"]) == (4, 0)
    assert result["p_value"] == pytest.approx(0.125)


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# ----------------------------------------------------------------- scoring


def _tasks():
    return [
        {"task_id": "t1", "choices": ["a", "b", "c", "d"], "answer": 0},
        {"task_id": "t2", "choices": ["a", "b", "c", "d"], "answer": 1},
        {"task_id": "t3", "choices": ["a", "b", "c", "d"], "answer": 2},
        {"task_id": "t4", "choices": ["a", "b", "c", "d"], "answer": 3},
    ]


def test_score_choices_unparsable_is_uncovered_not_wrong():
    preds = {
        "t1": {"text": "A", "error": None},
        "t2": {"text": "B", "error": None},
        "t3": {"text": "I don't know", "error": None},  # 未覆盖
        # t4 缺预测 -> 未覆盖
    }
    result = score_choices(_tasks(), preds, n_boot=100)
    assert result["n"] == 4
    assert result["n_covered"] == 2
    assert result["coverage"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(1.0)  # 只在覆盖题上算
    assert result["n_correct"] == 2
    assert result["n_unparsable"] == 1
    assert result["n_error"] == 1

    rows = {row["task_id"]: row for row in result["per_task"]}
    assert rows["t3"]["covered"] is False
    assert rows["t3"]["pred"] is None
    assert rows["t3"]["error"] == "unparsable choice"
    assert rows["t4"]["covered"] is False
    assert rows["t4"]["error"] == "missing prediction"
    assert rows["t1"]["covered"] is True and rows["t1"]["error"] is None
    assert rows["t1"]["pred"] == 0 and rows["t1"]["gold"] == 0


def test_score_choices_wrong_answers_count_against_accuracy():
    preds = {
        "t1": {"text": "A", "error": None},  # 对
        "t2": {"text": "C", "error": None},  # 错
        "t3": {"text": "Answer: C", "error": None},  # 对
        "t4": {"text": "D", "error": None},  # 对
    }
    result = score_choices(_tasks(), preds, n_boot=100)
    assert result["n_covered"] == 4
    assert result["n_wrong"] == 1
    assert result["accuracy"] == pytest.approx(0.75)


def test_score_choices_adapter_errors_are_uncovered():
    preds = {
        "t1": {"text": "A", "error": None},
        "t2": {"text": None, "error": "HTTP 500: boom"},
        "t3": {"text": None, "error": "timeout"},
        "t4": {"text": None, "error": "empty response (max_tokens/reasoning truncation)"},
    }
    result = score_choices(_tasks(), preds, n_boot=100)
    assert result["n_covered"] == 1
    assert result["n_error"] == 3
    assert result["accuracy"] == pytest.approx(1.0)
    assert {row["error"] for row in result["per_task"] if not row["covered"]} == {
        "HTTP 500: boom", "timeout", "empty response (max_tokens/reasoning truncation)"
    }


def test_score_choices_deterministic_ci():
    preds = {t["task_id"]: {"text": "A", "error": None} for t in _tasks()}
    first = score_choices(_tasks(), preds, n_boot=200, seed=20261022)
    second = score_choices(_tasks(), preds, n_boot=200, seed=20261022)
    assert first["ci95"] == second["ci95"]


def test_score_choices_empty_suite():
    result = score_choices([], {}, n_boot=10)
    assert result["n"] == 0 and result["n_covered"] == 0
    assert result["accuracy"] == 0.0 and result["coverage"] == 0.0
    assert result["ci95"] == [0.0, 0.0]