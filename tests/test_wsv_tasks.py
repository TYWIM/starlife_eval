"""OpenWSV 冻结 suite 加载测试（不联网：只读包内 suites/*.jsonl）。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# 允许在未 pip install 的源码树里直接跑 `python -m pytest`（CI/沙箱常用）
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from starlife_eval.wsv.tasks import (
    RELATION_PATTERN,
    SUITE_FILES,
    SUITE_SIZES,
    SUITES,
    dataset_sha256,
    describe_suites,
    load_suite,
    suite_path,
)

TASK_FIELDS = {"task_id", "suite", "question", "images", "choices", "answer", "answers", "meta"}


def test_suites_tuple_and_sizes():
    assert SUITES == ("spatialmqa", "spatialmqa_relation", "pairs")
    assert SUITE_SIZES == {"spatialmqa": 1612, "spatialmqa_relation": 305, "pairs": 1000}


@pytest.mark.parametrize("suite", SUITES)
def test_load_suite_shape(suite):
    tasks = load_suite(suite)
    assert len(tasks) == SUITE_SIZES[suite]
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    for task in tasks[:5]:
        assert set(task.keys()) == TASK_FIELDS
        assert task["suite"] == suite
        assert isinstance(task["question"], str) and task["question"]
        assert isinstance(task["images"], list) and task["images"]
        assert isinstance(task["meta"], dict)


def test_spatialmqa_choices_and_answers():
    tasks = load_suite("spatialmqa")
    answer_counts = {2: 0, 4: 0, 6: 0}
    for task in tasks:
        assert task["choices"], task["task_id"]
        assert 0 <= task["answer"] < len(task["choices"])
        answer_counts[len(task["choices"])] = answer_counts.get(len(task["choices"]), 0) + 1
    # 冻结数据只有 2/4/6 选项（SpatialMQA 口径）
    assert answer_counts[2] + answer_counts[4] + answer_counts[6] == len(tasks)


def test_relation_subset_is_exactly_the_regex_matches():
    full = load_suite("spatialmqa")
    subset = load_suite("spatialmqa_relation")
    expected_ids = [
        t["task_id"] for t in full if RELATION_PATTERN.match(t["question"]) is not None
    ]
    assert len(subset) == 305
    assert [t["task_id"] for t in subset] == expected_ids
    for task in subset:
        match = RELATION_PATTERN.match(task["question"])
        assert match is not None
        assert task["meta"]["relation_x"] == match.group("x")
        assert task["meta"]["relation_y"] == match.group("y")


def test_pairs_suite_ground_truth_structure():
    tasks = load_suite("pairs")
    assert len(tasks) == 1000
    for task in tasks:
        meta = task["meta"]
        assert meta["split"] == "test"
        assert meta["before"].endswith(".jpg") and meta["after"].endswith(".jpg")
        assert task["answer"] is None and task["choices"] == []
        assert task["images"] == [meta["before"], meta["after"]]
        slots = [e["slot"] for e in meta["entities"]]
        assert slots == list(range(len(slots)))
        for change_slot in meta["changes"]:
            assert 0 <= int(change_slot) < len(slots)
        for rel in meta["relations"]:
            assert 0 <= rel["from_slot"] < len(slots)
            assert 0 <= rel["to_slot"] < len(slots)
            assert isinstance(rel["rel_pos_present"], bool)


def test_pairs_relation_directions_cover_both_orientations():
    """方向匹配协议依赖 GT 里同时存在正序/逆序关系，确认测试数据两向都有。"""
    tasks = load_suite("pairs")
    forward = backtrack = 0
    for task in tasks:
        for rel in task["meta"]["relations"]:
            if rel["from_slot"] < rel["to_slot"]:
                forward += 1
            else:
                backtrack += 1
    assert forward > 0 and backtrack > 0


def test_load_suite_is_deterministic():
    a = load_suite("spatialmqa_relation")
    b = load_suite("spatialmqa_relation")
    assert a == b


def test_unknown_suite_raises():
    with pytest.raises(ValueError, match="未知 suite"):
        load_suite("nope")


def test_suite_files_frozen_sha256():
    for suite in SUITES:
        digest = dataset_sha256(suite)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest == dataset_sha256(suite)
        assert suite_path(suite).exists()


def test_describe_suites():
    rows = describe_suites()
    assert [row["suite"] for row in rows] == list(SUITES)
    for row in rows:
        assert row["n"] == SUITE_SIZES[row["suite"]]
        assert len(row["sha256"]) == 64


def test_config_frozen_constants_align_with_package():
    """config.py 是冻结常量的唯一真源；包内实现必须与之一致。"""
    from starlife_eval.wsv import config, runner
    from starlife_eval.wsv.prompts import PROMPT_VERSION

    assert config.FRAMEWORK_VERSION == runner.FRAMEWORK_VERSION
    assert config.PROMPT_VERSION == PROMPT_VERSION
    assert config.SEEDS["sample"] == runner.SAMPLE_SEED
    assert config.SEEDS["bootstrap"] == runner.BOOTSTRAP_SEED
    assert set(config.SUITES) == set(SUITES)
    for name in SUITES:
        assert config.SUITES[name]["file"] == f"suites/{SUITE_FILES[name]}"


def test_framework_block_contract():
    from starlife_eval.wsv.runner import framework_block

    block = framework_block("spatialmqa")
    assert set(block) == {
        "version", "dataset_sha256", "detector", "prompt_version",
        "scorer", "seed", "bootstrap_seed",
    }
    assert block["version"] == "openwsv-1"
    assert block["prompt_version"] == "openwsv-1"
    assert block["dataset_sha256"] == dataset_sha256("spatialmqa")
    assert block["seed"] == 20261020
    assert block["bootstrap_seed"] == 20261022
    assert block["scorer"].startswith("starlife_eval@")
    assert set(block["detector"]) == {
        "model", "weights_sha256", "box_threshold", "text_threshold",
    }
    assert block["detector"]["model"] == "IDEA-Research/grounding-dino-tiny"
    assert block["detector"]["box_threshold"] == 0.2
    assert block["detector"]["text_threshold"] == 0.2