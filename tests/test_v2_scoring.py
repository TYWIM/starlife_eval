import json
from pathlib import Path

import pytest

from starlife_eval.v2.prompts import build_prompt, parse_choice, parse_json_answer, parse_short
from starlife_eval.v2.schema import Prediction, Task, load_predictions, load_tasks, write_predictions, write_tasks
from starlife_eval.v2.scoring import bootstrap_ci, is_correct, mcnemar, score_run


def _choice_task() -> Task:
    return Task.from_dict({
        "task_id": "t-1", "dataset": "unit", "capability": "T1", "answer_type": "choice",
        "question": "Where is the mug relative to the monitor?",
        "choices": ["left", "right", "behind"], "answer": 0,
    })


def test_choice_scoring_and_prompt():
    task = _choice_task()
    assert is_correct(task, Prediction(task_id="t-1", choice=0))
    assert not is_correct(task, Prediction(task_id="t-1", choice=1))
    assert not is_correct(task, Prediction(task_id="t-1", error="boom"))
    prompt = build_prompt(task, "direct")
    assert "Where is the mug" in prompt and "A. left" in prompt and "Answer:" in prompt


def test_parse_choice_variants():
    assert parse_choice("Answer: B", 3) == 1
    assert parse_choice("b", 3) == 1
    assert parse_choice("(C)", 3) == 2
    assert parse_choice("nonsense", 3) is None
    assert parse_choice("Answer: Z", 3) is None


def test_short_and_json_scoring():
    short = Task.from_dict({"task_id": "s-1", "dataset": "unit", "capability": "T2",
                            "answer_type": "short", "question": "x", "answers": ["42"],
                            "tolerance": 0.5})
    assert is_correct(short, Prediction(task_id="s-1", text="42.3"))
    assert not is_correct(short, Prediction(task_id="s-1", text="43.0"))
    js = Task.from_dict({"task_id": "j-1", "dataset": "unit", "capability": "T4",
                         "answer_type": "json", "question": "x",
                         "answer": {"n_objects": 3, "change": "appeared"}})
    assert is_correct(js, Prediction(task_id="j-1", value={"n_objects": 3, "change": "appeared"}))
    assert not is_correct(js, Prediction(task_id="j-1", value={"n_objects": 4, "change": "appeared"}))
    assert is_correct(js, Prediction(task_id="j-1", value=parse_json_answer('```json\n{"n_objects":3,"change":"appeared"}\n```')))


def test_bootstrap_ci_bounds():
    flags = [True] * 90 + [False] * 10
    point, lo, hi = bootstrap_ci(flags, draws=200, seed=1)
    assert 0.9 == pytest.approx(point, abs=0.02)
    assert lo <= point <= hi


def test_mcnemar():
    a = [True] * 10 + [False] * 10
    b = [True] * 20
    r = mcnemar(a, b)
    assert r["b"] == 10 and r["c"] == 0
    assert r["p_value"] < 0.01


def test_roundtrip(tmp_path: Path):
    tasks = [_choice_task()]
    tasks_path = tmp_path / "tasks.jsonl"
    assert write_tasks(tasks, tasks_path) == 1
    loaded = load_tasks(tasks_path)
    assert loaded[0].task_id == "t-1" and loaded[0].choices == ("left", "right", "behind")
    pred_path = tmp_path / "preds.jsonl"
    write_predictions([Prediction(task_id="t-1", choice=0)], pred_path)
    assert load_predictions(pred_path)["t-1"].choice == 0


def test_score_run_report():
    tasks = [_choice_task()]
    report = score_run("m", tasks, {"t-1": Prediction(task_id="t-1", choice=0)}).report()
    assert report["accuracy"] == 1.0
    assert report["by_capability"]["T1"]["n"] == 1
    assert "ci95" in report
