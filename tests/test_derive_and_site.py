"""Offline tests for the T4/T6 derived-suite generator and site aggregation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "wsv" / "derive_text_suites.py"
AGG = REPO_ROOT / "scripts" / "wsv" / "build_site_data.py"

#: derived suites are frozen data: pin their sha256 (rerun --check after changing
#: pairs_test.jsonl or the derivation rules; both files then change deliberately)
CHANGE_SHA256 = "7012ac77153c7321209c882b451a8a0f1771240e3979f5c18077a4eecbc3e302"
STATE_REPLAY_SHA256 = "3bd06b52e7c71d0756fedd962c1d895c2516fbf1cae7f29b4e263e4e51962bb2"


def test_derived_suites_are_deterministic():
    """Rerunning the derivation must reproduce the frozen jsonl byte-for-byte."""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    by_file = {row["file"]: row for row in payload if "file" in row}
    assert by_file["change_test.jsonl"]["sha256"] == CHANGE_SHA256
    assert by_file["state_replay_test.jsonl"]["sha256"] == STATE_REPLAY_SHA256
    assert payload[-1] == {"drift": False}


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("build_site_data", AGG)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_report(model_id: str, suite: str, n: int = 4, correct: int = 3) -> dict:
    per_task = [
        {"task_id": f"{suite}-{i}", "covered": i < n, "pred": 0 if i < correct else 1, "gold": 0, "error": None}
        for i in range(n)
    ]
    return {
        "framework": {"version": "openwsv-1"},
        "suite": suite,
        "n": n,
        "model": {"id": model_id, "kind": "vlm", "uses_harness_boxes": False, "adapter": "fake"},
        "metrics": {"accuracy": correct / n, "coverage": 1.0, "ci95": [0.1, 0.9]},
        "per_task": per_task,
    }


def _payload() -> dict:
    boards = [{"id": "overall", "label": "Overall", "n": None, "state": "pending"}]
    for board in ("t4", "t5", "t6", "t7"):
        boards.append({"id": board, "label": board.upper(), "n": 4, "state": "pending"})
    return {
        "benchmark": {"name": "OpenWSV", "short": "WSV", "version": "1.1",
                      "framework": "openwsv-1", "updated": "2026-01-01"},
        "composite": {"label": "Overall", "method": "equal weight"},
        "defaultBoard": "overall",
        "boards": boards,
        "protocol": [],
        "datasets": [],
        "repro": {"command": "x", "repo": "https://github.com/umr-xiaomai/openwsv"},
    }


def test_aggregate_fills_capability_boards_and_overall(tmp_path):
    module = _load_aggregator()
    reports = tmp_path / "reports"
    reports.mkdir()
    suites = {"t4": "change", "t5": "relation_change", "t6": "state_replay", "t7": "cross_frame"}
    for suite in suites.values():
        (reports / f"{suite}__cn_fake.json").write_text(
            json.dumps(_fake_report("cn:fake", suite)), encoding="utf-8"
        )
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")

    data = module.build(reports, payload_path)
    boards = {board["id"]: board for board in data["boards"]}
    for board in ("t4", "t5", "t6", "t7"):
        assert "state" not in boards[board]
        assert boards[board]["rows"][0]["id"] == "fake"
        assert boards[board]["rows"][0]["value"] == 0.75
        assert boards[board]["rows"][0]["ci"] == [0.1, 0.9]
    overall = boards["overall"]
    assert "state" not in overall
    assert overall["rows"][0]["value"] == 0.75
    assert overall["rows"][0]["ci"] is not None
    assert len(overall["rows"][0]["ci"]) == 2


def test_aggregate_leaves_boards_pending_when_partial(tmp_path):
    module = _load_aggregator()
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "change__cn_fake.json").write_text(
        json.dumps(_fake_report("cn:fake", "change")), encoding="utf-8"
    )
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")

    data = module.build(reports, payload_path)
    boards = {board["id"]: board for board in data["boards"]}
    assert boards["t4"]["rows"][0]["id"] == "fake"
    for board in ("t5", "t6", "t7", "overall"):
        assert boards[board]["state"] == "pending"
