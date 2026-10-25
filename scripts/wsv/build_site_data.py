"""Aggregate OpenWSV result JSONs into the frontend data payload.

Reads result JSON files produced by ``python -m starlife_eval.wsv run``
(``reports/openwsv-v2/{suite}__{model}.json``), fills the T4-T7 boards and the
``overall`` composite in ``site/src/data/openwsv.generated.json``, and leaves
boards without any report untouched (so a partial run still produces a valid site).

Board definitions (labels, notes, n) live in the site JSON and are preserved;
only ``rows``/``state`` are rewritten.  ``overall`` is the equal-weight mean of
T4-T7 accuracies for models that have all four suites; its 95% CI is a
capability-stratified bootstrap (resample covered tasks inside each suite, then
average the four suite accuracies), seed 20261022.

Usage::

    python scripts/wsv/build_site_data.py
    python scripts/wsv/build_site_data.py --reports reports/openwsv-v2 \\
        --payload site/src/data/openwsv.generated.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = REPO_ROOT / "reports" / "openwsv-v2"
DEFAULT_PAYLOAD = REPO_ROOT / "site" / "src" / "data" / "openwsv.generated.json"

CAPABILITY_BOARDS: tuple[str, ...] = ("t4", "t5", "t6", "t7")
SUITE_FOR_BOARD = {
    "t4": "change",
    "t5": "relation_change",
    "t6": "state_replay",
    "t7": "cross_frame",
}
SUITE_ORDER: tuple[str, ...] = tuple(SUITE_FOR_BOARD[board] for board in CAPABILITY_BOARDS)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261022


def _display_id(model_id: str) -> str:
    """Gold model rows show the vendor-neutral model name (no gateway prefix)."""
    value = str(model_id)
    return value.split(":", 1)[1] if value.startswith("cn:") and value.count(":") >= 1 else value


def _suite_shas() -> dict[str, str]:
    """Current frozen sha256 per suite (stale reports from older data versions are skipped)."""
    import sys

    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from starlife_eval.wsv.tasks import dataset_sha256

    return {suite: dataset_sha256(suite) for suite in SUITE_FOR_BOARD.values()}


def _load_report(
    reports: Path,
    suite: str,
    model_id: str,
    expected_sha: str | None = None,
) -> dict[str, Any] | None:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_id)
    path = reports / f"{suite}__{safe}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("per_task"), list):
        return None
    report_sha = str((data.get("framework") or {}).get("dataset_sha256") or "")
    if expected_sha is not None and report_sha != expected_sha:
        return None  # stale data version: never aggregate it into the leaderboard
    data["__path"] = str(path)
    return data


def _row_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    meta = report.get("model") or {}
    metrics = report.get("metrics") or {}
    return {
        "id": _display_id(str(meta.get("id", ""))),
        "label": _display_id(str(meta.get("id", ""))),
        "value": round(float(metrics.get("accuracy") or 0.0), 4),
        "ci": [round(float(v), 4) for v in (metrics.get("ci95") or [0.0, 0.0])],
        "coverage": round(float(metrics.get("coverage") or 0.0), 4),
        "usesHarnessBoxes": bool(
            meta.get("uses_harness_boxes", str(meta.get("kind", "vlm")) == "structured")
        ),
        "n": int(report.get("n") or len(report.get("per_task") or ())),
    }


def _covered_flags(report: Mapping[str, Any]) -> list[bool]:
    flags: list[bool] = []
    for row in report.get("per_task") or ():
        if row.get("covered"):
            flags.append(bool(int(row.get("pred")) == int(row.get("gold"))))
    return flags


def _composite_rows(
    reports: Path,
    models: Sequence[str],
    suite_shas: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    available: dict[str, dict[str, Mapping[str, Any]]] = {}
    for suite in SUITE_ORDER:
        for model in models:
            report = _load_report(
                reports, suite, model, (suite_shas or {}).get(suite)
            )
            if report is not None:
                available.setdefault(model, {})[suite] = report

    rows: list[dict[str, Any]] = []
    flags_by_model: dict[str, dict[str, list[bool]]] = {}
    for model in models:
        per_suite = available.get(model) or {}
        if set(per_suite) != set(SUITE_ORDER):
            continue
        accuracies = [float((per_suite[s].get("metrics") or {}).get("accuracy") or 0.0) for s in SUITE_ORDER]
        coverages = [float((per_suite[s].get("metrics") or {}).get("coverage") or 0.0) for s in SUITE_ORDER]
        n_total = sum(int(per_suite[s].get("n") or 0) for s in SUITE_ORDER)
        n_covered = sum(
            sum(1 for t in per_suite[s].get("per_task") or () if t.get("covered"))
            for s in SUITE_ORDER
        )
        rows.append(
            {
                "id": _display_id(model),
                "label": _display_id(model),
                "value": round(float(np.mean(accuracies)), 4),
                "ci": None,  # filled by the stratified bootstrap below
                "coverage": round(n_covered / n_total, 4) if n_total else 0.0,
                "usesHarnessBoxes": False,
                "n": n_total,
            }
        )
        flags_by_model[model] = {s: _covered_flags(per_suite[s]) for s in SUITE_ORDER}

    # capability-stratified bootstrap: average of the four suite accuracies
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for row in rows:
        model = next(m for m in flags_by_model if _display_id(m) == row["id"])
        per_suite = flags_by_model[model]
        draws: list[float] = []
        for _ in range(BOOTSTRAP_DRAWS):
            means = []
            for suite in SUITE_ORDER:
                flags = per_suite[suite]
                if not flags:
                    continue
                idx = rng.integers(0, len(flags), size=len(flags))
                means.append(float(np.asarray(flags, dtype=np.float64)[idx].mean()))
            draws.append(float(np.mean(means)))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        row["ci"] = [round(float(lo), 4), round(float(hi), 4)]

    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows


def _model_ids(reports: Path) -> list[str]:
    ids: set[str] = set()
    for path in reports.glob("*__*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = str((data.get("model") or {}).get("id") or "")
        if model:
            ids.add(model)
    return sorted(ids)


def build(reports: Path, payload_path: Path) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    models = _model_ids(reports)
    suite_shas = _suite_shas()
    boards = {board["id"]: board for board in payload.get("boards") or ()}
    filled = 0

    for board_id in CAPABILITY_BOARDS:
        board = boards.get(board_id)
        if board is None:
            continue
        suite = SUITE_FOR_BOARD[board_id]
        rows = []
        for model in models:
            report = _load_report(reports, suite, model, suite_shas.get(suite))
            if report is not None:
                rows.append(_row_from_report(report))
        if not rows:
            continue
        board.pop("state", None)
        board["rows"] = sorted(rows, key=lambda r: r["value"], reverse=True)
        board["n"] = int(max(r.get("n") or 0 for r in rows))
        filled += 1

    overall = boards.get("overall")
    if overall is not None:
        composite = _composite_rows(reports, models, suite_shas)
        if composite:
            overall.pop("state", None)
            overall["rows"] = composite
            overall["n"] = int(composite[0].get("n") or 0) or None

    payload.setdefault("benchmark", {})["updated"] = datetime.now(timezone.utc).astimezone().date().isoformat()
    payload["_generated"] = {
        "reports": str(reports),
        "models": models,
        "boards_filled": filled,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate OpenWSV reports into the site payload")
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS))
    parser.add_argument("--payload", default=str(DEFAULT_PAYLOAD))
    args = parser.parse_args(list(argv) if argv is not None else None)

    reports = Path(args.reports)
    if not reports.exists():
        print(f"[build_site_data] no reports directory: {reports}")
        return 0
    payload_path = Path(args.payload)
    data = build(reports, payload_path)
    payload_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    boards = {board["id"]: len(board.get("rows") or ()) for board in data["boards"]}
    print(json.dumps({"payload": str(payload_path), "boards": boards,
                      "models": data.get("_generated", {}).get("models")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
