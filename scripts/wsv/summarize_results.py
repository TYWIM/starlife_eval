"""Summarize OpenWSV result JSONs into markdown tables + pairwise McNemar tests.

Reads ``reports/openwsv-v2/{suite}__{model}.json`` and writes
``reports/openwsv-v2/TABLES.md`` plus ``summary.json``.  Scores are exactly the
per-task records produced by the frozen runner (accuracy over covered tasks,
coverage, 95% task-level bootstrap CI); pairwise comparisons use the exact
McNemar test from :mod:`starlife_eval.wsv.scoring` aligned on task_id.

Usage::

    python scripts/wsv/summarize_results.py
    python scripts/wsv/summarize_results.py --reports reports/openwsv-v2 --out reports/openwsv-v2/TABLES.md
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from starlife_eval.wsv.scoring import mcnemar  # noqa: E402

DEFAULT_REPORTS = REPO_ROOT / "reports" / "openwsv-v2"
SUITE_ORDER = ("change", "relation_change", "state_replay", "cross_frame")
SUITE_TITLES = {
    "change": "T4 Spatial change",
    "relation_change": "T5 Relation change",
    "state_replay": "T6 State replay",
    "cross_frame": "T7 Cross-frame disjoint",
}


def _display(model_id: str) -> str:
    return model_id.split(":", 1)[1] if model_id.startswith("cn:") else model_id


def _suite_shas() -> dict[str, str]:
    from starlife_eval.wsv.tasks import dataset_sha256

    return {suite: dataset_sha256(suite) for suite in SUITE_ORDER}


def _load(reports: Path) -> dict[str, dict[str, dict[str, Any]]]:
    shas = _suite_shas()
    by_suite: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(reports.glob("*__*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        suite = str(data.get("suite") or "")
        model = str((data.get("model") or {}).get("id") or "")
        if suite not in SUITE_ORDER or not model or not isinstance(data.get("per_task"), list):
            continue
        report_sha = str((data.get("framework") or {}).get("dataset_sha256") or "")
        if report_sha != shas.get(suite):
            continue  # stale data version: do not report it as a current score
        covered = sum(1 for t in data.get("per_task") or () if t.get("covered"))
        if covered == 0:
            continue  # no valid reading: never list a 0-coverage run as a score
        tasks = data.get("per_task") or ()
        transport = sum(1 for t in tasks if str(t.get("error") or "").startswith("HTTP "))
        if transport > max(5, 0.05 * len(tasks)):
            continue  # provider/transport failure dominated the run: not a model reading
        # prefer the largest run per (suite, model): full run over smoke dirs
        by_suite.setdefault(suite, {})[model] = data
    return by_suite


def _flags(report: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in report.get("per_task") or ():
        if row.get("covered"):
            out[str(row.get("task_id"))] = bool(int(row.get("pred")) == int(row.get("gold")))
    return out


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _fmt_ci(ci: Sequence[float] | None) -> str:
    if not ci or len(ci) != 2:
        return "—"
    return f"[{100.0 * float(ci[0]):.2f}, {100.0 * float(ci[1]):.2f}]"


def build_tables(by_suite: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    lines: list[str] = ["# OpenWSV T4-T7 results", ""]
    lines.append("Accuracy is computed over covered tasks only; coverage is reported separately.")
    lines.append("95% CI = task-level bootstrap (1,000 draws, seed 20261022).")
    lines.append("")
    for suite in SUITE_ORDER:
        models = by_suite.get(suite) or {}
        if not models:
            continue
        lines.append(f"## {SUITE_TITLES[suite]} (`{suite}`)")
        lines.append("")
        lines.append("| model | n | accuracy | coverage | 95% CI |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        ordered = sorted(
            models.items(),
            key=lambda kv: float((kv[1].get("metrics") or {}).get("accuracy") or 0.0),
            reverse=True,
        )
        for model, report in ordered:
            metrics = report.get("metrics") or {}
            lines.append(
                f"| {_display(model)} | {report.get('n', 0)} | "
                f"{_fmt_pct(metrics.get('accuracy'))} | {_fmt_pct(metrics.get('coverage'))} | "
                f"{_fmt_ci(metrics.get('ci95'))} |"
            )
        lines.append("")

        # pairwise exact McNemar on tasks covered by both models
        lines.append("<details><summary>Pairwise exact McNemar (discordant tasks)</summary>")
        lines.append("")
        lines.append("| A | B | A-only correct | B-only correct | discordant | p |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for a, b in combinations(ordered, 2):
            fa, fb = _flags(a[1]), _flags(b[1])
            common = sorted(set(fa) & set(fb))
            result = mcnemar([fa[t] for t in common], [fb[t] for t in common])
            lines.append(
                f"| {_display(a[0])} | {_display(b[0])} | {result['b']} | {result['c']} | "
                f"{result['n_discordant']} | {result['p_value']:.4g} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_summary(by_suite: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suite, models in by_suite.items():
        out[suite] = {
            _display(model): {
                "n": report.get("n"),
                "accuracy": (report.get("metrics") or {}).get("accuracy"),
                "coverage": (report.get("metrics") or {}).get("coverage"),
                "ci95": (report.get("metrics") or {}).get("ci95"),
                "dataset_sha256": (report.get("framework") or {}).get("dataset_sha256"),
            }
            for model, report in sorted(models.items())
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize OpenWSV result JSONs")
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS))
    parser.add_argument("--out", default=None, help="markdown output (default TABLES.md)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    reports = Path(args.reports)
    by_suite = _load(reports)
    if not by_suite:
        print(f"[summarize] no result JSONs under {reports}")
        return 0
    out = Path(args.out) if args.out else reports / "TABLES.md"
    out.write_text(build_tables(by_suite), encoding="utf-8")
    (reports / "summary.json").write_text(
        json.dumps(build_summary(by_suite), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(out), "suites": {k: len(v) for k, v in by_suite.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
