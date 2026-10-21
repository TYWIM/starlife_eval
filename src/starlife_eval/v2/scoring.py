"""Scoring and statistics for v2 tasks.

All scores are per-task correctness in [0, 1]; a report aggregates by capability and by dataset,
with 95% bootstrap CIs over tasks and an optional paired McNemar test between two prediction files.
No judge model is involved: correctness is deterministic (choice index, accepted short answers,
numeric tolerance, or field-level JSON match).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import CAPABILITIES, CAPABILITY_NAMES, Prediction, Task


def _norm(text: str) -> str:
    t = str(text).strip().lower()
    t = re.sub(r"[^\w\s\-./%]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _num(text: str) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def is_correct(task: Task, pred: Prediction | None) -> bool:
    if pred is None or pred.error:
        return False
    if task.answer_type == "choice":
        return pred.choice is not None and int(pred.choice) == int(task.answer)
    if task.answer_type == "short":
        if pred.text is None:
            return False
        if task.answers:
            accepted = [str(a) for a in task.answers]
        elif task.answer is not None:
            accepted = [str(task.answer)]
        else:
            accepted = []
        if not accepted:
            return False
        if task.tolerance is not None:
            got = _num(pred.text)
            if got is None:
                return False
            for a in accepted:
                want = _num(a)
                if want is not None and abs(got - want) <= task.tolerance:
                    return True
            return False
        return any(_norm(pred.text) == _norm(a) for a in accepted)
    if task.answer_type == "json":
        if not isinstance(pred.value, Mapping) or not isinstance(task.answer, Mapping):
            return False
        keys = [k for k in task.answer.keys() if not k.startswith("_")]
        if not keys:
            return False
        hits = 0
        for k in keys:
            want = task.answer[k]
            got = pred.value.get(k)
            if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                hits += int(abs(float(got) - float(want)) <= float(task.meta.get("json_tolerance", 1e-6)))
            else:
                hits += int(got == want)
        return hits / len(keys) >= float(task.meta.get("json_pass_ratio", 1.0))
    raise ValueError(f"unknown answer_type {task.answer_type!r}")


def bootstrap_ci(flags: Sequence[bool], draws: int = 1000, seed: int = 20261021, level: float = 0.95) -> tuple[float, float, float]:
    if not flags:
        return 0.0, 0.0, 0.0
    arr = np.asarray(flags, dtype=np.float64)
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    return point, float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def mcnemar(flags_a: Sequence[bool], flags_b: Sequence[bool]) -> dict[str, Any]:
    """Exact-ish McNemar test between two aligned correctness vectors."""
    if len(flags_a) != len(flags_b):
        raise ValueError("mcnemar needs equally long vectors")
    b = sum(1 for a, c in zip(flags_a, flags_b) if (not a) and c)
    c = sum(1 for a, d in zip(flags_a, flags_b) if a and (not d))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "note": "no discordant pairs"}
    # two-sided exact binomial test on discordant pairs
    p_one = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": min(1.0, 2 * p_one)}


@dataclass
class ScoredRun:
    model: str
    tasks: Sequence[Task]
    preds: Mapping[str, Prediction]
    flags: list[bool]

    def by(self, key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[bool]] = {}
        for task, flag in zip(self.tasks, self.flags):
            val = getattr(task, key, None)
            if val is None:
                continue
            groups.setdefault(str(val), []).append(flag)
        out = {}
        for name, flags in sorted(groups.items()):
            point, lo, hi = bootstrap_ci(flags)
            out[name] = {"n": len(flags), "accuracy": round(point, 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "correct": int(sum(flags))}
        return out

    def report(self, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        point, lo, hi = bootstrap_ci(self.flags)
        n = len(self.flags)
        abst = sum(1 for t in self.tasks
                   if (self.preds.get(t.task_id) is None)
                   or (self.preds[t.task_id].choice is None and self.preds[t.task_id].text is None
                       and self.preds[t.task_id].value is None))
        errors = sum(1 for t in self.tasks if (p := self.preds.get(t.task_id)) is not None and p.error)
        by_cap = self.by("capability")
        by_ds = self.by("dataset")
        mean_lat = [p.latency_s for p in self.preds.values() if p.latency_s is not None]
        return {
            "model": self.model,
            "n_tasks": n,
            "accuracy": round(point, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "abstentions": abst,
            "errors": errors,
            "latency_s_mean": round(float(np.mean(mean_lat)), 3) if mean_lat else None,
            "by_capability": by_cap,
            "by_dataset": by_ds,
            "capability_names": CAPABILITY_NAMES,
            "protocol": "v2 task matrix: deterministic scoring, 95% bootstrap CI over tasks (1,000 draws)",
            **dict(meta or {}),
        }


def score_run(model: str, tasks: Sequence[Task], preds: Mapping[str, Prediction],
              meta: Mapping[str, Any] | None = None) -> ScoredRun:
    flags = [is_correct(t, preds.get(t.task_id)) for t in tasks]
    return ScoredRun(model=model, tasks=tasks, preds=preds, flags=flags)


def write_report_md(report: Mapping[str, Any], path: Path) -> Path:
    lines = [f"# v2 benchmark report — {report['model']}", "",
             f"- tasks: {report['n_tasks']}",
             f"- accuracy: **{report['accuracy']:.4f}** (95% CI {report['ci95'][0]:.4f}–{report['ci95'][1]:.4f})",
             f"- abstentions: {report['abstentions']}  |  errors: {report['errors']}"]
    if report.get("latency_s_mean") is not None:
        lines.append(f"- mean latency: {report['latency_s_mean']:.3f}s/task")
    lines += ["", "## By capability", "", "| capability | n | accuracy | 95% CI |", "|---|---|---|---|"]
    for cap in CAPABILITIES:
        if cap in report["by_capability"]:
            r = report["by_capability"][cap]
            lines.append(f"| {cap} {CAPABILITY_NAMES[cap]} | {r['n']} | {r['accuracy']:.4f} | "
                         f"{r['ci95'][0]:.4f}–{r['ci95'][1]:.4f} |")
    lines += ["", "## By dataset", "", "| dataset | n | accuracy | 95% CI |", "|---|---|---|---|"]
    for name, r in report["by_dataset"].items():
        lines.append(f"| {name} | {r['n']} | {r['accuracy']:.4f} | {r['ci95'][0]:.4f}–{r['ci95'][1]:.4f} |")
    lines += ["", "Scores are deterministic; no LLM judge. See docs/BENCHMARK_V2.md for the protocol."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
