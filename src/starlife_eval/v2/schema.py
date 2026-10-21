"""Task/prediction schema for the v2 (task-matrix) benchmark.

The v2 protocol is model-agnostic: every participant answers the same JSONL tasks and returns
predictions in the same JSONL schema, regardless of whether the model is an API VLM, an open-weights
VLM, or a specialist/world-model adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

CAPABILITIES = ("T1", "T2", "T3", "T4", "T5", "T6")
CAPABILITY_NAMES = {
    "T1": "spatial relations (multi-hop)",
    "T2": "6D location / orientation",
    "T3": "counterfactual object reasoning",
    "T4": "state change / world-state consistency",
    "T5": "object & attribute understanding",
    "T6": "compositional reasoning",
}
ANSWER_TYPES = ("choice", "short", "json")


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    capability: str
    answer_type: str
    question: str
    images: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    answer: Any = None
    # short answers accept a list of aliases; numeric short answers use `tolerance`
    answers: tuple[str, ...] = ()
    tolerance: float | None = None
    split: str = "test"
    meta: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Task":
        cap = str(d.get("capability", ""))
        if cap not in CAPABILITIES:
            raise ValueError(f"unknown capability {cap!r} in task {d.get('task_id')}")
        at = str(d.get("answer_type", "choice"))
        if at not in ANSWER_TYPES:
            raise ValueError(f"unknown answer_type {at!r} in task {d.get('task_id')}")
        return Task(
            task_id=str(d["task_id"]),
            dataset=str(d.get("dataset", "unknown")),
            capability=cap,
            answer_type=at,
            question=str(d["question"]),
            images=tuple(str(x) for x in (d.get("images") or ())),
            choices=tuple(str(x) for x in (d.get("choices") or ())),
            answer=d.get("answer"),
            answers=tuple(str(x) for x in (d.get("answers") or ())),
            tolerance=None if d.get("tolerance") is None else float(d["tolerance"]),
            split=str(d.get("split", "test")),
            meta=dict(d.get("meta") or {}),
        )


def load_tasks(path: Path | str) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        task = Task.from_dict(json.loads(line))
        if task.task_id in seen:
            raise ValueError(f"duplicate task_id {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def write_tasks(tasks: Iterable[Task], path: Path | str) -> int:
    lines = []
    for t in tasks:
        d = {
            "task_id": t.task_id, "dataset": t.dataset, "capability": t.capability,
            "answer_type": t.answer_type, "question": t.question, "images": list(t.images),
            "choices": list(t.choices), "answer": t.answer, "answers": list(t.answers),
            "tolerance": t.tolerance, "split": t.split, "meta": dict(t.meta),
        }
        lines.append(json.dumps(d, ensure_ascii=False))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


@dataclass(frozen=True)
class Prediction:
    task_id: str
    choice: int | None = None
    text: str | None = None
    value: Any = None  # structured JSON answer
    raw: str | None = None
    error: str | None = None
    template: str | None = None
    latency_s: float | None = None

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Prediction":
        return Prediction(
            task_id=str(d["task_id"]),
            choice=None if d.get("choice") is None else int(d["choice"]),
            text=d.get("text"),
            value=d.get("value"),
            raw=d.get("raw"),
            error=d.get("error"),
            template=d.get("template"),
            latency_s=None if d.get("latency_s") is None else float(d["latency_s"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "choice": self.choice, "text": self.text,
            "value": self.value, "raw": self.raw, "error": self.error,
            "template": self.template, "latency_s": self.latency_s,
        }


def load_predictions(path: Path | str) -> dict[str, Prediction]:
    out: dict[str, Prediction] = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Prediction.from_dict(json.loads(line))
        out[p.task_id] = p
    return out


def write_predictions(preds: Iterable[Prediction], path: Path | str) -> int:
    lines = [json.dumps(p.to_dict(), ensure_ascii=False) for p in preds]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)
