"""Converters: official dataset release -> unified v2 task JSONL.

The unified schema (see docs/BENCHMARK_V2.md) is a JSONL of tasks:

    {"task_id", "dataset", "capability", "answer_type", "question", "images",
     "choices", "answer", "answers", "tolerance", "split", "meta"}

Converters only read the official test split; they never copy data into this repository. If a
dataset's official release has a new shape, extend the alias tables below (and record the mapping in
docs/BENCHMARK_V2.md) — do not fork the schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import Task, write_tasks

# field aliases seen in spatial-reasoning datasets
IMAGE_KEYS = ("images", "image", "image_path", "image_paths", "img", "rgb", "picture")
QUESTION_KEYS = ("question", "query", "prompt", "text", "instruction")
CHOICE_KEYS = ("choices", "options", "candidates", "answers_choices")
ANSWER_INDEX_KEYS = ("answer_index", "label", "correct", "answer_idx", "target")
ANSWER_TEXT_KEYS = ("answer", "answers", "ground_truth", "gt")


def _first(rec: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def _as_images(value: Any, base_dir: Path | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    else:
        items = [str(v) for v in value]
    out = []
    for item in items:
        p = Path(item)
        if base_dir is not None and not p.is_absolute():
            p = base_dir / p
        out.append(str(p))
    return tuple(out)


def _as_choices(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(str(value[k]) for k in sorted(value))
    return tuple(str(v) for v in value)


def _normalize_choice_record(rec: Mapping[str, Any], *, dataset: str, capability: str,
                             base_dir: Path | None, split: str = "test") -> Task:
    choices = _as_choices(_first(rec, CHOICE_KEYS))
    if not choices:
        raise ValueError(f"{dataset}: record without choices; inspect the raw format first")
    answer = _first(rec, ANSWER_INDEX_KEYS)
    if answer is None:
        text = _first(rec, ANSWER_TEXT_KEYS)
        if text is None:
            raise ValueError(f"{dataset}: record without answer")
        if isinstance(text, str) and text.strip().upper() in [chr(65 + i) for i in range(len(choices))]:
            answer = ord(text.strip().upper()) - 65
        else:
            try:
                answer = choices.index(str(text))
            except ValueError as exc:
                raise ValueError(f"{dataset}: answer text {text!r} not in choices") from exc
    task_id = str(rec.get("id") or rec.get("task_id") or rec.get("uid") or "")
    if not task_id:
        raise ValueError(f"{dataset}: record without id; add a stable id in the converter")
    return Task(
        task_id=f"{dataset}-{task_id}",
        dataset=dataset,
        capability=capability,
        answer_type="choice",
        question=str(_first(rec, QUESTION_KEYS) or ""),
        images=_as_images(_first(rec, IMAGE_KEYS), base_dir),
        choices=choices,
        answer=int(answer),
        split=str(rec.get("split", split)),
        meta={k: v for k, v in rec.items() if k in ("difficulty", "question_type", "scene", "category")},
    )


def convert_jsonl_choice(raw: Path, out: Path, *, dataset: str, capability: str,
                         split: str = "test") -> int:
    """Generic converter for official releases that are a JSONL of multiple-choice records."""
    tasks: list[Task] = []
    base = raw.parent
    with raw.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = _normalize_choice_record(rec, dataset=dataset, capability=capability,
                                            base_dir=base, split=split)
            if task.split != split:
                continue
            tasks.append(task)
    return write_tasks(tasks, out)


# --- per-dataset entry points -------------------------------------------------
# Implemented once the official release is downloaded and its fields are confirmed; until then the
# converter attempts the generic JSONL mapping and fails with an explicit message.

def convert_spatialmqa(raw: Path, out: Path) -> int:
    return convert_jsonl_choice(raw, out, dataset="spatialmqa", capability="T1")


def convert_spatial457(raw: Path, out: Path) -> int:
    return convert_jsonl_choice(raw, out, dataset="spatial457", capability="T2")


def convert_mindedit(raw: Path, out: Path) -> int:
    return convert_jsonl_choice(raw, out, dataset="mindedit", capability="T3")


def convert_spatialdise(raw: Path, out: Path) -> int:
    return convert_jsonl_choice(raw, out, dataset="spatialdise", capability="T1")


def convert_spatial3d(raw: Path, out: Path) -> int:
    raise NotImplementedError(
        "spatial3d diagnostic tasks are generated from dataset_spatial_v3 pairs; implement "
        "convert_spatial3d once the T4/T5 question templates are frozen (docs/BENCHMARK_V2.md)."
    )


CONVERTERS = {
    "spatialmqa": convert_spatialmqa,
    "spatial457": convert_spatial457,
    "mindedit": convert_mindedit,
    "spatialdise": convert_spatialdise,
    "spatial3d": convert_spatial3d,
}


def convert(dataset: str, raw: Path, out: Path) -> int:
    if dataset not in CONVERTERS:
        raise KeyError(f"no converter for {dataset!r}; known: {sorted(CONVERTERS)}")
    return CONVERTERS[dataset](raw, out)
