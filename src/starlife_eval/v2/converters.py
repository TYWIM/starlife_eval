"""Converters: official dataset release -> unified v2 task JSONL.

The unified schema (see docs/BENCHMARK_V2.md) is a JSONL of tasks:

    {"task_id", "dataset", "capability", "answer_type", "question", "images",
     "choices", "answer", "answers", "tolerance", "split", "meta"}

Converters only read the official test/dev split; they never copy data into this repository
automatically — the packaging step decides what is redistributable (licenses differ per dataset).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import Task, write_tasks

IMAGE_KEYS = ("images", "image", "image_path", "image_paths", "img", "rgb", "picture")
QUESTION_KEYS = ("question", "query", "prompt", "text", "instruction")
CHOICE_KEYS = ("choices", "options", "candidates")
ANSWER_INDEX_KEYS = ("answer_index", "label", "correct", "answer_idx", "target")
ANSWER_TEXT_KEYS = ("answer", "answers", "ground_truth", "gt")


def _first(rec: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def _short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _as_images(value: Any, base_dir: Path | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else [str(v) for v in value]
    out = []
    for item in items:
        p = Path(item)
        if base_dir is not None and not p.is_absolute():
            candidate = base_dir / p
            if not candidate.exists() and (base_dir / "images" / p.name).exists():
                candidate = base_dir / "images" / p.name
            p = candidate
        out.append(str(p))
    return tuple(out)


def _as_choices(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return tuple(value.split("|")) if "|" in value else (value,)
    if isinstance(value, Mapping):
        return tuple(str(value[k]) for k in sorted(value))
    return tuple(str(v) for v in value)


def _normalize_choice_record(rec: Mapping[str, Any], *, dataset: str, capability: str,
                             base_dir: Path | None, index: int, split: str = "test") -> Task:
    raw_choices = _as_choices(_first(rec, CHOICE_KEYS))
    if not raw_choices:
        raise ValueError(f"{dataset}: record without choices")
    # some releases prefix each option with "A. ", some are plain text
    cleaned: list[str] = []
    for i, c in enumerate(raw_choices):
        prefix = f"{chr(65 + i)}."
        cleaned.append(c[len(prefix):].strip() if c.startswith(prefix) else c.strip())
    choices = tuple(cleaned)

    answer = _first(rec, ANSWER_INDEX_KEYS)
    if answer is None:
        text = str(_first(rec, ANSWER_TEXT_KEYS) or "").strip()
        if not text:
            raise ValueError(f"{dataset}: record without answer")
        if len(text) == 1 and text.upper() in [chr(65 + i) for i in range(len(choices))]:
            answer = ord(text.upper()) - 65
        else:
            try:
                answer = choices.index(text)
            except ValueError as exc:
                raise ValueError(f"{dataset}: answer {text!r} not in choices") from exc
    task_id = str(rec.get("id") or rec.get("task_id") or rec.get("uid") or "")
    if not task_id:
        image = _first(rec, IMAGE_KEYS)
        stem = Path(str(image)).stem if image else f"i{index:05d}"
        task_id = f"{stem}-{_short_hash(str(rec.get('question', '')))}"
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
        meta={k: str(v) for k, v in rec.items()
              if k in ("difficulty", "question_type", "scene", "category", "task", "room", "triplet")},
    )


def convert_jsonl_choice(raw: Path, out: Path, *, dataset: str, capability: str,
                         split: str = "test") -> int:
    tasks: list[Task] = []
    base = raw.parent
    for index, line in enumerate(raw.read_text(encoding="utf-8-sig").splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tasks.append(_normalize_choice_record(rec, dataset=dataset, capability=capability,
                                              base_dir=base, index=index, split=split))
    return write_tasks(tasks, out)


# --- per-dataset entry points -------------------------------------------------

def convert_spatialmqa(raw: Path, out: Path) -> int:
    """SpatialMQA (ACL 2025): dev.jsonl with {image, question, options, answer}; images/ dir."""
    path = raw if raw.is_dir() else raw.parent
    jsonl = path / "dev.jsonl"
    if not jsonl.exists():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no JSONL release under {path}")
        jsonl = candidates[0]
    return convert_jsonl_choice(jsonl, out, dataset="spatialmqa", capability="T1", split="dev")


def convert_mindedit(raw: Path, out: Path) -> int:
    """MindEdit-Bench: parquet task files with embedded view1..view3 image bytes."""
    import pyarrow.parquet as pq

    root = raw if raw.is_dir() else raw.parent
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {root}")
    image_dir = out.parent / "images" / "mindedit"
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[Task] = []
    index = 0
    for pf in parquet_files:
        table = pq.read_table(pf)
        for row in table.to_pylist():
            choices = _as_choices(row.get("choices") or row.get("options"))
            answer = str(row.get("answer", "")).strip()
            if len(answer) == 1 and answer.upper() in [chr(65 + i) for i in range(len(choices))]:
                answer_idx = ord(answer.upper()) - 65
            else:
                try:
                    answer_idx = choices.index(answer)
                except ValueError:
                    continue
            images = []
            for key in ("view1", "view2", "view3", "image", "input_image"):
                blob = row.get(key)
                if isinstance(blob, (bytes, bytearray)):
                    fn = image_dir / f"{row.get('id', f'row{index:05d}')}_{key}.jpg"
                    if not fn.exists():
                        fn.write_bytes(bytes(blob))
                    images.append(str(fn))
                elif isinstance(blob, Mapping) and blob.get("bytes"):
                    fn = image_dir / f"{row.get('id', f'row{index:05d}')}_{key}.jpg"
                    if not fn.exists():
                        fn.write_bytes(bytes(blob["bytes"]))
                    images.append(str(fn))
            task_id = str(row.get("id") or f"row{index:05d}")
            tasks.append(Task(
                task_id=f"mindedit-{task_id}", dataset="mindedit", capability="T3",
                answer_type="choice", question=str(row.get("question", "")),
                images=tuple(images), choices=tuple(choices), answer=answer_idx,
                split="test",
                meta={**{k: str(row.get(k)) for k in ("task", "room", "triplet") if row.get(k) is not None},
                      **({"views": "view1,view2,view3"} if len(images) == 3 else {})},
            ))
            index += 1
    return write_tasks(tasks, out)


def convert_spatial457(raw: Path, out: Path) -> int:
    """Spatial457: image folder + annotations JSON/JSONL (layout finalized after the official dump)."""
    root = raw if raw.is_dir() else raw.parent
    for candidate in sorted(root.rglob("*.jsonl")):
        return convert_jsonl_choice(candidate, out, dataset="spatial457", capability="T2")
    for candidate in sorted(root.rglob("*.json")):
        data = json.loads(candidate.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("data") or data.get("annotations")
        if not records:
            continue
        tmp = out.with_suffix(".tmp.jsonl")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
        try:
            return convert_jsonl_choice(tmp, out, dataset="spatial457", capability="T2")
        finally:
            tmp.unlink(missing_ok=True)
    raise FileNotFoundError(f"no annotations under {root} (data.zip may need extraction)")


def convert_spatialdise(raw: Path, out: Path) -> int:
    """Spatial-DISE: CSV task tables (license CC-BY-NC-SA-4.0 -> download only, not redistributed)."""
    import csv

    root = raw if raw.is_dir() else raw.parent
    csvs = [p for p in root.rglob("*.csv") if p.name in ("test.csv", "val.csv", "DISE-benchmark.csv")]
    if not csvs:
        raise FileNotFoundError(f"no CSV task table under {root}")
    tasks: list[Task] = []
    index = 0
    for c in csvs:
        with c.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    task = _normalize_choice_record(row, dataset="spatialdise", capability="T1",
                                                    base_dir=root, index=index, split="test")
                except Exception:
                    index += 1
                    continue
                tasks.append(task)
                index += 1
    return write_tasks(tasks, out)


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
