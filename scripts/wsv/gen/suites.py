"""Build the frozen ``*.jsonl`` rows for the T5 / T7 suites.

Row schema matches the existing suites (``task_id, dataset, capability,
answer_type, question, images, choices, answer, answers, tolerance, split,
meta``).  Image paths are relative to the data root ``data/`` and look like
``openwsv-data-1.1/t5_frames/relation_change-000000_before.jpg``.

``meta`` carries the full structured state used to derive the ground truth
(entities with 2D bboxes + relation signatures per frame).  The runner never
sends ``meta`` to a model (only ``question`` / ``images`` / ``choices`` reach
the prompt), it exists for future structured scoring and for auditability.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import GENERATOR_VERSION, SEED
from .camera import (
    CAMERA_POSITION,
    CAMERA_TARGET,
    CX,
    CY,
    FX,
    FY,
    HEIGHT,
    WIDTH,
)
from .objects import bbox_center_of, bbox_size_of, describe, depth_of, image_bbox
from .scenes import T5_ANSWER_INDEX, T5_CLASSES, T5_OPTIONS, T7_ANSWER_INDEX, T7_DIRECTIONS, T7_OPTIONS

DATASET_DIR = "openwsv-data-1.1"
T5_FRAMES_DIR = "t5_frames"
T7_FRAMES_DIR = "t7_frames"
T5_FILE = "relation_change_test.jsonl"
T7_FILE = "cross_frame_test.jsonl"
T5_DATASET = "relation_change"
T7_DATASET = "cross_frame"

CAMERA_META: dict = {
    "model": "pinhole",
    "width": WIDTH,
    "height": HEIGHT,
    "fx": FX,
    "fy": FY,
    "cx": CX,
    "cy": CY,
    "position": [float(value) for value in CAMERA_POSITION],
    "target": [float(value) for value in CAMERA_TARGET],
}


# ------------------------------------------------------------------ helpers
def _round_list(values, digits: int = 2) -> list[float]:
    return [round(float(value), digits) for value in values]


def _image_path(suite_frames_dir: str, task_id: str, suffix: str) -> str:
    return f"{DATASET_DIR}/{suite_frames_dir}/{task_id}_{suffix}.jpg"


def _entity_meta(obj: dict) -> dict:
    bbox = image_bbox(obj)
    if bbox is None:  # pragma: no cover - planners guarantee visibility
        raise ValueError(f"object {obj.get('object_id')!r} is behind the camera")
    center = bbox_center_of(obj)
    return {
        "id": str(obj["object_id"]),
        "description": describe(obj),
        "type": str(obj["kind"]),
        "color": str(obj["color"]),
        "bbox": _round_list(bbox),
        "bbox_size": _round_list(bbox_size_of(obj)),
        "center": _round_list(center),
        "center_3d": _round_list(obj["center"], 4),
        "depth_m": round(float(depth_of(obj)), 4),
    }


def _relations_meta(objects: list[dict]) -> list[dict]:
    """All unambiguous ordered (lower id -> higher id) relations in one frame."""
    from .scenes import relation_between, _object_center  # local import: avoid cycles at import time

    ordered = sorted(objects, key=lambda obj: str(obj["object_id"]))
    relations: list[dict] = []
    for first_index in range(len(ordered)):
        for second_index in range(first_index + 1, len(ordered)):
            first = ordered[first_index]
            second = ordered[second_index]
            relation = relation_between(_object_center(first), _object_center(second))
            if relation is None:
                continue
            relations.append(
                {
                    "from": str(first["object_id"]),
                    "to": str(second["object_id"]),
                    "type": relation["type"],
                    "bucket": relation["bucket"],
                    "signature": relation["signature"],
                    "center_distance_px": relation["center_distance_px"],
                }
            )
    return relations


def _frame_meta(objects: list[dict], image_path: str) -> dict:
    return {
        "image": image_path,
        "entities": [_entity_meta(obj) for obj in objects],
        "relations": _relations_meta(objects),
    }


def _find_object(frames: list[list[dict]], object_id: str) -> dict:
    for objects in frames:
        for obj in objects:
            if str(obj["object_id"]) == object_id:
                return obj
    raise KeyError(f"object {object_id!r} not present in any frame")


def _dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)


# ---------------------------------------------------------------- T5 builder
def t5_row(plan: dict) -> dict:
    """One ``relation_change`` row from a T5 plan."""
    task_id = str(plan["task_id"])
    frames = plan["frames"]
    before_objects = frames["before"]
    after_objects = frames["after"]
    a_object = _find_object([before_objects, after_objects], "e0")
    b_object = _find_object([before_objects, after_objects], "e1")

    before_path = _image_path(T5_FRAMES_DIR, task_id, "before")
    after_path = _image_path(T5_FRAMES_DIR, task_id, "after")
    question = (
        f"How did the relation between the {describe(a_object)} and the "
        f"{describe(b_object)} change between the two frames?"
    )
    meta = {
        "generator": GENERATOR_VERSION,
        "seed": SEED,
        "task_id": task_id,
        "change_class": str(plan["change_class"]),
        "change_variant": plan["variant"],
        "camera": CAMERA_META,
        "target": {
            "a": {
                "id": "e0",
                "description": describe(a_object),
                "type": str(a_object["kind"]),
                "color": str(a_object["color"]),
            },
            "b": {
                "id": "e1",
                "description": describe(b_object),
                "type": str(b_object["kind"]),
                "color": str(b_object["color"]),
            },
        },
        "relation_before": plan["relation_before"],
        "relation_after": plan["relation_after"],
        "frames": {
            "before": _frame_meta(before_objects, before_path),
            "after": _frame_meta(after_objects, after_path),
        },
    }
    if meta["change_class"] not in T5_CLASSES:  # pragma: no cover - planner invariant
        raise ValueError(f"invalid change class {meta['change_class']!r}")
    return {
        "task_id": task_id,
        "dataset": T5_DATASET,
        "capability": "T5",
        "answer_type": "choice",
        "question": question,
        "images": [before_path, after_path],
        "choices": list(T5_OPTIONS),
        "answer": int(plan["answer"]),
        "answers": [],
        "tolerance": None,
        "split": "test",
        "meta": meta,
    }


# ---------------------------------------------------------------- T7 builder
def t7_row(plan: dict) -> dict:
    """One ``cross_frame`` row from a T7 plan."""
    from .scenes import _object_center  # local import: avoid cycles at import time

    task_id = str(plan["task_id"])
    frame1 = plan["frame1"]
    frame2 = plan["frame2"]
    x_object = _find_object([frame1], plan["x_id"])
    y_object = _find_object([frame2], plan["y_id"])
    x_meta = _entity_meta(x_object)
    y_meta = _entity_meta(y_object)
    frame1_path = _image_path(T7_FRAMES_DIR, task_id, "f1")
    frame2_path = _image_path(T7_FRAMES_DIR, task_id, "f2")
    question = (
        f"The {describe(x_object)} appears only in the first image, and the "
        f"{describe(y_object)} appears only in the second image. In the shared camera "
        f"view, where is the {describe(x_object)} relative to the {describe(y_object)}?"
    )
    meta = {
        "generator": GENERATOR_VERSION,
        "seed": SEED,
        "task_id": task_id,
        "camera": CAMERA_META,
        "direction": str(plan["direction"]),
        "direction_info": plan["direction_info"],
        "target": {"x": x_meta, "y": y_meta},
        "image1": _frame_meta(frame1, frame1_path),
        "image2": _frame_meta(frame2, frame2_path),
    }
    if meta["direction"] not in T7_DIRECTIONS:  # pragma: no cover - planner invariant
        raise ValueError(f"invalid direction {meta['direction']!r}")
    return {
        "task_id": task_id,
        "dataset": T7_DATASET,
        "capability": "T7",
        "answer_type": "choice",
        "question": question,
        "images": [frame1_path, frame2_path],
        "choices": list(T7_OPTIONS),
        "answer": int(plan["answer"]),
        "answers": [],
        "tolerance": None,
        "split": "test",
        "meta": meta,
    }


# ------------------------------------------------------------------- writers
def build_rows_t5(plans: list[dict]) -> list[dict]:
    return [t5_row(plan) for plan in plans]


def build_rows_t7(plans: list[dict]) -> list[dict]:
    return [t7_row(plan) for plan in plans]


def write_jsonl(rows: list[dict], path: Path) -> int:
    """Write rows as UTF-8 (no BOM), one compact JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_dumps(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path.stat().st_size


__all__ = [
    "DATASET_DIR",
    "T5_FILE",
    "T7_FILE",
    "T5_DATASET",
    "T7_DATASET",
    "CAMERA_META",
    "T5_ANSWER_INDEX",
    "T7_ANSWER_INDEX",
    "t5_row",
    "t7_row",
    "build_rows_t5",
    "build_rows_t7",
    "write_jsonl",
]
