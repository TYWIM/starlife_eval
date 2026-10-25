"""Post-generation validation of the T5/T7 suites and frame files.

Everything here is read-only: it loads the written ``*.jsonl`` files and the
JPEG headers from disk and re-derives the ground truth from each row's ``meta``
structured state, so a silent planner bug cannot slip through into the shipped
data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from . import GENERATOR_VERSION, SEED
from .camera import HEIGHT, WIDTH
from .scenes import (
    T5_ANSWER_INDEX,
    T5_CLASSES,
    T5_OPTIONS,
    T7_ANSWER_INDEX,
    T7_DIRECTIONS,
    T7_MIN_GAP_X,
    T7_MIN_GAP_Y,
    T7_OPTIONS,
    direction_between,
    relation_between,
)

T5_EXPECTED = 600
T7_EXPECTED = 400
T5_PER_CLASS = 150
T7_PER_DIRECTION = 100

_ROW_FIELDS = (
    "task_id",
    "dataset",
    "capability",
    "answer_type",
    "question",
    "images",
    "choices",
    "answer",
    "answers",
    "tolerance",
    "split",
    "meta",
)


class ValidationError(Exception):
    """Raised when a generated suite does not satisfy its contract."""


# --------------------------------------------------------------------- utils
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"missing suite file: {path}")
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt write
            raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValidationError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _check_image(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        raise ValidationError(f"missing image file: {path}")
    size = path.stat().st_size
    with Image.open(path) as handle:
        if handle.format != "JPEG":
            raise ValidationError(f"{path}: expected JPEG, got {handle.format}")
        width, height = handle.size
    if (width, height) != (WIDTH, HEIGHT):
        raise ValidationError(f"{path}: expected {WIDTH}x{HEIGHT}, got {width}x{height}")
    return width, height, size


def _image_inventory(data_root: Path, relative_paths: list[str], seen: dict[str, int]) -> None:
    for relative in relative_paths:
        path = Path(relative)
        if path.is_absolute():
            raise ValidationError(f"image path must be relative to the data root: {relative!r}")
        absolute = data_root / path
        _width, _height, size = _check_image(absolute)
        if relative in seen:
            raise ValidationError(f"image referenced twice: {relative!r}")
        seen[relative] = size


def _row_common_checks(row: dict, dataset: str, capability: str, choices: list[str]) -> None:
    for field in _ROW_FIELDS:
        if field not in row:
            raise ValidationError(f"task {row.get('task_id')!r}: missing field {field!r}")
    if row["dataset"] != dataset:
        raise ValidationError(f"task {row['task_id']!r}: dataset {row['dataset']!r} != {dataset!r}")
    if row["capability"] != capability:
        raise ValidationError(f"task {row['task_id']!r}: capability {row['capability']!r} != {capability!r}")
    if row["answer_type"] != "choice":
        raise ValidationError(f"task {row['task_id']!r}: answer_type must be 'choice'")
    if row["split"] != "test":
        raise ValidationError(f"task {row['task_id']!r}: split must be 'test'")
    if list(row["choices"]) != choices:
        raise ValidationError(f"task {row['task_id']!r}: fixed choices changed")
    if row["answers"] != []:
        raise ValidationError(f"task {row['task_id']!r}: answers must be empty for choice tasks")
    if row["tolerance"] is not None:
        raise ValidationError(f"task {row['task_id']!r}: tolerance must be null")
    if not isinstance(row["answer"], int) or isinstance(row["answer"], bool):
        raise ValidationError(f"task {row['task_id']!r}: answer must be int")
    if not 0 <= row["answer"] < len(choices):
        raise ValidationError(f"task {row['task_id']!r}: answer out of range")
    if not isinstance(row["images"], list) or len(row["images"]) != 2:
        raise ValidationError(f"task {row['task_id']!r}: exactly two images required")
    if not str(row["question"]).strip():
        raise ValidationError(f"task {row['task_id']!r}: empty question")
    meta = row.get("meta") or {}
    if meta.get("generator") != GENERATOR_VERSION:
        raise ValidationError(f"task {row['task_id']!r}: generator version mismatch")
    if meta.get("seed") != SEED:
        raise ValidationError(f"task {row['task_id']!r}: seed mismatch")


def _entity_centers(frame_meta: dict) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for entity in frame_meta["entities"]:
        center = entity["center"]
        centers[str(entity["id"])] = (float(center[0]), float(center[1]))
        bbox = entity["bbox"]
        if not (0.0 <= bbox[0] < bbox[2] <= WIDTH and 0.0 <= bbox[1] < bbox[3] <= HEIGHT):
            raise ValidationError(f"entity {entity['id']!r}: bbox out of image bounds: {bbox}")
    descriptions = [entity["description"] for entity in frame_meta["entities"]]
    if len(set(descriptions)) != len(descriptions):
        raise ValidationError(f"duplicate entity descriptions in frame: {descriptions}")
    return centers


# ------------------------------------------------------------------------ T5
def validate_t5(
    rows: list[dict],
    data_root: Path,
    seen_images: dict[str, int],
    *,
    expected_n: int = T5_EXPECTED,
    strict: bool = True,
) -> dict:
    if len(rows) != expected_n:
        raise ValidationError(f"relation_change: expected {expected_n} rows, got {len(rows)}")
    class_counts: Counter = Counter()
    variant_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    seen_ids: set[str] = set()

    for row in rows:
        task_id = str(row["task_id"])
        if not task_id.startswith("relation_change-"):
            raise ValidationError(f"unexpected task_id prefix: {task_id!r}")
        if task_id in seen_ids:
            raise ValidationError(f"duplicate task_id {task_id!r}")
        seen_ids.add(task_id)
        meta = row["meta"]
        _row_common_checks(row, "relation_change", "T5", list(T5_OPTIONS))

        change_class = str(meta["change_class"])
        if change_class not in T5_CLASSES:
            raise ValidationError(f"task {task_id!r}: unknown change class {change_class!r}")
        class_counts[change_class] += 1
        change_variant = meta.get("change_variant")
        if change_variant is not None:
            # only `changed` tasks carry a variant sub-type ("type"/"bucket")
            variant_counts[str(change_variant)] += 1
        if int(row["answer"]) != T5_ANSWER_INDEX[change_class]:
            raise ValidationError(f"task {task_id!r}: answer index does not match change class")
        answer_counts[int(row["answer"])] += 1

        before_meta = meta["frames"]["before"]
        after_meta = meta["frames"]["after"]
        if before_meta["image"] != row["images"][0] or after_meta["image"] != row["images"][1]:
            raise ValidationError(f"task {task_id!r}: meta frame image paths disagree with images[]")
        before_centers = _entity_centers(before_meta)
        after_centers = _entity_centers(after_meta)

        def relation_for(centers: dict[str, tuple[float, float]]) -> dict | None:
            if "e0" not in centers or "e1" not in centers:
                return None
            return relation_between(centers["e0"], centers["e1"])

        relation_before = relation_for(before_centers)
        relation_after = relation_for(after_centers)
        if (meta["relation_before"] is None) != (relation_before is None):
            raise ValidationError(f"task {task_id!r}: relation_before presence mismatch")
        if (meta["relation_after"] is None) != (relation_after is None):
            raise ValidationError(f"task {task_id!r}: relation_after presence mismatch")
        if relation_before is not None and meta["relation_before"]["signature"] != relation_before["signature"]:
            raise ValidationError(f"task {task_id!r}: relation_before signature mismatch")
        if relation_after is not None and meta["relation_after"]["signature"] != relation_after["signature"]:
            raise ValidationError(f"task {task_id!r}: relation_after signature mismatch")

        before_ids = {str(entity["id"]) for entity in before_meta["entities"]}
        after_ids = {str(entity["id"]) for entity in after_meta["entities"]}
        target_a = str(meta["target"]["a"]["description"])
        target_b = str(meta["target"]["b"]["description"])
        if not target_a or not target_b or target_a == target_b:
            raise ValidationError(f"task {task_id!r}: invalid target descriptions")
        if target_a not in row["question"] or target_b not in row["question"]:
            raise ValidationError(f"task {task_id!r}: question does not mention both targets")

        if change_class == "appeared":
            if relation_before is not None or relation_after is None:
                raise ValidationError(f"task {task_id!r}: appeared semantics violated")
            if len(after_ids - before_ids) != 1 or before_ids - after_ids:
                raise ValidationError(f"task {task_id!r}: appeared must add exactly one object")
        elif change_class == "disappeared":
            if relation_before is None or relation_after is not None:
                raise ValidationError(f"task {task_id!r}: disappeared semantics violated")
            if len(before_ids - after_ids) != 1 or after_ids - before_ids:
                raise ValidationError(f"task {task_id!r}: disappeared must remove exactly one object")
        elif change_class == "changed":
            if relation_before is None or relation_after is None:
                raise ValidationError(f"task {task_id!r}: changed needs a relation in both frames")
            if relation_before["signature"] == relation_after["signature"]:
                raise ValidationError(f"task {task_id!r}: changed signature is identical")
            if meta.get("change_variant") == "type" and relation_before["type"] == relation_after["type"]:
                raise ValidationError(f"task {task_id!r}: type-change variant has equal types")
            if meta.get("change_variant") == "bucket" and (
                relation_before["type"] != relation_after["type"]
                or relation_before["bucket"] == relation_after["bucket"]
            ):
                raise ValidationError(f"task {task_id!r}: bucket-change variant mismatch")
        else:  # stayed_same
            if relation_before is None or relation_after is None:
                raise ValidationError(f"task {task_id!r}: stayed_same needs a relation in both frames")
            if relation_before["signature"] != relation_after["signature"]:
                raise ValidationError(f"task {task_id!r}: stayed_same signature changed")
            if before_ids != after_ids:
                raise ValidationError(f"task {task_id!r}: stayed_same object sets differ")

        _image_inventory(data_root, [str(path) for path in row["images"]], seen_images)

    if strict:
        if set(class_counts) != set(T5_CLASSES):
            raise ValidationError(f"relation_change: missing classes: {set(T5_CLASSES) - set(class_counts)}")
        for change_class in T5_CLASSES:
            if class_counts[change_class] != T5_PER_CLASS:
                raise ValidationError(
                    f"relation_change: class {change_class!r} has {class_counts[change_class]} != {T5_PER_CLASS}"
                )
        for index in range(4):
            if answer_counts[index] != T5_PER_CLASS:
                raise ValidationError(f"relation_change: answer index {index} unbalanced: {answer_counts[index]}")
    return {
        "n": len(rows),
        "dataset": "relation_change",
        "capability": "T5",
        "change_classes": dict(sorted(class_counts.items())),
        "change_variants": dict(sorted(variant_counts.items(), key=lambda item: str(item[0]))),
        "answer_index": {str(index): answer_counts[index] for index in range(4)},
    }


# ------------------------------------------------------------------------ T7
def validate_t7(
    rows: list[dict],
    data_root: Path,
    seen_images: dict[str, int],
    *,
    expected_n: int = T7_EXPECTED,
    strict: bool = True,
) -> dict:
    if len(rows) != expected_n:
        raise ValidationError(f"cross_frame: expected {expected_n} rows, got {len(rows)}")
    direction_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    seen_ids: set[str] = set()

    for row in rows:
        task_id = str(row["task_id"])
        if not task_id.startswith("cross_frame-"):
            raise ValidationError(f"unexpected task_id prefix: {task_id!r}")
        if task_id in seen_ids:
            raise ValidationError(f"duplicate task_id {task_id!r}")
        seen_ids.add(task_id)
        meta = row["meta"]
        _row_common_checks(row, "cross_frame", "T7", list(T7_OPTIONS))

        direction = str(meta["direction"])
        if direction not in T7_DIRECTIONS:
            raise ValidationError(f"task {task_id!r}: unknown direction {direction!r}")
        direction_counts[direction] += 1
        if int(row["answer"]) != T7_ANSWER_INDEX[direction]:
            raise ValidationError(f"task {task_id!r}: answer index does not match direction")
        answer_counts[int(row["answer"])] += 1

        image1 = meta["image1"]
        image2 = meta["image2"]
        if image1["image"] != row["images"][0] or image2["image"] != row["images"][1]:
            raise ValidationError(f"task {task_id!r}: meta image paths disagree with images[]")
        if len(image1["entities"]) != 3 or len(image2["entities"]) != 3:
            raise ValidationError(f"task {task_id!r}: each frame must contain exactly 3 objects")
        descriptions_1 = {str(entity["description"]) for entity in image1["entities"]}
        descriptions_2 = {str(entity["description"]) for entity in image2["entities"]}
        ids_1 = {str(entity["id"]) for entity in image1["entities"]}
        ids_2 = {str(entity["id"]) for entity in image2["entities"]}
        if descriptions_1 & descriptions_2:
            raise ValidationError(f"task {task_id!r}: object descriptions overlap across frames")
        if ids_1 & ids_2:
            raise ValidationError(f"task {task_id!r}: object ids overlap across frames")
        for frame_meta in (image1, image2):
            _entity_centers(frame_meta)
        if meta["target"]["x"]["description"] not in descriptions_1:
            raise ValidationError(f"task {task_id!r}: target X not in the first image")
        if meta["target"]["y"]["description"] not in descriptions_2:
            raise ValidationError(f"task {task_id!r}: target Y not in the second image")
        if meta["target"]["x"]["description"] not in row["question"] or meta["target"]["y"]["description"] not in row["question"]:
            raise ValidationError(f"task {task_id!r}: question does not mention both targets")

        x_center = (float(meta["target"]["x"]["center"][0]), float(meta["target"]["x"]["center"][1]))
        y_center = (float(meta["target"]["y"]["center"][0]), float(meta["target"]["y"]["center"][1]))
        info = direction_between(x_center, y_center)
        if info is None:
            raise ValidationError(f"task {task_id!r}: ambiguous cross-frame direction")
        if info["direction"] != direction:
            raise ValidationError(f"task {task_id!r}: direction mismatch {info['direction']} != {direction}")
        minimum_gap = T7_MIN_GAP_X if info["dominant_axis"] == "x" else T7_MIN_GAP_Y
        if info["main_axis_gap_px"] < minimum_gap:
            raise ValidationError(f"task {task_id!r}: main-axis gap below the 8% rule")
        if info["dominant_ratio"] is not None and info["dominant_ratio"] < 1.5:
            raise ValidationError(f"task {task_id!r}: direction dominance below 1.5x")

        _image_inventory(data_root, [str(path) for path in row["images"]], seen_images)

    if strict:
        if set(direction_counts) != set(T7_DIRECTIONS):
            raise ValidationError(f"cross_frame: missing directions: {set(T7_DIRECTIONS) - set(direction_counts)}")
        for direction in T7_DIRECTIONS:
            if direction_counts[direction] != T7_PER_DIRECTION:
                raise ValidationError(
                    f"cross_frame: direction {direction!r} has {direction_counts[direction]} != {T7_PER_DIRECTION}"
                )
        for index in range(4):
            if answer_counts[index] != T7_PER_DIRECTION:
                raise ValidationError(f"cross_frame: answer index {index} unbalanced: {answer_counts[index]}")
    return {
        "n": len(rows),
        "dataset": "cross_frame",
        "capability": "T7",
        "directions": dict(sorted(direction_counts.items())),
        "answer_index": {str(index): answer_counts[index] for index in range(4)},
    }


def verify_sha256sums(data_dir: Path) -> dict:
    """Verify ``SHA256SUMS`` covers every data file and every hash matches."""
    sums_path = data_dir / "SHA256SUMS"
    if not sums_path.exists():
        raise ValidationError(f"missing {sums_path}")
    listed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    actual = {
        path.relative_to(data_dir).as_posix(): sha256_file(path)
        for path in data_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(listed) != set(actual):
        missing = sorted(set(actual) - set(listed))
        extra = sorted(set(listed) - set(actual))
        raise ValidationError(f"SHA256SUMS mismatch: missing={missing[:5]} extra={extra[:5]}")
    mismatched = [relative for relative, digest in actual.items() if listed[relative] != digest]
    if mismatched:
        raise ValidationError(f"SHA256SUMS hash mismatch for {mismatched[:5]}")
    return {"files": len(actual), "sha256sums_sha256": sha256_file(sums_path)}


def image_inventory_summary(seen_images: dict[str, int], prefix: str) -> dict:
    sizes = [size for relative, size in seen_images.items() if relative.startswith(prefix)]
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "bytes_total": int(sum(sizes)),
        "bytes_min": int(min(sizes)),
        "bytes_mean": int(sum(sizes) / len(sizes)),
        "bytes_max": int(max(sizes)),
    }


def validate_all(
    t5_rows: list[dict],
    t7_rows: list[dict],
    data_root: Path,
    *,
    strict: bool = True,
) -> dict:
    """Run every suite/image check; returns a report summary (raises on failure)."""
    seen_images: dict[str, int] = {}
    t5_summary = validate_t5(t5_rows, data_root, seen_images, expected_n=len(t5_rows), strict=strict)
    t7_summary = validate_t7(t7_rows, data_root, seen_images, expected_n=len(t7_rows), strict=strict)
    expected_images = 2 * len(t5_rows) + 2 * len(t7_rows)
    if len(seen_images) != expected_images:
        raise ValidationError(f"expected {expected_images} unique images, found {len(seen_images)}")
    return {
        "t5": t5_summary,
        "t7": t7_summary,
        "images": {
            "t5_frames": image_inventory_summary(seen_images, "openwsv-data-1.1/t5_frames/"),
            "t7_frames": image_inventory_summary(seen_images, "openwsv-data-1.1/t7_frames/"),
            "total_files": len(seen_images),
            "total_bytes": int(sum(seen_images.values())),
        },
    }


__all__ = [
    "ValidationError",
    "sha256_file",
    "load_jsonl",
    "validate_t5",
    "validate_t7",
    "validate_all",
    "verify_sha256sums",
]
