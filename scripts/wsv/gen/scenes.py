"""Deterministic task planners for the T5 (relation change) and T7 (cross frame) suites.

Both planners are pure functions of ``(task_index, seed)``: the per-task RNG is
derived from ``SeedSequence([seed, suite_code, index])``, so limiting a run
(``--limit``) never changes the scenes of the tasks that are generated, and two
full runs produce byte-identical data.

Ground-truth conventions
------------------------

* Image coordinates are pixels, ``u`` to the right and ``v`` downwards
  (1024x768, fx=fy=800, cx=512, cy=384).
* A relation between A and B is the direction of A's projected 2D bbox centre
  relative to B's: ``left`` / ``right`` / ``above`` / ``below`` (A relative to
  B), plus a distance bucket ``close`` / ``mid`` / ``far``.  The relation
  signature is ``"<type>|<bucket>"``.
* A relation only exists in a frame when *both* objects are present and their
  projected centres are unambiguous: the dominant axis must dominate the other
  axis by >=1.5x and the centre distance must sit inside a bucket range (with a
  guard band around bucket edges).  The same dominance guard is used for the
  T7 directions, together with the SPEC's minimum main-axis gap (8% of the
  image width / height).

All T5 answers are computed by diffing these 3D-state relations between the
before/after frames; the full structured state is written into each row's
``meta`` (never shown to a model).
"""

from __future__ import annotations

import functools
import math

import numpy as np

from . import SEED
from .camera import CAMERA, WIDTH, HEIGHT, bbox_center
from .objects import (
    COLOR_NAMES,
    TYPES,
    bbox_center_of,
    bbox_size_of,
    image_bbox,
    make_cube,
    make_cylinder,
    make_sphere,
    silhouette_points,
)
from .render import TABLE_X0, TABLE_X1, TABLE_Z0, TABLE_Z1

# --------------------------------------------------------------- task layout
T5_SUITE = "relation_change"
T7_SUITE = "cross_frame"

T5_CLASSES: tuple[str, ...] = ("appeared", "disappeared", "changed", "stayed_same")
T5_ANSWER_INDEX = {"appeared": 0, "disappeared": 1, "changed": 2, "stayed_same": 3}
T5_OPTIONS: tuple[str, ...] = (
    "It appeared.",
    "It disappeared.",
    "It changed.",
    "It stayed the same.",
)

T7_DIRECTIONS: tuple[str, ...] = ("left", "right", "above", "below")
T7_ANSWER_INDEX = {"left": 0, "right": 1, "above": 2, "below": 3}
T7_OPTIONS: tuple[str, ...] = ("Left.", "Right.", "Above.", "Below.")

T5_TOTAL = 600
T7_TOTAL = 400

# ------------------------------------------------------------- visual layout
#: allowed region for an object's projected 2D bbox centre
CENTER_U = (94.0, 930.0)
CENTER_V = (198.0, 460.0)
#: projected bbox half-size target range in pixels
HALF_PX = (21.0, 29.0)
EDGE_MARGIN = 16.0
TOP_LIMIT = 26.0
BOTTOM_LIMIT = 552.0
MIN_BBOX_GAP = 14.0
MIN_BBOX_SIDE = 22.0
MAX_BBOX_SIDE = 104.0
TABLE_MARGIN = 0.015

#: relation distance buckets in projected pixels + guard band around the edges
BUCKET_EDGES = (152.0, 264.0)
BUCKET_GUARD = 24.0
BUCKET_DISTANCES = {
    "close": (104.0, 124.0),
    "mid": (180.0, 236.0),
    "far": (294.0, 380.0),
}
REL_DOMINANCE = 1.5
REL_CONSTRUCT_DOMINANCE = 1.8
REL_MIN_DISTANCE = 58.0
REL_MAX_ANGLE = 0.45  # radians, keeps |cos|/|sin| >= 2.0

#: T7 main-axis minimum gaps (SPEC: 8% of image width / height)
T7_MIN_GAP_X = 0.08 * WIDTH
T7_MIN_GAP_Y = 0.08 * HEIGHT
T7_CONSTRUCT_GAP_MARGIN = 1.15

_INVERSE_DIRECTION = {"left": "right", "right": "left", "above": "below", "below": "above"}
_BASE_VECTOR = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "above": (0.0, -1.0),
    "below": (0.0, 1.0),
}

#: T7 primary offset ranges (horizontal / vertical, pixels)
T7_H_PRIMARY = (150.0, 320.0)
T7_V_PRIMARY = (105.0, 200.0)
T7_SECONDARY_FRACTION = 1.0 / 2.5


# --------------------------------------------------------------------- helpers
def task_rng(seed: int, suite_code: int, index: int) -> np.random.Generator:
    """Per-task RNG: independent of the number of tasks generated."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(suite_code), int(index)]))


@functools.lru_cache(maxsize=None)
def t5_class_schedule(n: int = T5_TOTAL, seed: int = SEED) -> list[str]:
    """Balanced class schedule: blocks of 4 contain every class once."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 5, 0]))
    schedule: list[str] = []
    blocks = n // len(T5_CLASSES)
    for _ in range(blocks):
        schedule.extend(T5_CLASSES[int(i)] for i in rng.permutation(len(T5_CLASSES)))
    for i in range(n - blocks * len(T5_CLASSES)):  # only for non-multiple-of-4 smoke runs
        schedule.append(T5_CLASSES[i % len(T5_CLASSES)])
    return schedule


@functools.lru_cache(maxsize=None)
def t7_direction_schedule(n: int = T7_TOTAL, seed: int = SEED) -> list[str]:
    """Balanced direction schedule: blocks of 4 contain every direction once."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7, 0]))
    schedule: list[str] = []
    blocks = n // len(T7_DIRECTIONS)
    for _ in range(blocks):
        schedule.extend(T7_DIRECTIONS[int(i)] for i in rng.permutation(len(T7_DIRECTIONS)))
    for i in range(n - blocks * len(T7_DIRECTIONS)):
        schedule.append(T7_DIRECTIONS[i % len(T7_DIRECTIONS)])
    return schedule


def _draw_colors(rng: np.random.Generator, count: int) -> list[str]:
    return [str(name) for name in rng.permutation(np.array(COLOR_NAMES))[:count]]


def _draw_kinds(rng: np.random.Generator, count: int) -> list[str]:
    kinds = [str(kind) for kind in rng.choice(np.array(TYPES), size=count, replace=True)]
    if count >= 3 and len(set(kinds)) == 1:  # avoid an all-one-primitive scene
        kinds[1] = TYPES[(TYPES.index(kinds[1]) + 1) % len(TYPES)]
        kinds[2] = TYPES[(TYPES.index(kinds[2]) + 2) % len(TYPES)]
    return kinds


def _sample_screen(rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.uniform(*CENTER_U)), float(rng.uniform(*CENTER_V)))


def _object_center(obj: dict) -> tuple[float, float]:
    center = bbox_center_of(obj)
    if center is None:  # pragma: no cover - guarded by validation
        raise ValueError("object behind camera")
    return center


def _object_bbox(obj: dict) -> tuple[float, float, float, float]:
    bbox = image_bbox(obj)
    if bbox is None:  # pragma: no cover - guarded by validation
        raise ValueError("object behind camera")
    return bbox


def _xz_extent(obj: dict) -> tuple[float, float, float, float]:
    """World footprint ``(xmin, xmax, zmin, zmax)`` of the object on the table."""
    kind = obj["kind"]
    center_x = float(obj["center"][0])
    center_z = float(obj["center"][2])
    if kind in ("cylinder", "sphere"):
        radius = float(obj["radius"])
        return (center_x - radius, center_x + radius, center_z - radius, center_z + radius)
    points = silhouette_points(obj)
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    )


def _on_table(obj: dict) -> bool:
    xmin, xmax, zmin, zmax = _xz_extent(obj)
    return (
        xmin >= TABLE_X0 + TABLE_MARGIN
        and xmax <= TABLE_X1 - TABLE_MARGIN
        and zmin >= TABLE_Z0 + TABLE_MARGIN
        and zmax <= TABLE_Z1 - TABLE_MARGIN
    )


def _bbox_visible(bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return (
        x0 >= EDGE_MARGIN
        and y0 >= TOP_LIMIT
        and x1 <= WIDTH - EDGE_MARGIN
        and y1 <= BOTTOM_LIMIT
    )


def _object_ok(obj: dict) -> bool:
    bbox = image_bbox(obj)
    if bbox is None:
        return False
    if not _bbox_visible(bbox):
        return False
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return MIN_BBOX_SIDE <= width <= MAX_BBOX_SIDE and MIN_BBOX_SIDE <= height <= MAX_BBOX_SIDE


def _bbox_gap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Largest axis-aligned gap between two bboxes (negative when they overlap)."""
    dx = max(first[0] - second[2], second[0] - first[2])
    dy = max(first[1] - second[3], second[1] - first[3])
    return max(dx, dy)


def _frame_ok(objects: list[dict]) -> bool:
    seen_ids: set[str] = set()
    bboxes: list[tuple[float, float, float, float]] = []
    for obj in objects:
        object_id = str(obj["object_id"])
        if object_id in seen_ids:
            return False
        seen_ids.add(object_id)
        if not _object_ok(obj) or not _on_table(obj):
            return False
        bboxes.append(_object_bbox(obj))
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bbox_gap(bboxes[i], bboxes[j]) < MIN_BBOX_GAP:
                return False
    return True


def _pick_half_px(rng: np.random.Generator) -> float:
    return float(rng.uniform(*HALF_PX))


def _new_object_at(
    rng: np.random.Generator,
    object_id: str,
    color: str,
    kind: str,
    u: float,
    v: float,
    half_px: float | None = None,
) -> dict | None:
    """Build one object whose projected bbox centre is close to image point (u, v)."""
    half_px = _pick_half_px(rng) if half_px is None else float(half_px)
    center_height = 0.05
    built: dict | None = None
    for _ in range(3):
        point = CAMERA.point_at_height(u, v, center_height)
        if point is None:
            return None
        depth = CAMERA.depth(point)
        if depth < 0.42 or depth > 3.2:
            return None
        meters_per_px = depth / 800.0
        if kind == "cube":
            half = half_px * meters_per_px * 0.82
            built = make_cube(
                object_id,
                color,
                (float(point[0]), half, float(point[2])),
                (half, half, half),
                float(rng.uniform(-0.5, 0.5)),
            )
            center_height = half
        elif kind == "cylinder":
            radius = half_px * meters_per_px * 0.86
            height = radius * 2.0 * float(rng.uniform(1.05, 1.55))
            built = make_cylinder(
                object_id,
                color,
                (float(point[0]), height / 2.0, float(point[2])),
                radius,
                height,
            )
            center_height = height / 2.0
        else:
            radius = half_px * meters_per_px * 0.92
            built = make_sphere(
                object_id,
                color,
                (float(point[0]), radius, float(point[2])),
                radius,
            )
            center_height = radius
    return built


def _move_object_to(obj: dict, u: float, v: float) -> dict | None:
    """Move an object (keeping its physical size) so its projected centre nears (u, v)."""
    center_height = float(obj["center"][1])
    point = CAMERA.point_at_height(u, v, center_height)
    if point is None:
        return None
    moved = dict(obj)
    moved["center"] = [float(point[0]), center_height, float(point[2])]
    return moved


# ------------------------------------------------------------------ relations
def relation_between(
    center_a: tuple[float, float],
    center_b: tuple[float, float],
) -> dict | None:
    """Guarded relation of A relative to B (``None`` when ambiguous)."""
    du = center_a[0] - center_b[0]
    dv = center_a[1] - center_b[1]
    abs_du = abs(du)
    abs_dv = abs(dv)
    dominant = max(abs_du, abs_dv)
    minor = min(abs_du, abs_dv)
    if dominant < REL_MIN_DISTANCE:
        return None
    if minor * REL_DOMINANCE > dominant:
        return None  # too diagonal / not clearly dominant
    distance = math.hypot(du, dv)
    low, high = BUCKET_EDGES
    if distance <= low - BUCKET_GUARD:
        bucket = "close"
    elif low + BUCKET_GUARD <= distance <= high - BUCKET_GUARD:
        bucket = "mid"
    elif distance >= high + BUCKET_GUARD:
        bucket = "far"
    else:
        return None  # inside a guard band around a bucket edge
    if abs_du >= abs_dv:
        relation_type = "right" if du > 0 else "left"
    else:
        relation_type = "below" if dv > 0 else "above"
    return {
        "type": relation_type,
        "bucket": bucket,
        "signature": f"{relation_type}|{bucket}",
        "center_distance_px": round(distance, 2),
        "du_px": round(du, 2),
        "dv_px": round(dv, 2),
    }


def direction_between(
    center_x: tuple[float, float],
    center_y: tuple[float, float],
) -> dict | None:
    """Guarded T7 direction of X relative to Y (image plane), with SPEC margins."""
    du = center_x[0] - center_y[0]
    dv = center_x[1] - center_y[1]
    abs_du = abs(du)
    abs_dv = abs(dv)
    if abs_du >= abs_dv:
        axis = "x"
        gap = abs_du
        minor = abs_dv
        minimum_gap = T7_MIN_GAP_X
        direction = "right" if du > 0 else "left"
    else:
        axis = "y"
        gap = abs_dv
        minor = abs_du
        minimum_gap = T7_MIN_GAP_Y
        direction = "below" if dv > 0 else "above"
    if gap < minimum_gap:
        return None
    ratio = gap / minor if minor > 0.0 else float("inf")
    if ratio < 1.5:
        return None
    return {
        "direction": direction,
        "dominant_axis": axis,
        "main_axis_gap_px": round(gap, 2),
        "dominant_ratio": round(ratio, 3) if math.isfinite(ratio) else None,
        "du_px": round(du, 2),
        "dv_px": round(dv, 2),
    }


def _sample_relation_offset(rng: np.random.Generator, relation_type: str, bucket: str) -> tuple[float, float]:
    distance = float(rng.uniform(*BUCKET_DISTANCES[bucket]))
    angle = float(rng.uniform(-REL_MAX_ANGLE, REL_MAX_ANGLE))
    base = _BASE_VECTOR[relation_type]
    perpendicular = (-base[1], base[0])
    offset_u = distance * (base[0] * math.cos(angle) + perpendicular[0] * math.sin(angle))
    offset_v = distance * (base[1] * math.cos(angle) + perpendicular[1] * math.sin(angle))
    return (offset_u, offset_v)


def _relation_offset_spec(rng: np.random.Generator) -> tuple[str, str]:
    relation_type = str(rng.choice(np.array(("left", "right", "above", "below"))))
    bucket = str(rng.choice(np.array(("close", "mid", "far"))))
    return relation_type, bucket


def _place_standalone(
    rng: np.random.Generator,
    object_id: str,
    color: str,
    kind: str,
) -> dict | None:
    for _ in range(120):
        u, v = _sample_screen(rng)
        candidate = _new_object_at(rng, object_id, color, kind, u, v)
        if candidate is None:
            continue
        if _object_ok(candidate) and _on_table(candidate):
            return candidate
    return None


def _place_related(
    rng: np.random.Generator,
    object_id: str,
    color: str,
    kind: str,
    anchor: dict,
    relation_type: str,
    bucket: str,
) -> dict | None:
    anchor_center = _object_center(anchor)
    anchor_bbox = _object_bbox(anchor)
    for _ in range(90):
        offset_u, offset_v = _sample_relation_offset(rng, relation_type, bucket)
        candidate = _new_object_at(
            rng,
            object_id,
            color,
            kind,
            anchor_center[0] + offset_u,
            anchor_center[1] + offset_v,
        )
        if candidate is None:
            continue
        if not _object_ok(candidate) or not _on_table(candidate):
            continue
        if _bbox_gap(_object_bbox(candidate), anchor_bbox) < MIN_BBOX_GAP:
            continue
        relation = relation_between(_object_center(candidate), anchor_center)
        if relation is None or relation["type"] != relation_type or relation["bucket"] != bucket:
            continue
        return candidate
    return None


def _move_related(
    rng: np.random.Generator,
    obj: dict,
    anchor: dict,
    relation_type: str,
    bucket: str,
) -> dict | None:
    anchor_center = _object_center(anchor)
    anchor_bbox = _object_bbox(anchor)
    for _ in range(120):
        offset_u, offset_v = _sample_relation_offset(rng, relation_type, bucket)
        candidate = _move_object_to(obj, anchor_center[0] + offset_u, anchor_center[1] + offset_v)
        if candidate is None:
            continue
        if not _object_ok(candidate) or not _on_table(candidate):
            continue
        if _bbox_gap(_object_bbox(candidate), anchor_bbox) < MIN_BBOX_GAP:
            continue
        relation = relation_between(_object_center(candidate), anchor_center)
        if relation is None or relation["type"] != relation_type or relation["bucket"] != bucket:
            continue
        return candidate
    return None


def _place_extra(
    rng: np.random.Generator,
    object_id: str,
    color: str,
    kind: str,
    frame_objects: list[list[dict]],
) -> dict | None:
    """Place an object that must be valid & separated in every given frame."""
    for _ in range(140):
        u, v = _sample_screen(rng)
        candidate = _new_object_at(rng, object_id, color, kind, u, v)
        if candidate is None:
            continue
        if not _object_ok(candidate) or not _on_table(candidate):
            continue
        candidate_bbox = _object_bbox(candidate)
        if all(
            _bbox_gap(candidate_bbox, _object_bbox(other)) >= MIN_BBOX_GAP
            for objects in frame_objects
            for other in objects
        ):
            return candidate
    return None


def _move_extra(
    rng: np.random.Generator,
    obj: dict,
    frame_objects: list[list[dict]],
    min_displacement: float,
) -> dict | None:
    """Move an object (keeping its size) to a valid spot in the given frames."""
    origin_center = _object_center(obj)
    for _ in range(160):
        u, v = _sample_screen(rng)
        candidate = _move_object_to(obj, u, v)
        if candidate is None:
            continue
        if not _object_ok(candidate) or not _on_table(candidate):
            continue
        center = _object_center(candidate)
        if math.hypot(center[0] - origin_center[0], center[1] - origin_center[1]) < min_displacement:
            continue
        candidate_bbox = _object_bbox(candidate)
        if all(
            _bbox_gap(candidate_bbox, _object_bbox(other)) >= MIN_BBOX_GAP
            for objects in frame_objects
            for other in objects
        ):
            return candidate
    return None


# ------------------------------------------------------------ T5 construction
def _try_build_appeared(rng: np.random.Generator) -> dict | None:
    colors = _draw_colors(rng, 4)
    kinds = _draw_kinds(rng, 4)
    specs = {f"e{i}": (colors[i], kinds[i]) for i in range(4)}

    relation_type, bucket = _relation_offset_spec(rng)
    transient_id = "e0" if float(rng.random()) < 0.5 else "e1"
    anchor_id = "e1" if transient_id == "e0" else "e0"
    neighbor_type = relation_type if transient_id == "e1" else _INVERSE_DIRECTION[relation_type]

    anchor = _place_standalone(rng, anchor_id, *specs[anchor_id])
    if anchor is None:
        return None
    transient = _place_related(
        rng, transient_id, *specs[transient_id], anchor, neighbor_type, bucket
    )
    if transient is None:
        return None

    distractors: list[dict] = []
    for distractor_id in ("e2", "e3"):
        placed = _place_extra(
            rng,
            distractor_id,
            *specs[distractor_id],
            frame_objects=[[anchor] + distractors, [anchor, transient] + distractors],
        )
        if placed is None:
            return None
        distractors.append(placed)

    return {
        "change_class": "appeared",
        "variant": None,
        "frames": {
            "before": [anchor] + distractors,
            "after": [anchor, transient] + distractors,
        },
    }


def _try_build_disappeared(rng: np.random.Generator) -> dict | None:
    colors = _draw_colors(rng, 4)
    kinds = _draw_kinds(rng, 4)
    specs = {f"e{i}": (colors[i], kinds[i]) for i in range(4)}

    relation_type, bucket = _relation_offset_spec(rng)
    transient_id = "e0" if float(rng.random()) < 0.5 else "e1"
    anchor_id = "e1" if transient_id == "e0" else "e0"
    neighbor_type = relation_type if transient_id == "e1" else _INVERSE_DIRECTION[relation_type]

    anchor = _place_standalone(rng, anchor_id, *specs[anchor_id])
    if anchor is None:
        return None
    transient = _place_related(
        rng, transient_id, *specs[transient_id], anchor, neighbor_type, bucket
    )
    if transient is None:
        return None

    distractors: list[dict] = []
    for distractor_id in ("e2", "e3"):
        placed = _place_extra(
            rng,
            distractor_id,
            *specs[distractor_id],
            frame_objects=[[anchor, transient] + distractors, [anchor] + distractors],
        )
        if placed is None:
            return None
        distractors.append(placed)

    return {
        "change_class": "disappeared",
        "variant": None,
        "frames": {
            "before": [anchor, transient] + distractors,
            "after": [anchor] + distractors,
        },
    }


def _try_build_changed(rng: np.random.Generator) -> dict | None:
    colors = _draw_colors(rng, 4)
    kinds = _draw_kinds(rng, 4)
    specs = {f"e{i}": (colors[i], kinds[i]) for i in range(4)}

    variant = "type" if float(rng.random()) < 0.6 else "bucket"
    b_obj = _place_standalone(rng, "e1", *specs["e1"])
    if b_obj is None:
        return None

    if variant == "type":
        first_type = str(rng.choice(np.array(("left", "right", "above", "below"))))
        other_types = [d for d in ("left", "right", "above", "below") if d != first_type]
        second_type = str(rng.choice(np.array(other_types)))
        first_bucket = str(rng.choice(np.array(("close", "mid", "far"))))
        second_bucket = str(rng.choice(np.array(("close", "mid", "far"))))
    else:
        first_type = str(rng.choice(np.array(("left", "right", "above", "below"))))
        second_type = first_type
        pair = ("close", "far") if float(rng.random()) < 0.5 else ("far", "close")
        first_bucket, second_bucket = pair

    a_before = _place_related(
        rng, "e0", *specs["e0"], b_obj, first_type, first_bucket
    )
    if a_before is None:
        return None
    a_after = _move_related(rng, a_before, b_obj, second_type, second_bucket)
    if a_after is None:
        return None

    distractors: list[dict] = []
    for distractor_id in ("e2", "e3"):
        placed = _place_extra(
            rng,
            distractor_id,
            *specs[distractor_id],
            frame_objects=[[b_obj, a_before] + distractors, [b_obj, a_after] + distractors],
        )
        if placed is None:
            return None
        distractors.append(placed)

    return {
        "change_class": "changed",
        "variant": variant,
        "frames": {
            "before": [b_obj, a_before] + distractors,
            "after": [b_obj, a_after] + distractors,
        },
    }


def _try_build_stayed(rng: np.random.Generator) -> dict | None:
    colors = _draw_colors(rng, 4)
    kinds = _draw_kinds(rng, 4)
    specs = {f"e{i}": (colors[i], kinds[i]) for i in range(4)}

    relation_type, bucket = _relation_offset_spec(rng)
    b_obj = _place_standalone(rng, "e1", *specs["e1"])
    if b_obj is None:
        return None
    a_obj = _place_related(rng, "e0", *specs["e0"], b_obj, relation_type, bucket)
    if a_obj is None:
        return None

    static_objects = [a_obj, b_obj]
    second_distractor = _place_extra(
        rng, "e3", *specs["e3"], frame_objects=[static_objects, static_objects]
    )
    if second_distractor is None:
        return None

    baseline_objects = static_objects + [second_distractor]
    first_distractor_before = _place_extra(
        rng, "e2", *specs["e2"], frame_objects=[baseline_objects]
    )
    if first_distractor_before is None:
        return None
    first_distractor_after = _move_extra(
        rng,
        first_distractor_before,
        frame_objects=[baseline_objects],
        min_displacement=110.0,
    )
    if first_distractor_after is None:
        return None

    return {
        "change_class": "stayed_same",
        "variant": None,
        "frames": {
            "before": [a_obj, b_obj, first_distractor_before, second_distractor],
            "after": [a_obj, b_obj, first_distractor_after, second_distractor],
        },
    }


def _relation_of_pair(frame_objects: list[dict], a_id: str, b_id: str) -> dict | None:
    by_id = {str(obj["object_id"]): obj for obj in frame_objects}
    if a_id not in by_id or b_id not in by_id:
        return None
    return relation_between(_object_center(by_id[a_id]), _object_center(by_id[b_id]))


def _frame_shift(objects_a: list[dict], objects_b: list[dict]) -> float:
    """Largest projected centre displacement of a shared object id between frames."""
    by_id_b = {str(obj["object_id"]): obj for obj in objects_b}
    largest = 0.0
    for obj in objects_a:
        object_id = str(obj["object_id"])
        if object_id not in by_id_b:
            continue
        first = _object_center(obj)
        second = _object_center(by_id_b[object_id])
        largest = max(largest, math.hypot(first[0] - second[0], first[1] - second[1]))
    return largest


def _validate_t5_plan(plan: dict) -> dict | None:
    """Check class semantics; returns computed relations or ``None`` when invalid."""
    before = plan["frames"]["before"]
    after = plan["frames"]["after"]
    if not _frame_ok(before) or not _frame_ok(after):
        return None

    relation_before = _relation_of_pair(before, "e0", "e1")
    relation_after = _relation_of_pair(after, "e0", "e1")
    change_class = plan["change_class"]

    if change_class == "appeared":
        if relation_before is not None or relation_after is None:
            return None
        if _frame_shift(before, after) > 1.0:  # only the new object may differ
            return None
        before_ids = {str(obj["object_id"]) for obj in before}
        after_ids = {str(obj["object_id"]) for obj in after}
        if len(after_ids - before_ids) != 1 or before_ids - after_ids:
            return None
    elif change_class == "disappeared":
        if relation_before is None or relation_after is not None:
            return None
        if _frame_shift(before, after) > 1.0:
            return None
        before_ids = {str(obj["object_id"]) for obj in before}
        after_ids = {str(obj["object_id"]) for obj in after}
        if len(before_ids - after_ids) != 1 or after_ids - before_ids:
            return None
    elif change_class == "changed":
        if relation_before is None or relation_after is None:
            return None
        if relation_before["signature"] == relation_after["signature"]:
            return None
        variant = plan["variant"]
        if variant == "type" and relation_before["type"] == relation_after["type"]:
            return None
        if variant == "bucket" and (
            relation_before["type"] != relation_after["type"]
            or relation_before["bucket"] == relation_after["bucket"]
        ):
            return None
        before_by_id = {str(obj["object_id"]): obj for obj in before}
        after_by_id = {str(obj["object_id"]): obj for obj in after}
        first = _object_center(before_by_id["e0"])
        second = _object_center(after_by_id["e0"])
        if math.hypot(first[0] - second[0], first[1] - second[1]) < 90.0:
            return None
        if _frame_shift(
            [obj for obj in before if str(obj["object_id"]) not in ("e0",)],
            [obj for obj in after if str(obj["object_id"]) not in ("e0",)],
        ) > 1.0:
            return None
    else:  # stayed_same
        if relation_before is None or relation_after is None:
            return None
        if relation_before["signature"] != relation_after["signature"]:
            return None
        # target pair must be pixel-identical; the frames must still differ
        # somewhere else, otherwise "same" would be trivially guessable.
        by_id_before = {str(obj["object_id"]): obj for obj in before}
        by_id_after = {str(obj["object_id"]): obj for obj in after}
        if set(by_id_before) != set(by_id_after):
            return None
        target_shift = 0.0
        other_shift = 0.0
        for object_id, first_obj in by_id_before.items():
            first = _object_center(first_obj)
            second = _object_center(by_id_after[object_id])
            shift = math.hypot(first[0] - second[0], first[1] - second[1])
            if object_id in ("e0", "e1"):
                target_shift = max(target_shift, shift)
            else:
                other_shift = max(other_shift, shift)
        if target_shift > 0.5 or other_shift < 100.0:
            return None
    return {
        "relation_before": relation_before,
        "relation_after": relation_after,
    }


def plan_t5_task(index: int, seed: int = SEED) -> dict:
    """Plan one T5 task (pure/deterministic in ``index`` and ``seed``)."""
    schedule = t5_class_schedule(seed=seed)
    change_class = schedule[index]
    rng = task_rng(seed, 5, index)
    builders = {
        "appeared": _try_build_appeared,
        "disappeared": _try_build_disappeared,
        "changed": _try_build_changed,
        "stayed_same": _try_build_stayed,
    }
    builder = builders[change_class]
    for _ in range(40):
        raw = builder(rng)
        if raw is None:
            continue
        validated = _validate_t5_plan(raw)
        if validated is None:
            continue
        task_id = f"{T5_SUITE}-{index:06d}"
        plan = {
            "task_id": task_id,
            "change_class": raw["change_class"],
            "variant": raw["variant"],
            "answer": T5_ANSWER_INDEX[raw["change_class"]],
            "frames": raw["frames"],
            "relation_before": validated["relation_before"],
            "relation_after": validated["relation_after"],
        }
        return plan
    raise RuntimeError(f"failed to plan T5 task {index} ({change_class}) after 40 attempts")


# ------------------------------------------------------------ T7 construction
def _y_screen_region(rng: np.random.Generator, direction: str) -> tuple[float, float]:
    """Screen region for Y such that a valid X placement leaves room."""
    if direction == "right":
        u = float(rng.uniform(110.0, 470.0))
    elif direction == "left":
        u = float(rng.uniform(554.0, 914.0))
    else:
        u = float(rng.uniform(210.0, 814.0))
    if direction == "above":
        v = float(rng.uniform(330.0, 430.0))
    elif direction == "below":
        v = float(rng.uniform(215.0, 315.0))
    else:
        v = float(rng.uniform(240.0, 400.0))
    return (u, v)


def _sample_direction_offset(rng: np.random.Generator, direction: str) -> tuple[float, float]:
    if direction in ("left", "right"):
        primary = float(rng.uniform(*T7_H_PRIMARY))
    else:
        primary = float(rng.uniform(*T7_V_PRIMARY))
    secondary = float(rng.uniform(-primary * T7_SECONDARY_FRACTION, primary * T7_SECONDARY_FRACTION))
    base = _BASE_VECTOR[direction]
    perpendicular = (-base[1], base[0])
    offset_u = primary * base[0] + secondary * perpendicular[0]
    offset_v = primary * base[1] + secondary * perpendicular[1]
    return (offset_u, offset_v)


def _try_build_t7(direction: str, rng: np.random.Generator) -> dict | None:
    colors = _draw_colors(rng, 6)
    kinds = _draw_kinds(rng, 6)
    specs = {f"e{i}": (colors[i], kinds[i]) for i in range(6)}

    y_obj = None
    for _ in range(120):
        u, v = _y_screen_region(rng, direction)
        candidate = _new_object_at(rng, "e1", *specs["e1"], u, v)
        if candidate is None:
            continue
        if _object_ok(candidate) and _on_table(candidate):
            y_obj = candidate
            break
    if y_obj is None:
        return None
    y_center = _object_center(y_obj)

    x_obj = None
    for _ in range(160):
        offset_u, offset_v = _sample_direction_offset(rng, direction)
        candidate = _new_object_at(
            rng, "e0", *specs["e0"], y_center[0] + offset_u, y_center[1] + offset_v
        )
        if candidate is None:
            continue
        if not _object_ok(candidate) or not _on_table(candidate):
            continue
        info = direction_between(_object_center(candidate), y_center)
        if info is None or info["direction"] != direction:
            continue
        if info["main_axis_gap_px"] < (
            T7_CONSTRUCT_GAP_MARGIN
            * (T7_MIN_GAP_X if info["dominant_axis"] == "x" else T7_MIN_GAP_Y)
        ):
            continue
        x_obj = candidate
        break
    if x_obj is None:
        return None

    frame1 = [x_obj]
    frame2 = [y_obj]
    for companion_id in ("e2", "e3"):
        placed = _place_extra(rng, companion_id, *specs[companion_id], frame_objects=[frame1])
        if placed is None:
            return None
        frame1.append(placed)
    for companion_id in ("e4", "e5"):
        placed = _place_extra(rng, companion_id, *specs[companion_id], frame_objects=[frame2])
        if placed is None:
            return None
        frame2.append(placed)

    return {
        "direction": direction,
        "frame1": frame1,
        "frame2": frame2,
        "x_id": "e0",
        "y_id": "e1",
    }


def _validate_t7_plan(plan: dict, direction: str) -> dict | None:
    frame1 = plan["frame1"]
    frame2 = plan["frame2"]
    if len(frame1) != 3 or len(frame2) != 3:
        return None
    if not _frame_ok(frame1) or not _frame_ok(frame2):
        return None
    frame1_ids = {str(obj["object_id"]) for obj in frame1}
    frame2_ids = {str(obj["object_id"]) for obj in frame2}
    if frame1_ids & frame2_ids:
        return None
    if len({str(obj["color"]) for obj in frame1 + frame2}) != 6:
        return None
    by_id_1 = {str(obj["object_id"]): obj for obj in frame1}
    by_id_2 = {str(obj["object_id"]): obj for obj in frame2}
    if plan["x_id"] not in by_id_1 or plan["y_id"] not in by_id_2:
        return None
    info = direction_between(_object_center(by_id_1[plan["x_id"]]), _object_center(by_id_2[plan["y_id"]]))
    if info is None or info["direction"] != direction:
        return None
    if info["main_axis_gap_px"] < (
        T7_CONSTRUCT_GAP_MARGIN
        * (T7_MIN_GAP_X if info["dominant_axis"] == "x" else T7_MIN_GAP_Y)
    ):
        return None
    return info


def plan_t7_task(index: int, seed: int = SEED) -> dict:
    """Plan one T7 task (pure/deterministic in ``index`` and ``seed``)."""
    schedule = t7_direction_schedule(seed=seed)
    direction = schedule[index]
    rng = task_rng(seed, 7, index)
    for _ in range(60):
        raw = _try_build_t7(direction, rng)
        if raw is None:
            continue
        info = _validate_t7_plan(raw, direction)
        if info is None:
            continue
        return {
            "task_id": f"{T7_SUITE}-{index:06d}",
            "direction": direction,
            "answer": T7_ANSWER_INDEX[direction],
            "frame1": raw["frame1"],
            "frame2": raw["frame2"],
            "x_id": raw["x_id"],
            "y_id": raw["y_id"],
            "direction_info": info,
        }
    raise RuntimeError(f"failed to plan T7 task {index} ({direction}) after 60 attempts")


def plan_t5_tasks(count: int = T5_TOTAL, seed: int = SEED) -> list[dict]:
    return [plan_t5_task(index, seed=seed) for index in range(count)]


def plan_t7_tasks(count: int = T7_TOTAL, seed: int = SEED) -> list[dict]:
    return [plan_t7_task(index, seed=seed) for index in range(count)]


__all__ = [
    "T5_SUITE",
    "T7_SUITE",
    "T5_CLASSES",
    "T5_ANSWER_INDEX",
    "T5_OPTIONS",
    "T7_DIRECTIONS",
    "T7_ANSWER_INDEX",
    "T7_OPTIONS",
    "T5_TOTAL",
    "T7_TOTAL",
    "t5_class_schedule",
    "t7_direction_schedule",
    "plan_t5_task",
    "plan_t7_task",
    "plan_t5_tasks",
    "plan_t7_tasks",
    "relation_between",
    "direction_between",
]
