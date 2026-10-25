"""Deterministic numpy ray-cast renderer for the shared OpenWSV tabletop room.

The room is fixed (table top at ``y = 0``, backdrop wall at ``z = WALL_Z``,
floor at ``y = FLOOR_Y``, one directional light) and a single shared pinhole
camera looks at it, so both frames of every task show the same room/camera.

Rendering is analytic ray casting against boxes / vertical cylinders / spheres
plus two planes (wall, floor).  No textures and no text are ever drawn; the only
content is geometry and Lambertian shading, which keeps the procedure
deterministic and dependency-free (numpy + Pillow only).

Performance notes: the image is rendered in small horizontal strips (bounded
memory) and every primitive is culled by its projected screen-space window, so
small objects only cost the pixels they actually cover.  This keeps a
1024x768 frame at a fraction of a second to a second of single-core time and a
worker RSS of a few tens of MB (the machine is shared).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .camera import (
    CAMERA,
    CAMERA_POSITION,
    CX,
    CY,
    FX,
    FY,
    HEIGHT,
    WIDTH,
)
from .objects import base_center_of, footprint_radius

#: horizontal + vertical supersampling factor (2 -> 2048x1536 internal buffer)
SUPERSAMPLE = 2

#: internal rows rendered per chunk (memory bound, not quality)
_STRIP_ROWS = 64

# ------------------------------------------------------------------ room layout
TABLE_TOP_Y = 0.0
TABLE_X0, TABLE_X1 = -0.80, 0.80
TABLE_Z0, TABLE_Z1 = 0.38, 1.72
TABLE_THICKNESS = 0.05
WALL_Z = 2.10
FLOOR_Y = -0.78

WALL_TOP_RGB = (168, 173, 184)
WALL_BOTTOM_RGB = (203, 204, 207)
TABLE_RGB = (164, 127, 90)
TABLE_EDGE_RGB = (128, 97, 68)
FLOOR_RGB = (76, 72, 74)
LEG_RGB = (118, 89, 63)

#: direction the light travels (from upper-left-front towards the table)
LIGHT_DIR = np.array([0.34, -0.88, 0.34], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

AMBIENT = 0.46
DIFFUSE = 0.64
SPECULAR = 0.20
SPECULAR_POWER = 26.0

#: JPEG settings for the shipped frames (quality must stay at 90)
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 0

_INTERNAL_W = WIDTH * SUPERSAMPLE
_INTERNAL_H = HEIGHT * SUPERSAMPLE

TABLE_RGB_F32 = np.asarray(TABLE_RGB, dtype=np.float32)
TABLE_EDGE_F32 = np.asarray(TABLE_EDGE_RGB, dtype=np.float32)
WALL_BOTTOM_F32 = np.asarray(WALL_BOTTOM_RGB, dtype=np.float32)
WALL_TOP_F32 = np.asarray(WALL_TOP_RGB, dtype=np.float32)
FLOOR_RGB_F32 = np.asarray(FLOOR_RGB, dtype=np.float32)
LEG_RGB_F32 = np.asarray(LEG_RGB, dtype=np.float32)


def _vec(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------- primitives
def _prim_box(center, half, yaw, rgb, material) -> dict:
    return {
        "kind": "box",
        "center": _vec(center),
        "half": _vec(half),
        "yaw": float(yaw),
        "rgb": _vec(rgb),
        "material": material,
    }


def _prim_cylinder(center, radius, y0, y1, rgb, material) -> dict:
    return {
        "kind": "cylinder",
        "center": _vec(center),
        "radius": float(radius),
        "y0": float(y0),
        "y1": float(y1),
        "rgb": _vec(rgb),
        "material": material,
    }


def _prim_sphere(center, radius, rgb, material) -> dict:
    return {
        "kind": "sphere",
        "center": _vec(center),
        "radius": float(radius),
        "rgb": _vec(rgb),
        "material": material,
    }


def _prim_plane(point, normal, rgb, material, window=None) -> dict:
    return {
        "kind": "plane",
        "point": _vec(point),
        "normal": _vec(normal),
        "rgb": _vec(rgb),
        "material": material,
        "_window": window,
    }


def _rotate_y(vectors: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x = vectors[..., 0]
    z = vectors[..., 2]
    return np.stack(
        [cosine * x + sine * z, vectors[..., 1], -sine * x + cosine * z],
        axis=-1,
    )


def _screen_window(points: np.ndarray, margin: int = 4) -> tuple[int, int, int, int] | None:
    """Projected screen window ``(row0, row1, col0, col1)`` in internal pixels."""
    projected = CAMERA.project(points)
    if np.any(projected[:, 2] <= 0.05):
        return None
    u = projected[:, 0] * SUPERSAMPLE
    v = projected[:, 1] * SUPERSAMPLE
    col0 = int(math.floor(float(u.min()))) - margin
    col1 = int(math.ceil(float(u.max()))) + margin
    row0 = int(math.floor(float(v.min()))) - margin
    row1 = int(math.ceil(float(v.max()))) + margin
    return (
        max(0, min(_INTERNAL_H, row0)),
        max(0, min(_INTERNAL_H, row1)),
        max(0, min(_INTERNAL_W, col0)),
        max(0, min(_INTERNAL_W, col1)),
    )


def _primitive_sample_points(primitive: dict) -> np.ndarray | None:
    """Points whose projection over-approximates the primitive's screen footprint."""
    kind = primitive["kind"]
    center = primitive["center"].astype(np.float64)
    if kind == "box":
        half = primitive["half"].astype(np.float64)
        signs = np.array(
            [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
        return _rotate_y(signs * half, float(primitive["yaw"])) + center
    if kind == "cylinder":
        radius = float(primitive["radius"])
        angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        ring = np.stack(
            [np.cos(angles) * radius, np.zeros_like(angles), np.sin(angles) * radius], axis=-1
        )
        bottom = ring + np.array([center[0], float(primitive["y0"]), center[2]])
        top = ring + np.array([center[0], float(primitive["y1"]), center[2]])
        return np.concatenate([bottom, top], axis=0)
    if kind == "sphere":
        radius = float(primitive["radius"])
        signs = np.array(
            [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
        return center + signs * radius
    return None


def _plane_split_row() -> tuple[int, int]:
    """Row (internal) where the floor takes over from the wall as the far hit."""
    rows = np.arange(_INTERNAL_H, dtype=np.float64)
    v = (rows + 0.5) / SUPERSAMPLE
    dv = -(v - CY) / FY
    direction_y = float(CAMERA.up[1]) * dv + float(CAMERA.forward[1])
    direction_z = float(CAMERA.up[2]) * dv + float(CAMERA.forward[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_t = (FLOOR_Y - float(CAMERA_POSITION[1])) / direction_y
        wall_t = (WALL_Z - float(CAMERA_POSITION[2])) / direction_z
    floor_first = (floor_t > 0.0) & (floor_t < wall_t)
    wall_rows = np.flatnonzero(~floor_first)
    floor_rows = np.flatnonzero(floor_first)
    wall_last = int(wall_rows.max()) + 3 if wall_rows.size else 0
    floor_first_row = int(floor_rows.min()) - 3 if floor_rows.size else _INTERNAL_H
    return wall_last, floor_first_row


def build_primitives(objects: list[dict]) -> list[dict]:
    """Static room primitives + one primitive per object, each with a screen window."""
    table_center = (
        (TABLE_X0 + TABLE_X1) / 2.0,
        TABLE_TOP_Y - TABLE_THICKNESS / 2.0,
        (TABLE_Z0 + TABLE_Z1) / 2.0,
    )
    table_half = (
        (TABLE_X1 - TABLE_X0) / 2.0,
        TABLE_THICKNESS / 2.0,
        (TABLE_Z1 - TABLE_Z0) / 2.0,
    )
    primitives = [_prim_box(table_center, table_half, 0.0, TABLE_RGB, "table")]

    # two front legs (the back pair is never visible from this camera)
    for sign in (-1.0, 1.0):
        leg_center = (
            sign * 0.70,
            (TABLE_TOP_Y - TABLE_THICKNESS + FLOOR_Y) / 2.0,
            TABLE_Z0 + 0.09,
        )
        leg_half = (0.032, (TABLE_TOP_Y - TABLE_THICKNESS - FLOOR_Y) / 2.0, 0.032)
        primitives.append(_prim_box(leg_center, leg_half, 0.0, LEG_RGB, "leg"))

    wall_last, floor_first = _plane_split_row()
    full_width = (0, _INTERNAL_W)
    primitives.append(
        _prim_plane((0.0, 0.0, WALL_Z), (0.0, 0.0, -1.0), WALL_BOTTOM_RGB, "wall",
                    window=(0, wall_last, *full_width))
    )
    primitives.append(
        _prim_plane((0.0, FLOOR_Y, 0.0), (0.0, 1.0, 0.0), FLOOR_RGB, "floor",
                    window=(floor_first, _INTERNAL_H, *full_width))
    )

    for obj in objects:
        kind = obj["kind"]
        rgb = obj["rgb"]
        center = obj["center"]
        if kind == "cube":
            primitives.append(_prim_box(center, obj["half"], obj.get("yaw", 0.0), rgb, "object"))
        elif kind == "cylinder":
            height = float(obj["height"])
            primitives.append(
                _prim_cylinder(
                    center,
                    float(obj["radius"]),
                    float(center[1]) - height / 2.0,
                    float(center[1]) + height / 2.0,
                    rgb,
                    "object",
                )
            )
        elif kind == "sphere":
            primitives.append(_prim_sphere(center, float(obj["radius"]), rgb, "object"))
        else:  # pragma: no cover - guarded by the planner
            raise ValueError(f"unknown object kind {kind!r}")

    for primitive in primitives:
        if "_window" in primitive and primitive["_window"] is not None:
            continue
        samples = _primitive_sample_points(primitive)
        window = _screen_window(samples) if samples is not None else None
        if window is None:  # fall back to the full image (never happens for our camera)
            window = (0, _INTERNAL_H, 0, _INTERNAL_W)
        primitive["_window"] = window
    return primitives


# -------------------------------------------------------------- intersections
def _merge_flat(best_t, best_id, best_n, candidate_t, candidate_n, index):
    better = candidate_t < best_t
    if not better.any():
        return
    best_t[better] = candidate_t[better]
    best_id[better] = index
    best_n[better] = candidate_n[better]


def _intersect_box(origin, direction, center, half, yaw):
    count = direction.shape[0]
    if yaw:
        local_origin = _rotate_y((origin - center).astype(np.float64)[None, :], -yaw)[0]
        local_direction = _rotate_y(direction, -yaw)
    else:
        local_origin = (origin - center).astype(np.float64)
        local_direction = direction

    t_near = np.full(count, -np.inf, dtype=np.float32)
    t_far = np.full(count, np.inf, dtype=np.float32)
    near_axis = np.zeros(count, dtype=np.int8)
    near_sign = np.ones(count, dtype=np.float32)
    miss = np.zeros(count, dtype=bool)

    for axis in range(3):
        low = -float(half[axis])
        high = float(half[axis])
        direction_axis = local_direction[:, axis]
        origin_axis = float(local_origin[axis])
        parallel = np.abs(direction_axis) < 1e-9
        safe_direction = np.where(parallel, 1.0, direction_axis)
        t_low_raw = (low - origin_axis) / safe_direction
        t_high_raw = (high - origin_axis) / safe_direction
        t_low = np.minimum(t_low_raw, t_high_raw)
        t_high = np.maximum(t_low_raw, t_high_raw)
        miss |= parallel & ((origin_axis < low) | (origin_axis > high))

        take = (~parallel) & (t_low > t_near)
        near_axis = np.where(take, axis, near_axis)
        near_sign = np.where(take, np.where(t_low == t_low_raw, -1.0, 1.0), near_sign)
        t_near = np.maximum(t_near, np.where(parallel, -np.inf, t_low))
        t_far = np.minimum(t_far, np.where(parallel, np.inf, t_high))

    valid = (~miss) & (t_far > 0.0) & (t_near <= t_far)
    t_best = np.where(t_near > 0.0, t_near, t_far).astype(np.float32)
    t_best = np.where(valid, t_best, np.inf).astype(np.float32)

    local_normal = np.zeros((count, 3), dtype=np.float32)
    rows = np.arange(count)
    local_normal[rows, near_axis] = near_sign
    normal = _rotate_y(local_normal, yaw) if yaw else local_normal
    facing = (normal * direction).sum(axis=1) > 0.0
    normal[facing] *= -1.0
    return t_best, normal


def _intersect_cylinder(origin, direction, center, radius, y0, y1):
    count = direction.shape[0]
    ox = float(origin[0]) - float(center[0])
    oy = float(origin[1])
    oz = float(origin[2]) - float(center[2])
    dx = direction[:, 0]
    dy = direction[:, 1]
    dz = direction[:, 2]

    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    side_ok = (a > 1e-12) & (discriminant >= 0.0)
    root_disc = np.sqrt(np.where(side_ok, discriminant, 0.0))
    denominator = np.where(side_ok, 2.0 * a, 1.0)

    best_t = np.full(count, np.inf, dtype=np.float32)
    best_n = np.zeros((count, 3), dtype=np.float32)

    for numerator in (-b - root_disc, -b + root_disc):
        t_candidate = np.where(side_ok, numerator / denominator, np.inf)
        y_hit = oy + t_candidate * dy
        good = side_ok & (t_candidate > 1e-6) & (y_hit >= y0) & (y_hit <= y1)
        t_candidate = np.where(good, t_candidate, np.inf).astype(np.float32)
        x_hit = ox + t_candidate * dx
        z_hit = oz + t_candidate * dz
        normal = np.stack(
            [
                np.where(good, x_hit / max(radius, 1e-12), 0.0),
                np.zeros(count),
                np.where(good, z_hit / max(radius, 1e-12), 0.0),
            ],
            axis=-1,
        ).astype(np.float32)
        better = t_candidate < best_t
        best_t = np.where(better, t_candidate, best_t)
        best_n = np.where(better[:, None], normal, best_n)

    for y_cap, sign in ((y1, 1.0), (y0, -1.0)):
        parallel = np.abs(dy) < 1e-9
        safe_dy = np.where(parallel, 1.0, dy)
        t_candidate = (y_cap - oy) / safe_dy
        x_hit = ox + t_candidate * dx
        z_hit = oz + t_candidate * dz
        good = (~parallel) & (t_candidate > 1e-6) & (x_hit * x_hit + z_hit * z_hit <= radius * radius)
        t_candidate = np.where(good, t_candidate, np.inf).astype(np.float32)
        normal = np.zeros((count, 3), dtype=np.float32)
        normal[:, 1] = sign
        better = t_candidate < best_t
        best_t = np.where(better, t_candidate, best_t)
        best_n = np.where(better[:, None], normal, best_n)

    return best_t, best_n


def _intersect_sphere(origin, direction, center, radius):
    relative = origin - center
    a = (direction * direction).sum(axis=1)
    b = 2.0 * (direction @ relative)
    c = float(relative @ relative) - radius * radius
    discriminant = b * b - 4.0 * a * c
    ok = discriminant >= 0.0
    root = np.sqrt(np.where(ok, discriminant, 0.0))
    t_candidate = (-b - root) / np.where(a > 1e-12, 2.0 * a, 1.0)
    t_best = np.where(ok & (t_candidate > 1e-6), t_candidate, np.inf).astype(np.float32)
    points = origin + t_best[:, None] * direction
    normal = (points - center) / max(radius, 1e-12)
    normal[~np.isfinite(t_best)] = 0.0
    return t_best, normal.astype(np.float32)


def _intersect_plane(origin, direction, point, normal):
    denominator = direction @ normal
    ok = np.abs(denominator) > 1e-9
    safe = np.where(ok, denominator, 1.0)
    t_candidate = float((point - origin) @ normal) / safe
    t_best = np.where(ok & (t_candidate > 1e-6), t_candidate, np.inf).astype(np.float32)
    normal_grid = np.broadcast_to(normal, (direction.shape[0], 3)).copy()
    normal_grid[~np.isfinite(t_best)] = 0.0
    return t_best, normal_grid


def intersect(primitive: dict, origin: np.ndarray, direction: np.ndarray):
    kind = primitive["kind"]
    if kind == "box":
        return _intersect_box(
            origin, direction, primitive["center"], primitive["half"], primitive["yaw"]
        )
    if kind == "cylinder":
        return _intersect_cylinder(
            origin,
            direction,
            primitive["center"],
            primitive["radius"],
            primitive["y0"],
            primitive["y1"],
        )
    if kind == "sphere":
        return _intersect_sphere(origin, direction, primitive["center"], primitive["radius"])
    if kind == "plane":
        return _intersect_plane(origin, direction, primitive["point"], primitive["normal"])
    raise ValueError(f"unknown primitive kind {kind!r}")


# ------------------------------------------------------------------- shading
def _shade_objects(normals, points, origin, base_rgb):
    lambert = np.clip(-(normals * LIGHT_DIR[None, :]).sum(axis=1), 0.0, 1.0)
    brightness = AMBIENT + DIFFUSE * lambert
    view = origin[None, :] - points
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
    halfway = view - LIGHT_DIR[None, :]
    halfway /= np.linalg.norm(halfway, axis=1, keepdims=True) + 1e-9
    specular = np.clip((normals * halfway).sum(axis=1), 0.0, 1.0) ** SPECULAR_POWER
    return base_rgb * brightness[:, None] + SPECULAR * 255.0 * specular[:, None]


def _shade_wall(points):
    y = points[:, 1]
    mix = np.clip((y + 0.10) / 0.65, 0.0, 1.0)[:, None]
    base = WALL_BOTTOM_F32[None, :] * (1.0 - mix) + WALL_TOP_F32[None, :] * mix
    horizontal = 1.0 - 0.06 * np.clip(points[:, 0] / 1.6, -1.0, 1.0) ** 2
    return base * horizontal[:, None]


def _shade_floor(points):
    inside_x = np.clip((points[:, 0] - TABLE_X0) / 0.08, 0.0, 1.0) * np.clip(
        (TABLE_X1 - points[:, 0]) / 0.08, 0.0, 1.0
    )
    inside_z = np.clip((points[:, 2] - TABLE_Z0) / 0.08, 0.0, 1.0) * np.clip(
        (TABLE_Z1 - points[:, 2]) / 0.08, 0.0, 1.0
    )
    under_table = inside_x * inside_z
    factor = 0.95 - 0.48 * under_table
    return FLOOR_RGB_F32[None, :] * factor[:, None]


def _shadow_infos(objects: list[dict]) -> list[dict]:
    """Per-object shadow/AO world parameters + cached screen window."""
    infos: list[dict] = []
    light_ratio_x = float(LIGHT_DIR[0] / -LIGHT_DIR[1])
    light_ratio_z = float(LIGHT_DIR[2] / -LIGHT_DIR[1])
    for obj in objects:
        base_x, base_z = base_center_of(obj)
        radius = footprint_radius(obj)
        center_height = float(obj["center"][1])
        shadow_x = base_x + light_ratio_x * center_height
        shadow_z = base_z + light_ratio_z * center_height
        spread = radius * 1.75 + 0.012
        offset = math.hypot(shadow_x - base_x, shadow_z - base_z)
        window_radius = radius * 1.2 + offset + spread * 1.5
        middle_x = (shadow_x + base_x) / 2.0
        middle_z = (shadow_z + base_z) / 2.0
        angles = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
        probes = np.stack(
            [
                middle_x + window_radius * np.cos(angles),
                np.full(20, 0.01),
                middle_z + window_radius * np.sin(angles),
            ],
            axis=1,
        )
        infos.append(
            {
                "base_x": base_x,
                "base_z": base_z,
                "shadow_x": shadow_x,
                "shadow_z": shadow_z,
                "spread": spread,
                "ao_radius": radius * 1.05 + 0.008,
                "window": _screen_window(probes),
            }
        )
    return infos


def _table_colours(points_x, points_z, shadow_infos, row_start):
    rows, cols = points_x.shape
    factor = np.ones((rows, cols), dtype=np.float32)
    for info in shadow_infos:
        window = info["window"]
        if window is None:
            continue
        row0 = max(window[0], row_start)
        row1 = min(window[1], row_start + rows)
        if row0 >= row1:
            continue
        col0, col1 = window[2], window[3]
        local_row0 = row0 - row_start
        local_row1 = row1 - row_start
        px = points_x[local_row0:local_row1, col0:col1]
        pz = points_z[local_row0:local_row1, col0:col1]
        spread = info["spread"]
        dx = (px - info["shadow_x"]) / spread
        dz = (pz - info["shadow_z"]) / spread
        falloff = np.clip(1.0 - (dx * dx + dz * dz), 0.0, 1.0)
        ao_radius = info["ao_radius"]
        aox = (px - info["base_x"]) / ao_radius
        aoz = (pz - info["base_z"]) / ao_radius
        ao_falloff = np.clip(1.0 - (aox * aox + aoz * aoz), 0.0, 1.0)
        factor[local_row0:local_row1, col0:col1] *= (1.0 - 0.30 * falloff ** 1.3) * (
            1.0 - 0.24 * ao_falloff
        )
    distance_falloff = 0.94 + 0.06 * np.exp(
        -((points_x / 0.9) ** 2 + ((points_z - 0.9) / 1.1) ** 2)
    )
    return TABLE_RGB_F32[None, None, :] * (factor * distance_falloff)[..., None]


# ------------------------------------------------------------------ rendering
def _render_buffer(objects: list[dict], primitives: list[dict]) -> np.ndarray:
    internal_width = _INTERNAL_W
    internal_height = _INTERNAL_H
    origin = CAMERA.position.astype(np.float32)
    right = CAMERA.right.astype(np.float32)
    up = CAMERA.up.astype(np.float32)
    forward = CAMERA.forward.astype(np.float32)
    shadow_infos = _shadow_infos(objects)

    columns = (np.arange(internal_width, dtype=np.float32) + 0.5) / SUPERSAMPLE
    du = (columns - CX) / FX
    buffers = np.empty((internal_height, internal_width, 3), dtype=np.uint8)

    for row_start in range(0, internal_height, _STRIP_ROWS):
        row_end = min(row_start + _STRIP_ROWS, internal_height)
        rows = (np.arange(row_start, row_end, dtype=np.float32) + 0.5) / SUPERSAMPLE
        dv = -(rows - CY) / FY
        direction = (
            right[None, None, :] * du[None, :, None]
            + up[None, None, :] * dv[:, None, None]
            + forward[None, None, :]
        )
        strip_rows = row_end - row_start
        best_t = np.full((strip_rows, internal_width), np.inf, dtype=np.float32)
        best_id = np.full((strip_rows, internal_width), -1, dtype=np.int16)
        best_normal = np.zeros((strip_rows, internal_width, 3), dtype=np.float32)

        for index, primitive in enumerate(primitives):
            window = primitive["_window"]
            row0 = max(window[0], row_start)
            row1 = min(window[1], row_end)
            if row0 >= row1:
                continue
            col0 = max(0, window[2])
            col1 = min(internal_width, window[3])
            if col0 >= col1:
                continue
            local_row0 = row0 - row_start
            local_row1 = row1 - row_start
            full_width = col0 == 0 and col1 == internal_width
            if full_width:
                sub_direction = direction[local_row0:local_row1].reshape(-1, 3)
            else:
                sub_direction = direction[local_row0:local_row1, col0:col1].reshape(-1, 3)
            candidate_t, candidate_normal = intersect(primitive, origin, sub_direction)
            if full_width:
                sub_shape = (local_row1 - local_row0, internal_width)
                view_t = best_t[local_row0:local_row1]
                view_id = best_id[local_row0:local_row1]
                view_n = best_normal[local_row0:local_row1]
            else:
                sub_shape = (local_row1 - local_row0, col1 - col0)
                view_t = best_t[local_row0:local_row1, col0:col1]
                view_id = best_id[local_row0:local_row1, col0:col1]
                view_n = best_normal[local_row0:local_row1, col0:col1]
            better = candidate_t.reshape(sub_shape) < view_t
            if not better.any():
                continue
            view_t[better] = candidate_t.reshape(sub_shape)[better]
            view_id[better] = index
            view_n[better] = candidate_normal.reshape(sub_shape + (3,))[better]

        hit = best_id >= 0
        safe_t = np.where(hit, best_t, 0.0)
        points = origin[None, None, :] + safe_t[..., None] * direction
        colours = np.zeros((strip_rows, internal_width, 3), dtype=np.float32)
        table_rgb = None
        for index, primitive in enumerate(primitives):
            material = primitive["material"]
            mask = best_id == index
            if not mask.any():
                continue
            if material == "object":
                colours[mask] = _shade_objects(
                    best_normal[mask], points[mask], origin, primitive["rgb"]
                )
            elif material == "wall":
                colours[mask] = _shade_wall(points[mask])
            elif material == "floor":
                colours[mask] = _shade_floor(points[mask])
            elif material == "leg":
                colours[mask] = LEG_RGB_F32[None, :] * 0.72
            else:  # table
                if table_rgb is None:
                    table_rgb = _table_colours(
                        points[..., 0], points[..., 2], shadow_infos, row_start
                    )
                colours[mask] = table_rgb[mask]

        u_norm = (columns[None, :] - CX) / (WIDTH * 0.62)
        v_norm = (rows[:, None] - CY) / (HEIGHT * 0.62)
        vignette = np.clip(1.0 - 0.16 * (u_norm * u_norm + v_norm * v_norm), 0.82, 1.0)
        colours *= vignette[..., None]
        np.clip(colours, 0.0, 255.0, out=colours)
        buffers[row_start:row_end] = colours.astype(np.uint8)
    return buffers


def render_frame(objects: list[dict]) -> Image.Image:
    """Render one 1024x768 RGB frame for the given objects."""
    primitives = build_primitives(objects)
    buffer = _render_buffer(objects, primitives)
    image = Image.fromarray(buffer, mode="RGB")
    if SUPERSAMPLE != 1:
        image = image.resize((WIDTH, HEIGHT), resample=Image.BOX)
    return image


def save_frame(objects: list[dict], path) -> int:
    """Render + save one JPEG frame; returns the file size in bytes."""
    image = render_frame(objects)
    image.save(
        path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False,
        progressive=False,
    )
    return path.stat().st_size


__all__ = [
    "SUPERSAMPLE",
    "JPEG_QUALITY",
    "JPEG_SUBSAMPLING",
    "TABLE_TOP_Y",
    "TABLE_X0",
    "TABLE_X1",
    "TABLE_Z0",
    "TABLE_Z1",
    "WALL_Z",
    "FLOOR_Y",
    "build_primitives",
    "intersect",
    "render_frame",
    "save_frame",
]
