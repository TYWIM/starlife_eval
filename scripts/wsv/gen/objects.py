"""Object catalogue (colours / primitive types) and analytic projection helpers.

Every object is one of three simple shaded primitives that sit on the table
top: a near-cubic box (``cube``), a vertical ``cylinder`` or a ``sphere``.  The
human-readable description used in the questions is ``"<colour> <type>"``
(``"red cube"``, ``"blue cylinder"``, ...).  Within one task every object gets a
distinct colour, so descriptions are unique by construction.

Objects are stored as plain dicts (JSON/pickle friendly).  Geometry helpers
return numpy arrays.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .camera import CAMERA

#: colour name -> display RGB
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (198, 48, 46),
    "blue": (46, 88, 196),
    "green": (54, 150, 74),
    "yellow": (231, 190, 54),
    "orange": (232, 122, 38),
    "purple": (133, 71, 190),
    "cyan": (58, 172, 186),
    "magenta": (196, 64, 152),
}

#: stable colour order used by the deterministic planner
COLOR_NAMES: tuple[str, ...] = tuple(COLORS)

#: primitive kinds
TYPES: tuple[str, ...] = ("cube", "cylinder", "sphere")


# --------------------------------------------------------------------- makers
def make_cube(
    object_id: str,
    color: str,
    center: Sequence[float],
    half: Sequence[float],
    yaw: float = 0.0,
) -> dict:
    return {
        "object_id": str(object_id),
        "kind": "cube",
        "color": color,
        "rgb": list(COLORS[color]),
        "center": [float(value) for value in center],
        "half": [float(value) for value in half],
        "yaw": float(yaw),
    }


def make_cylinder(
    object_id: str,
    color: str,
    center: Sequence[float],
    radius: float,
    height: float,
) -> dict:
    return {
        "object_id": str(object_id),
        "kind": "cylinder",
        "color": color,
        "rgb": list(COLORS[color]),
        "center": [float(value) for value in center],
        "radius": float(radius),
        "height": float(height),
    }


def make_sphere(
    object_id: str,
    color: str,
    center: Sequence[float],
    radius: float,
) -> dict:
    return {
        "object_id": str(object_id),
        "kind": "sphere",
        "color": color,
        "rgb": list(COLORS[color]),
        "center": [float(value) for value in center],
        "radius": float(radius),
    }


def describe(obj: dict) -> str:
    """Human description used in questions (unique within a task)."""
    return f"{obj['color']} {obj['kind']}"


# ------------------------------------------------------------------ geometry
def _rotate_y(points: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate points around the world Y axis by ``yaw`` radians."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x = points[..., 0]
    z = points[..., 2]
    return np.stack(
        [cosine * x + sine * z, points[..., 1], -sine * x + cosine * z],
        axis=-1,
    )


def silhouette_points(obj: dict) -> np.ndarray:
    """Surface points whose projected bbox equals the object's projected bbox.

    * cube: the 8 (possibly yaw-rotated) corners;
    * cylinder: both rim circles sampled densely;
    * sphere: the exact perspective silhouette circle (view-tangent points).
    """
    kind = obj["kind"]
    center = np.asarray(obj["center"], dtype=np.float64)

    if kind == "cube":
        half = np.asarray(obj["half"], dtype=np.float64)
        signs = np.array(
            [
                (sx, sy, sz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        corners = signs * half
        return _rotate_y(corners, float(obj.get("yaw", 0.0))) + center

    if kind == "cylinder":
        radius = float(obj["radius"])
        height = float(obj["height"])
        angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        ring = np.stack(
            [np.cos(angles) * radius, np.zeros_like(angles), np.sin(angles) * radius],
            axis=-1,
        )
        bottom = ring + np.array([center[0], center[1] - height / 2.0, center[2]])
        top = ring + np.array([center[0], center[1] + height / 2.0, center[2]])
        return np.concatenate([bottom, top], axis=0)

    if kind == "sphere":
        radius = float(obj["radius"])
        relative = center - CAMERA.position
        distance = float(np.linalg.norm(relative))
        if distance <= radius * 1.001:
            raise ValueError("sphere intersects the camera; cannot project")
        view = relative / distance
        helper = np.array([0.0, 1.0, 0.0]) if abs(view[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        first = np.cross(view, helper)
        first /= np.linalg.norm(first)
        second = np.cross(view, first)
        ring_center = CAMERA.position + relative * (1.0 - radius * radius / (distance * distance))
        ring_radius = radius * math.sqrt(max(0.0, 1.0 - radius * radius / (distance * distance)))
        angles = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
        return (
            ring_center
            + ring_radius * np.cos(angles)[:, None] * first[None, :]
            + ring_radius * np.sin(angles)[:, None] * second[None, :]
        )

    raise ValueError(f"unknown object kind {kind!r}")


def image_bbox(obj: dict) -> tuple[float, float, float, float] | None:
    """Projected image bbox ``(u0, v0, u1, v1)`` (``None`` when behind the camera)."""
    return CAMERA.image_bbox(silhouette_points(obj))


def bbox_center_of(obj: dict) -> tuple[float, float] | None:
    bbox = image_bbox(obj)
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_size_of(obj: dict) -> tuple[float, float] | None:
    bbox = image_bbox(obj)
    if bbox is None:
        return None
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def footprint_xz(obj: dict) -> tuple[float, float, float, float]:
    """World ``(xmin, xmax, zmin, zmax)`` of the object footprint."""
    points = silhouette_points(obj)
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    )


def height_of(obj: dict) -> float:
    kind = obj["kind"]
    if kind == "cube":
        return float(obj["half"][1]) * 2.0
    if kind == "cylinder":
        return float(obj["height"])
    if kind == "sphere":
        return float(obj["radius"]) * 2.0
    raise ValueError(f"unknown object kind {kind!r}")


def base_center_of(obj: dict) -> tuple[float, float]:
    """``(x, z)`` of the footprint centre on the table plane."""
    return (float(obj["center"][0]), float(obj["center"][2]))


def footprint_radius(obj: dict) -> float:
    """Characteristic footprint radius in world metres (used for shadows)."""
    xmin, xmax, zmin, zmax = footprint_xz(obj)
    return max(xmax - xmin, zmax - zmin) / 2.0


def depth_of(obj: dict) -> float:
    return CAMERA.depth(obj["center"])


__all__ = [
    "COLORS",
    "COLOR_NAMES",
    "TYPES",
    "make_cube",
    "make_cylinder",
    "make_sphere",
    "describe",
    "silhouette_points",
    "image_bbox",
    "bbox_center_of",
    "bbox_size_of",
    "footprint_xz",
    "height_of",
    "base_center_of",
    "footprint_radius",
    "depth_of",
]
