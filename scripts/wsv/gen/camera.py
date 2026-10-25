"""Fixed pinhole camera used by every generated frame.

The camera calibration is part of the generator contract (and is copied into
``meta.json`` / each task's ``meta``)::

    width = 1024, height = 768
    fx = fy = 800, cx = 512, cy = 384

Both frames of a task use the exact same camera instance, which is what makes
the T7 image-plane direction question well defined (``A above B`` means "the
image-plane centre of A is above the image-plane centre of B in the shared
view").
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

WIDTH = 1024
HEIGHT = 768
FX = 800.0
FY = 800.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0

#: camera position in world metres (table top is the plane ``y = 0``)
CAMERA_POSITION: tuple[float, float, float] = (0.0, 0.45, -0.10)
#: look-at target: slightly below the table plane, ahead of the camera
CAMERA_TARGET: tuple[float, float, float] = (0.0, 0.02, 0.75)


def _normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"cannot normalize zero vector: {vector!r}")
    return array / norm


class Camera:
    """A right-handed pinhole camera looking along ``forward``."""

    __slots__ = ("position", "right", "up", "forward")

    def __init__(
        self,
        position: Sequence[float],
        right: Sequence[float],
        up: Sequence[float],
        forward: Sequence[float],
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.right = np.asarray(right, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.forward = np.asarray(forward, dtype=np.float64)

    # ---------------------------------------------------------------- builders
    @classmethod
    def look_at(
        cls,
        position: Sequence[float],
        target: Sequence[float],
        up: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> "Camera":
        position_array = np.asarray(position, dtype=np.float64)
        forward = _normalize(np.asarray(target, dtype=np.float64) - position_array)
        right = _normalize(np.cross(np.asarray(up, dtype=np.float64), forward))
        up_vector = np.cross(forward, right)
        return cls(position_array, right, up_vector, forward)

    # ------------------------------------------------------------- projection
    def to_camera(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        """World points -> camera coordinates ``(x right, y up, z forward)``."""
        array = np.asarray(list(points) if not isinstance(points, np.ndarray) else points, dtype=np.float64)
        relative = array - self.position
        return np.stack(
            [
                relative @ self.right,
                relative @ self.up,
                relative @ self.forward,
            ],
            axis=-1,
        )

    def project(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        """World points -> ``(u, v, depth)`` with ``v`` growing downwards."""
        camera = self.to_camera(points)
        depth = camera[..., 2]
        safe_depth = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
        u = CX + FX * camera[..., 0] / safe_depth
        v = CY - FY * camera[..., 1] / safe_depth
        return np.stack([u, v, depth], axis=-1)

    def project_one(self, point: Sequence[float]) -> tuple[float, float, float]:
        result = self.project([point])[0]
        return float(result[0]), float(result[1]), float(result[2])

    def depth(self, point: Sequence[float]) -> float:
        """Depth along the optical axis (metres)."""
        return float(np.dot(np.asarray(point, dtype=np.float64) - self.position, self.forward))

    # ------------------------------------------------------------------- rays
    def ray(self, u: float, v: float) -> np.ndarray:
        """World-space (unnormalised) ray direction through image point ``(u, v)``."""
        direction = np.array(
            [(u - CX) / FX, -(v - CY) / FY, 1.0], dtype=np.float64
        )
        return (
            self.right * direction[0]
            + self.up * direction[1]
            + self.forward * direction[2]
        )

    def point_at_height(self, u: float, v: float, height: float) -> np.ndarray | None:
        """World point whose projection is ``(u, v)`` and whose ``y == height``."""
        direction = self.ray(u, v)
        if abs(float(direction[1])) < 1e-9:
            return None
        distance = (height - float(self.position[1])) / float(direction[1])
        if distance <= 0.0:
            return None
        return self.position + distance * direction

    # ---------------------------------------------------------------- helpers
    def image_bbox(self, points: Iterable[Sequence[float]]) -> tuple[float, float, float, float] | None:
        """Projected axis-aligned bounding box of a point cloud (``u0, v0, u1, v1``)."""
        projected = self.project(points)
        if np.any(projected[:, 2] <= 0.05):
            return None
        u = projected[:, 0]
        v = projected[:, 1]
        return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


#: the single shared camera instance used by all generated frames
CAMERA = Camera.look_at(CAMERA_POSITION, CAMERA_TARGET)


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Centre of an ``(u0, v0, u1, v1)`` image bbox."""
    u0, v0, u1, v1 = (float(value) for value in bbox)
    return ((u0 + u1) / 2.0, (v0 + v1) / 2.0)


def bbox_size(bbox: Sequence[float]) -> tuple[float, float]:
    """Width/height of an ``(u0, v0, u1, v1)`` image bbox."""
    u0, v0, u1, v1 = (float(value) for value in bbox)
    return (u1 - u0, v1 - v0)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union of two image bboxes."""
    ax0, ay0, ax1, ay1 = (float(value) for value in first)
    bx0, by0, bx1, by1 = (float(value) for value in second)
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


__all__ = [
    "WIDTH",
    "HEIGHT",
    "FX",
    "FY",
    "CX",
    "CY",
    "CAMERA_POSITION",
    "CAMERA_TARGET",
    "CAMERA",
    "Camera",
    "bbox_center",
    "bbox_size",
    "bbox_iou",
]
