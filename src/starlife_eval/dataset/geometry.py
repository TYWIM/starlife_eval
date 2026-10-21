"""bbox 几何特征（9 维）。

逐行移植 ``starlife/starlife_scripts/train_heads.py:143-166 pair_geometry`` 与
``train_heads.py:130-140 load_manifest`` 的 bbox 定位方式。评测要求 v2g 数字
（relation top1 0.7745）可复现，因此公式、变量顺序、clamp 常量、dtype 全部保持一致：

    [dxn, dyn, wa/wb, ha/hb, area_a/area_b, iou, dist_norm, sax/1024, say/768]

注意：**不做 z-score**。train_heads.main 里的 z-score 只用于可选 ``--slot-geom``
（train_heads.py:230-239，change 头的 slot 级几何），``pair_geometry`` 的结果是
原样拼进 pair_feat 的（train_heads.py:264-267 / eval_compare_cloud.py:154-158）。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..types import DatasetManifest

IMAGE_W = 1024.0
IMAGE_H = 768.0
GEOMETRY_DIM = 9


def pair_geometry(rel: Mapping[str, object], bbox: Mapping[int, np.ndarray], img_w: float = IMAGE_W,
                  img_h: float = IMAGE_H) -> np.ndarray:
    """9 维 bbox 几何特征；``rel["frm"]``/``rel["to"]`` 必须是全局 slot 下标。

    ported from train_heads.py:143-166 pair_geometry
    """
    n = len(rel["frm"])  # type: ignore[arg-type]
    if n == 0:
        return np.zeros((0, GEOMETRY_DIM), dtype=np.float32)
    fa = np.stack([bbox[int(i)] for i in rel["frm"]])  # type: ignore[arg-type]
    fb = np.stack([bbox[int(i)] for i in rel["to"]])  # type: ignore[arg-type]
    sax, say = (fa[:, 0] + fa[:, 2]) * 0.5, (fa[:, 1] + fa[:, 3]) * 0.5
    hbx, hby = (fb[:, 0] + fb[:, 2]) * 0.5, (fb[:, 1] + fb[:, 3]) * 0.5
    haw = np.maximum((fa[:, 2] - fa[:, 0]) * 0.5, 1.0)
    hah = np.maximum((fa[:, 3] - fa[:, 1]) * 0.5, 1.0)
    hbw = np.maximum((fb[:, 2] - fb[:, 0]) * 0.5, 1.0)
    hbh = np.maximum((fb[:, 3] - fb[:, 1]) * 0.5, 1.0)
    dxn, dyn = (sax - hbx) / hbw, (say - hby) / hbh
    wa, ha = fa[:, 2] - fa[:, 0], fa[:, 3] - fa[:, 1]
    wb, hb = fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1]
    inter_w = np.clip(np.minimum(fa[:, 2], fb[:, 2]) - np.maximum(fa[:, 0], fb[:, 0]), 0, None)
    inter_h = np.clip(np.minimum(fa[:, 3], fb[:, 3]) - np.maximum(fa[:, 1], fb[:, 1]), 0, None)
    inter = inter_w * inter_h
    union = np.maximum(wa * ha + wb * hb - inter, 1.0)
    iou = inter / union
    dist = np.sqrt((sax - hbx) ** 2 + (say - hby) ** 2) / np.sqrt(img_w ** 2 + img_h ** 2)
    return np.stack([dxn, dyn, wa / wb, ha / hb, (wa * ha) / np.maximum(wb * hb, 1.0),
                     iou, dist, sax / img_w, say / img_h], axis=1).astype(np.float32)


def bbox_map_from_manifest(manifest: DatasetManifest) -> dict[int, np.ndarray]:
    """全局 slot 下标 → bbox（np.float32[4]）。

    与 train_heads.py:130-140 load_manifest 的 offsets + bbox 字典等价：
    ``offsets[i+1] = offsets[i] + len(rows[i]["entities"])``。
    """
    bbox: dict[int, np.ndarray] = {}
    for pair in manifest.pairs:
        base = manifest.slot_offsets[pair.index]
        for entity in pair.entities:
            bbox[base + entity.slot] = np.asarray(entity.bbox, dtype=np.float32)
    return bbox


def pair_geometry_from_manifest(rel: Mapping[str, object], manifest: DatasetManifest,
                                img_w: float = IMAGE_W, img_h: float = IMAGE_H) -> np.ndarray:
    """:func:`pair_geometry` + :func:`bbox_map_from_manifest` 的组合。"""
    return pair_geometry(rel, bbox_map_from_manifest(manifest), img_w=img_w, img_h=img_h)


def bboxes_for_global_slots(bbox: Mapping[int, np.ndarray], slots: Sequence[int]) -> np.ndarray:
    """按给定全局下标取 bbox 矩阵 (n, 4)，缺 slot 直接报错（不补零框）。"""
    missing = [int(s) for s in slots if int(s) not in bbox]
    if missing:
        raise KeyError(f"bbox_map 缺少全局 slot {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return np.stack([bbox[int(s)] for s in slots])
