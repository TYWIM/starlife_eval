"""特征缓存 npz 加载与校验。

契约见 ``ARCHITECTURE.md`` §3，键名/语义与
``starlife/starlife_scripts/extract_features.py`` (``run_spatial``, 219-243 行) 一致：

``np.load(path, allow_pickle=False)`` + 同名 ``*.meta.json``。这里**不硬编码 5120**：
主干特征维度一律从 npz 形状 / meta["dim"] 推导；键名也不许改名或补别名。

只读 + 严格校验；缺键、形状不一致、meta 与 npz 维度冲突都直接报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..types import CHANGE_LABELS, FeatureBundle

# 头推理 / 事件 / 状态回放真正依赖的键（extract_features.run_spatial 全部会写）。
REQUIRED_KEYS: tuple[str, ...] = (
    "slot_before",
    "slot_after",
    "slot_mask",
    "slot_present",
    "slot_change",
    "slot_type",
    "slot_pair",
    "pair_split",
    "rel_from",
    "rel_to",
    "rel_type",
    "rel_pos",
    "rel_contact",
    "rel_support",
    "rel_dist",
)

# 帧级特征当前不参与头推理；允许缺失或 (N, 0)（--skip-frames）。
OPTIONAL_KEYS: tuple[str, ...] = ("img_before", "img_after")

SLOT_ROW_KEYS: tuple[str, ...] = (
    "slot_mask",
    "slot_present",
    "slot_change",
    "slot_type",
    "slot_pair",
)

REL_ROW_KEYS: tuple[str, ...] = (
    "rel_from",
    "rel_to",
    "rel_type",
    "rel_pos",
    "rel_contact",
    "rel_support",
    "rel_dist",
)

META_KEYS: tuple[str, ...] = ("type_keys", "rel_types", "rel_pos_keys")


class FeatureError(ValueError):
    """特征缓存结构不合法。"""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise FeatureError(msg)


def _meta_path(npz_path: Path, meta_path: str | Path | None) -> Path:
    if meta_path is not None:
        return Path(meta_path)
    # 与 train_heads.load_spatial / extract_features.run_spatial 相同：
    # spatial_v2.npz -> spatial_v2.meta.json
    return npz_path.with_suffix(".meta.json")


def load_features(npz_path: str | Path, meta_path: str | Path | None = None) -> FeatureBundle:
    """读取特征 npz + meta → :class:`FeatureBundle`（数组为只读视图）。"""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"缺少特征缓存：{path}")

    mpath = _meta_path(path, meta_path)
    if not mpath.exists():
        raise FileNotFoundError(f"缺少特征 meta：{mpath}")
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    _require(isinstance(meta, dict), f"{mpath} 期望 JSON 对象")

    with np.load(path, allow_pickle=False) as npz:
        arrays: dict[str, np.ndarray] = {k: npz[k] for k in npz.files}

    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise FeatureError(
            f"{path} 缺少必需键 {missing}；实际键={sorted(arrays)}"
            "（键名必须与 extract_features.run_spatial 一致，不允许改键/补别名）"
        )
    missing_meta = [k for k in META_KEYS if k not in meta]
    if missing_meta:
        raise FeatureError(f"{mpath} 缺少键 {missing_meta}; 实际键={sorted(meta)}")

    _validate(npz_path=path, meta_path=mpath, arrays=arrays, meta=meta)

    bundle = FeatureBundle(path=path, arrays=arrays, meta=meta)
    return bundle


def _validate(*, npz_path: Path, meta_path: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
    sb, sa = arrays["slot_before"], arrays["slot_after"]
    _require(sb.ndim == 2 and sa.ndim == 2, f"{npz_path}: slot_before/slot_after 必须是二维 (N_slot, D)")
    _require(sb.shape == sa.shape, f"{npz_path}: slot_before{sb.shape} 与 slot_after{sa.shape} 形状不一致")
    n_slot, dim = int(sb.shape[0]), int(sb.shape[1])
    _require(dim > 0, f"{npz_path}: 特征维度为 0（主干特征未抽取？）")

    if "dim" in meta:
        meta_dim = int(meta["dim"])
        # meta.dim==0 表示抽取时用了 --skip-frames（未存整帧特征，img_before/img_after 为
        # (N_pair, 0)），此时主干特征维度只能从 slot_before/slot_after 的形状推导。
        if meta_dim > 0:
            _require(
                meta_dim == dim,
                f"{meta_path}: meta.dim={meta_dim} 与 npz 实际维度 {dim} 不一致",
            )

    for key in SLOT_ROW_KEYS:
        arr = arrays[key]
        _require(arr.ndim == 1, f"{npz_path}: {key} 必须是一维，实际 shape={arr.shape}")
        _require(int(arr.shape[0]) == n_slot, f"{npz_path}: {key} 长度 {arr.shape[0]} != N_slot {n_slot}")

    pair_split = arrays["pair_split"]
    _require(pair_split.ndim == 1, f"{npz_path}: pair_split 必须是一维")
    n_pair = int(pair_split.shape[0])
    _require(n_pair > 0, f"{npz_path}: pair_split 为空")

    slot_pair = arrays["slot_pair"]
    if n_slot:
        _require(int(slot_pair.min()) >= 0, f"{npz_path}: slot_pair 含负值")
        _require(int(slot_pair.max()) < n_pair, f"{npz_path}: slot_pair 最大值 {int(slot_pair.max())} >= pair 数 {n_pair}")

    n_types = len(meta["type_keys"])
    n_rel_types = len(meta["rel_types"])
    n_rel_pos = len(meta["rel_pos_keys"])
    _require(n_types > 0 and n_rel_types > 0 and n_rel_pos > 0, f"{meta_path}: type_keys/rel_types/rel_pos_keys 不能为空")

    for key in REL_ROW_KEYS:
        arr = arrays[key]
        _require(arr.ndim == 1, f"{npz_path}: {key} 必须是一维，实际 shape={arr.shape}")
    n_rel = int(arrays["rel_from"].shape[0])
    for key in REL_ROW_KEYS:
        _require(int(arrays[key].shape[0]) == n_rel, f"{npz_path}: {key} 长度 {arrays[key].shape[0]} != N_rel {n_rel}")

    if n_rel:
        frm, to = arrays["rel_from"], arrays["rel_to"]
        _require(int(frm.min()) >= 0 and int(to.min()) >= 0, f"{npz_path}: rel_from/rel_to 含负值")
        _require(
            int(max(frm.max(), to.max())) < n_slot,
            f"{npz_path}: rel_from/rel_to 全局下标越界（N_slot={n_slot}）",
        )

    _check_index_range(npz_path, "slot_change", arrays["slot_change"], len(CHANGE_LABELS))
    _check_index_range(npz_path, "slot_type", arrays["slot_type"], n_types)
    _check_index_range(npz_path, "rel_type", arrays["rel_type"], n_rel_types)
    _check_index_range(npz_path, "rel_pos", arrays["rel_pos"], n_rel_pos)

    for key in OPTIONAL_KEYS:
        if key in arrays:
            arr = arrays[key]
            _require(arr.ndim == 2, f"{npz_path}: {key} 必须是二维 (N_pair, D_frame)")
            _require(int(arr.shape[0]) == n_pair, f"{npz_path}: {key} 行数 {arr.shape[0]} != N_pair {n_pair}")


def _check_index_range(npz_path: Path, key: str, arr: np.ndarray, n_classes: int) -> None:
    if arr.size == 0:
        return
    lo, hi = int(arr.min()), int(arr.max())
    _require(lo >= 0 and hi < n_classes, f"{npz_path}: {key} 取值范围 [{lo},{hi}] 超出类别数 {n_classes}")


def feature_dim(bundle: FeatureBundle) -> int:
    """主干特征维度 D（从数组形状推导，不硬编码）。"""
    return int(bundle["slot_after"].shape[1])


def slot_feature_dim(bundle: FeatureBundle) -> int:
    """change 头输入维度 3D+1（train_heads.main:228-229）。"""
    return 3 * feature_dim(bundle) + 1


def type_feature_dim(bundle: FeatureBundle) -> int:
    """type 头输入维度 2D（train_heads.main:252）。"""
    return 2 * feature_dim(bundle)


def pair_base_dim(bundle: FeatureBundle) -> int:
    """relation/attrs 基础输入维度 3D+1（train_heads.main:261-263）。"""
    return 3 * feature_dim(bundle) + 1


def relation_onehot_dim(bundle: FeatureBundle) -> int:
    """rel_pos 条件 one-hot 维度 len(rel_types)+1（train_heads.main:268）。"""
    return len(bundle.rel_types) + 1
