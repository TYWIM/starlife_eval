"""头级指标（L2）：镜像 ``train_heads.py::main`` 在 test split 上的评测口径。

口径来源（**只读**移植，不 import 原脚本；原脚本在 ``starlife/`` 下）：
``starlife/starlife_scripts/train_heads.py`` 第 240-335 行 + ``train_classifier`` 的
``macro_f1`` 用法：

* ``slot_split = z["pair_split"][z["slot_pair"]]``，test 掩码即 ``slot_split == split``；
* ``change``: 上述掩码 + ``z["slot_change"]``，4 类；
* ``type``: 同掩码 + ``z["slot_type"]``，``n_classes = len(meta["type_keys"])``；
* ``relation``: ``relation`` 采样结果（``train_heads.build_relation_data`` 的 dict，
  由 dataset 层传入，本模块**不重采样**）里 ``split == split`` 的样本，
  ``n_classes = rel["n_types"]``（= 11 关系 + none）；**含负样本**；
* ``rel_pos``: ``split == split & ~is_none`` 的真关系，``n_classes = len(meta["rel_pos_keys"])``；
* ``attrs``: ``split == split`` 的样本，contact / support 各做二分类 macro-F1，
  ``attrs = (contact_f1 + support_f1) / 2``（与 ``train_attr_head`` 第 332-335 行一致）。

全部指标使用 :func:`starlife_eval.verifiers.metrics.macro_f1_train` 口径
（跳过 ``tp+fp+fn==0`` 的类），**类别数从 meta 锁定**，不因 test 缺类而变化；
浮点不 round（JSON 序列化时由 ``metrics.json_safe`` 统一处理）。

本模块不依赖 torch / dataset / models：``z`` 可以是 ``np.load`` 的 NpzFile、
``FeatureBundle`` 或普通 dict；``head`` 是 ``types.HeadPrediction``。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..types import CHANGE_LABELS, HeadPrediction
from .metrics import macro_f1_train, top1

__all__ = ["evaluate_head_metrics"]


def _rel_get(relation: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        return np.asarray(relation[key])
    except KeyError as exc:  # pragma: no cover - 明确报错信息
        raise KeyError(
            f"relation 采样结果缺少字段 {key!r}（应来自 "
            "dataset.negative_sampling.build_relation_data 的 dict）"
        ) from exc


def _as_int64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.int64).reshape(-1)


def _head_array(head: HeadPrediction, name: str, expected: int | None = None) -> np.ndarray:
    arr = getattr(head, name, None)
    if arr is None:
        raise ValueError(f"head 缺少字段 {name!r}")
    arr = np.asarray(arr).reshape(-1)
    if expected is not None and arr.shape[0] != expected:
        raise ValueError(f"head.{name} 长度 {arr.shape[0]} != 期望 {expected}（与 npz 行未对齐？）")
    return arr


def _split_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """统一的头级结果结构：{top1, macro_f1, per_class_f1, n}。"""
    macro, per = macro_f1_train(y_true, y_pred, n_classes)
    return {
        "top1": top1(y_true, y_pred),
        "macro_f1": macro,
        "per_class_f1": per,  # 长度恒为 n_classes，未出现/未参与的类为 nan
        "n": int(y_true.shape[0]),
    }


def evaluate_head_metrics(
    z: Any,
    meta: Mapping[str, Any],
    head: HeadPrediction,
    relation: Mapping[str, Any] | str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """在 ``split`` 上计算 5 个头级指标（全精度，**不 round**）。

    Parameters
    ----------
    z:
        特征数组视图（npz / FeatureBundle / dict），需要 ``pair_split``、``slot_pair``、
        ``slot_change``、``slot_type``。
    meta:
        特征元信息，需要 ``type_keys``、``rel_pos_keys``。
    head:
        ``types.HeadPrediction``，change/type 按全局 slot 行对齐，relation / rel_pos /
        attrs 按 relation 采样结果的行对齐（含负样本）。
    relation:
        ``train_heads.build_relation_data`` 产出的 dict（dataset 层负责生成，含负样本）。
        若调用方把 split 作为第 4 个位置参数传入（``evaluate_head_metrics(z, meta, head,
        "test")``）会被识别并纠正；relation 缺失时明确报错，不做隐式重采样。
    split:
        manifest 的 split 字段值，默认 ``"test"``。

    Returns
    -------
    dict
        ``{change:{top1,macro_f1,per_class_f1,n}, type:{...}, relation:{...},
        rel_pos:{...}, attrs:{contact_f1,support_f1,attrs,n}}``
        ``per_class_f1`` 中不可定义的类为 ``nan``（JSON 序列化前请过 ``json_safe``）。
    """
    # 兼容两种调用风格：第 4 个位置参数既可能是 relation dict，也可能是 split 字符串
    if isinstance(relation, str) and isinstance(split, Mapping):
        # (z, meta, head, "test", rel) 风格
        relation, split = split, relation
    elif isinstance(relation, str) and split == "test":
        # (z, meta, head, "test") 风格：只给了 split
        split, relation = relation, None
    if relation is None:
        raise ValueError(
            "evaluate_head_metrics 需要 relation 采样结果（含负样本的 dict）。"
            "请传入 dataset.negative_sampling 生成的结果；本模块不自行重采样，"
            "否则 relation macro-F1 无法复现。"
        )

    # ---------------- 掩码：train_heads.main 第 240-241 行 ----------------
    pair_split = np.asarray(z["pair_split"]).reshape(-1)
    slot_pair = _as_int64(z["slot_pair"])
    slot_split = pair_split[slot_pair]
    slot_mask = slot_split == split
    n_slots = slot_pair.shape[0]

    head_change = _head_array(head, "change", n_slots)
    head_type = _head_array(head, "type", n_slots)

    # ---------------- change（4 类）--------------------------------------
    change_true = _as_int64(z["slot_change"])[slot_mask]
    change = _split_scores(change_true, head_change[slot_mask], len(CHANGE_LABELS))

    # ---------------- type（len(meta["type_keys"]) 类，类别数锁定）--------
    type_keys = list(meta["type_keys"])
    type_true = _as_int64(z["slot_type"])[slot_mask]
    type_res = _split_scores(type_true, head_type[slot_mask], len(type_keys))

    # ---------------- relation（含负样本，n_types 类）---------------------
    rel_split = np.asarray(_rel_get(relation, "split")).reshape(-1)
    rel_label = _as_int64(_rel_get(relation, "label"))
    rel_pos_gt = _as_int64(_rel_get(relation, "pos"))
    rel_contact = _as_int64(_rel_get(relation, "contact"))
    rel_support = _as_int64(_rel_get(relation, "support"))
    n_rel = int(rel_split.shape[0])
    if rel_label.shape[0] != n_rel or rel_pos_gt.shape[0] != n_rel:
        raise ValueError(
            f"relation 字段长度不一致: split={n_rel} label={rel_label.shape[0]} "
            f"pos={rel_pos_gt.shape[0]}"
        )

    # n_types 以采样结果为准（= len(rel["types"]) + 1），同样锁定类别数
    if "n_types" in relation:
        n_rel_types = int(relation["n_types"])
    else:  # pragma: no cover - 兼容调用方漏传 n_types
        n_rel_types = int(rel_label.max()) + 1 if n_rel else 1

    head_relation = _head_array(head, "relation", n_rel)
    head_rel_pos = _head_array(head, "rel_pos", n_rel)
    head_attrs = np.asarray(head.attrs)
    if head_attrs.ndim != 2 or head_attrs.shape[0] != n_rel or head_attrs.shape[1] < 2:
        raise ValueError(
            f"head.attrs 应为 (n_rel,>=2)，实际 {head_attrs.shape}，n_rel={n_rel}"
        )

    rel_mask = rel_split == split
    relation_res = _split_scores(rel_label[rel_mask], head_relation[rel_mask], n_rel_types)

    # ---------------- rel_pos（只看真关系）-------------------------------
    is_none = np.asarray(_rel_get(relation, "is_none")).astype(bool).reshape(-1)
    rel_pos_keys = list(meta["rel_pos_keys"])
    pos_mask = rel_mask & (~is_none)
    rel_pos_res = _split_scores(
        rel_pos_gt[pos_mask], head_rel_pos[pos_mask], len(rel_pos_keys)
    )

    # ---------------- attrs（contact/support 各一个二分类 macro-F1）------
    attr_true = np.stack([rel_contact, rel_support], axis=1)[rel_mask].astype(np.int64)
    attr_pred = head_attrs[rel_mask].astype(np.int64)
    contact_f1, _ = macro_f1_train(attr_true[:, 0], attr_pred[:, 0], 2)
    support_f1, _ = macro_f1_train(attr_true[:, 1], attr_pred[:, 1], 2)
    attrs_res = {
        "contact_f1": contact_f1,
        "support_f1": support_f1,
        # 与 train_attr_head 第 332-335 行一致：两项等权平均
        "attrs": float(np.mean([contact_f1, support_f1])),
        "n": int(attr_true.shape[0]),
    }

    return {
        "change": change,
        "type": type_res,
        "relation": relation_res,
        "rel_pos": rel_pos_res,
        "attrs": attrs_res,
    }
