"""头级与 60 对对照协议共用的纯数值指标（只依赖 numpy + 标准库）。

口径来源（**只读**移植，`starlife/` 原仓库不被 import）：
- :func:`macro_f1_train` 逐行移植
  ``starlife/starlife_scripts/train_heads.py::macro_f1``（第 35-46 行）：
  逐类 F1；``tp+fp+fn == 0`` 的类记 ``nan``；对非 ``nan`` 类等权平均；全 ``nan`` 时 ``0.0``。
- :func:`macro_f1_compare` 逐行移植
  ``starlife/starlife_scripts/eval_compare_cloud.py::macro_f1``（第 31-41 行）：
  逐类 F1；**``tp == 0`` 的类整类跳过**；全部跳过时 ``0.0``。

两者语义不同，**不可互换**（这是 change macro-F1 本地 0.6983/0.7232 与云端 0.7781
差异的来源）：
- 头级全 split 评测（L2，ARCHITECTURE §5.1）→ :func:`macro_f1_train`；
- 60 对云端对照协议（ARCHITECTURE §5.2）→ :func:`macro_f1_compare`。

本模块禁止 import torch / dataset / models，输入输出都是 numpy / Python 标量。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

__all__ = [
    "macro_f1_train",
    "macro_f1_compare",
    "top1",
    "accuracy",
    "per_class_f1",
    "confusion_matrix",
    "MetricSet",
    "json_safe",
]


# --------------------------------------------------------------------- helpers


def _as_1d_ints(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """转成对齐的一维 int64 数组（长度不一致立即报错，避免静默错位）。"""
    t = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if t.shape[0] != p.shape[0]:
        raise ValueError(f"y_true/y_pred 长度不一致: {t.shape[0]} vs {p.shape[0]}")
    return t, p


def _class_counts(t: np.ndarray, p: np.ndarray, c: int) -> tuple[int, int, int]:
    """返回 (tp, fp, fn)，与 train_heads.macro_f1 的取法逐字一致。"""
    tp = int(((p == c) & (t == c)).sum())
    fp = int(((p == c) & (t != c)).sum())
    fn = int(((p != c) & (t == c)).sum())
    return tp, fp, fn


# -------------------------------------------------------------- macro F1 (two)


def macro_f1_train(y_true: Any, y_pred: Any, n_classes: int) -> tuple[float, list[float]]:
    """train_heads 口径的 macro-F1：跳过 ``tp+fp+fn == 0`` 的类。

    逐行对齐 ``train_heads.py::macro_f1``::

        for c in range(n_classes):
            tp/fp/fn ...
            if tp + fp + fn == 0: f1s.append(nan); continue
            f1s.append(2*tp / (2*tp + fp + fn))
        return mean(valid) if valid else 0.0, f1s

    Returns
    -------
    (score, per_class_f1)
        ``per_class_f1`` 长度恒等于 ``n_classes``，未参与平均的类为 ``nan``。
    """
    if n_classes < 1:
        raise ValueError(f"n_classes 必须 >= 1，实际 {n_classes}")
    t, p = _as_1d_ints(y_true, y_pred)
    f1s: list[float] = []
    for c in range(n_classes):
        tp, fp, fn = _class_counts(t, p, c)
        if tp + fp + fn == 0:
            f1s.append(float("nan"))
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    valid = [f for f in f1s if not math.isnan(f)]
    return (float(np.mean(valid)) if valid else 0.0), f1s


def macro_f1_compare(y_true: Any, y_pred: Any, n_classes: int) -> float:
    """eval_compare_cloud 口径的 macro-F1：**跳过 ``tp == 0`` 的类**。

    逐行对齐 ``eval_compare_cloud.py::macro_f1``::

        for c in range(n):
            tp/fp/fn ...
            if tp == 0: continue
            p = tp/(tp+fp); r = tp/(tp+fn)
            f1s.append(2*p*r/(p+r))
        return float(np.mean(f1s)) if f1s else 0.0

    注意：云端的 change macro-F1 ``0.7781`` 依赖此口径；换成 train 口径会变小。
    """
    if n_classes < 1:
        raise ValueError(f"n_classes 必须 >= 1，实际 {n_classes}")
    t, p = _as_1d_ints(y_true, y_pred)
    f1s: list[float] = []
    for c in range(n_classes):
        tp, fp, fn = _class_counts(t, p, c)
        if tp == 0:
            continue
        precision = tp / (tp + fp)  # tp>0 时分母 >= 1
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


# ----------------------------------------------------------------- accuracy etc


def top1(y_true: Any, y_pred: Any) -> float:
    """逐样本 top-1 准确率；空输入返回 ``0.0``（与对照脚本的空集兜底一致）。"""
    t, p = _as_1d_ints(y_true, y_pred)
    if t.shape[0] == 0:
        return 0.0
    return float((t == p).mean())


#: top1 的同义名（ARCHITECTURE §5.1 用 top1 / accuracy 两种称呼）。
accuracy = top1


def per_class_f1(
    y_true: Any,
    y_pred: Any,
    n_classes_locked: int,
) -> list[float]:
    """锁定类别数的逐类 F1。

    **类别数必须来自数据集元信息（例如 ``len(meta["type_keys"])``），
    不能由 test 集实际出现的类推出**：test 里缺某类时该位置仍是 ``nan``，
    列表长度保持不变，指标分母口径不会漂移。
    """
    _, per = macro_f1_train(y_true, y_pred, n_classes_locked)
    return per


def confusion_matrix(y_true: Any, y_pred: Any, n_classes: int) -> list[list[int]]:
    """``n_classes × n_classes`` 混淆矩阵（``matrix[true][pred]``）。

    越界标签（<0 或 >=n_classes，例如模型吐出非法类别）被忽略；
    它们仍会通过 :func:`top1` 计入错误（与原脚本 ``pred == y_true`` 的均值一致）。
    """
    if n_classes < 1:
        raise ValueError(f"n_classes 必须 >= 1，实际 {n_classes}")
    t, p = _as_1d_ints(y_true, y_pred)
    n = int(n_classes)
    valid = (t >= 0) & (t < n) & (p >= 0) & (p < n)
    flat = t[valid] * n + p[valid]
    counts = np.bincount(flat, minlength=n * n).reshape(n, n)
    return counts.astype(np.int64).tolist()


# -------------------------------------------------------------------- MetricSet


class MetricSet:
    """可锁定类别数的分块累积指标器（change/type/relation/rel_pos 通用）。

    用法::

        ms = MetricSet(n_classes=len(meta["type_keys"]))
        for chunk in chunks:
            ms.accumulate(y_true_chunk, y_pred_chunk)
        result = ms.compute()   # {top1, macro_f1, per_class, confusion, n, ...}

    ``top1`` / ``confusion`` 用全部样本（与原脚本一致）；``macro_f1`` / ``per_class``
    用 :func:`macro_f1_train` 口径（L2 头级指标）。
    """

    def __init__(self, n_classes: int) -> None:
        if int(n_classes) < 1:
            raise ValueError(f"n_classes 必须 >= 1，实际 {n_classes}")
        self.n_classes = int(n_classes)
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []

    # -- 累积 ---------------------------------------------------------------
    def accumulate(self, y_true: Any, y_pred: Any) -> "MetricSet":
        t, p = _as_1d_ints(y_true, y_pred)
        self._chunks.append((t, p))
        return self

    # 兼容其它命名
    update = accumulate
    add = accumulate

    def reset(self) -> None:
        self._chunks.clear()

    @property
    def n(self) -> int:
        return int(sum(t.shape[0] for t, _ in self._chunks))

    # -- 计算 ---------------------------------------------------------------
    def _stacked(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._chunks:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        return (
            np.concatenate([t for t, _ in self._chunks]),
            np.concatenate([p for _, p in self._chunks]),
        )

    def compute(self) -> dict[str, Any]:
        """返回未 round 的全精度指标（JSON 序列化前再过 :func:`json_safe`）。"""
        t, p = self._stacked()
        macro, per = macro_f1_train(t, p, self.n_classes)
        return {
            "top1": top1(t, p),
            "macro_f1": macro,
            "macro_f1_compare": macro_f1_compare(t, p, self.n_classes),
            "per_class": per,
            "confusion": confusion_matrix(t, p, self.n_classes),
            "n": int(t.shape[0]),
        }


# ---------------------------------------------------------------- JSON helpers


def json_safe(obj: Any) -> Any:
    """把指标里的 ``nan`` 转成 ``None``、numpy 标量转 Python 标量。

    报告/前端契约（ARCHITECTURE §5.6）要求不可测维度是 ``null`` 而不是 ``NaN``；
    ``json.dumps(..., allow_nan=False)`` 也需要这一步才能写出合法 JSON。
    """
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
