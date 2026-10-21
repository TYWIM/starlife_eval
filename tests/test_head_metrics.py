"""头级指标单测：镜像 train_heads.main 的 test split 口径，手算数值断言。

玩具数据（3 个 pair，全局 slot 0..5）：
    slot_pair   = [0,0,1,1,2,2]
    pair_split  = [test, test, train]
    slot_change = [0,1,2,0 | 3,0]       (test: [0,1,2,0])
    slot_type   = [0,0,1,1 | 2,2]       (test: [0,0,1,1])
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from starlife_eval.types import HeadPrediction
from starlife_eval.verifiers.head_metrics import evaluate_head_metrics

TYPE_KEYS = ["a", "b", "c"]
REL_TYPES = ["near", "on"]  # 11 关系在真实数据里是 11 个，这里用 2 个便于手算
REL_POS_KEYS = ["front", "back"]
N_REL_TYPES = len(REL_TYPES) + 1  # none = 2


def _z() -> dict:
    return {
        "slot_pair": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "pair_split": np.array(["test", "test", "train"]),
        "slot_change": np.array([0, 1, 2, 0, 3, 0], dtype=np.int64),
        "slot_type": np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
    }


def _meta() -> dict:
    return {"type_keys": TYPE_KEYS, "rel_pos_keys": REL_POS_KEYS}


def _relation() -> dict:
    """4 条关系样本：2 条 test 真关系 + 1 条 test 负样本 + 1 条 train 负样本。"""
    label = np.array([1, 0, N_REL_TYPES - 1, N_REL_TYPES - 1], dtype=np.int32)  # near, on, none, none
    return {
        "frm": np.array([0, 2, 0, 2], dtype=np.int32),
        "to": np.array([1, 3, 2, 1], dtype=np.int32),
        "label": label,
        "pos": np.array([0, 1, -1, -1], dtype=np.int32),
        "contact": np.array([1, 0, 0, 0], dtype=np.int8),
        "support": np.array([0, 1, 0, 0], dtype=np.int8),
        "split": np.array(["test", "test", "test", "train"]),
        "is_none": label == (N_REL_TYPES - 1),
        "n_types": N_REL_TYPES,
        "types": REL_TYPES,
        "rel_pos_keys": REL_POS_KEYS,
    }


def _head() -> HeadPrediction:
    return HeadPrediction(
        change=np.array([0, 2, 2, 1, 0, 0], dtype=np.int64),
        type=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        relation=np.array([1, 2, 2, 2], dtype=np.int64),  # test 行: near✓, on->none✗, none✓
        rel_pos=np.array([0, 0, -1, -1], dtype=np.int64),  # test 真关系行: [0,0]
        attrs=np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64),
        n_relation_positives=2,
    )


def test_change_metrics_exact():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())["change"]
    # true=[0,1,2,0] pred=[0,2,2,1] -> 命中 row0/row2
    assert res["n"] == 4
    assert res["top1"] == pytest.approx(0.5)
    # class0: tp1 fp0 fn1 -> 2/3 ; class1: tp0 fp1 fn1 -> 0 ; class2: tp1 fp1 fn0 -> 2/3
    assert res["per_class_f1"][0] == pytest.approx(2 / 3, rel=1e-15)
    assert res["per_class_f1"][1] == 0.0
    assert res["per_class_f1"][2] == pytest.approx(2 / 3, rel=1e-15)
    assert math.isnan(res["per_class_f1"][3])
    # macro_f1_train = (2/3 + 0 + 2/3) / 3
    assert res["macro_f1"] == pytest.approx((2 / 3 + 0.0 + 2 / 3) / 3, rel=1e-15)
    assert len(res["per_class_f1"]) == 4  # 4 类锁定


def test_type_metrics_lock_n_classes_from_meta():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())["type"]
    assert res["n"] == 4
    assert res["top1"] == 1.0
    assert res["macro_f1"] == 1.0
    # test 里没有 class2（c），但 per_class 长度仍锁定为 len(type_keys)=3
    assert len(res["per_class_f1"]) == 3
    assert res["per_class_f1"][:2] == [1.0, 1.0]
    assert math.isnan(res["per_class_f1"][2])


def test_relation_metrics_include_negatives_and_lock_n_types():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())["relation"]
    # test 行（含负样本）: true=[near,on,none] pred=[near,none,none] -> 命中 2/3
    assert res["n"] == 3
    assert res["top1"] == pytest.approx(2 / 3, rel=1e-15)
    # class0(on): tp0 fp0 fn1 -> 0 ; class1(near): tp1 -> 1.0 ; class2(none): tp1 fp1 -> 2/3
    assert res["per_class_f1"][0] == 0.0
    assert res["per_class_f1"][1] == 1.0
    assert res["per_class_f1"][2] == pytest.approx(2 / 3, rel=1e-15)
    assert res["macro_f1"] == pytest.approx((0.0 + 1.0 + 2 / 3) / 3, rel=1e-15)
    assert len(res["per_class_f1"]) == N_REL_TYPES


def test_rel_pos_metrics_only_on_non_none_test_relations():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())["rel_pos"]
    # 掩码 = test & ~none -> 2 条真关系; true=[0,1] pred=[0,0]
    assert res["n"] == 2
    assert res["top1"] == pytest.approx(0.5)
    # class0: tp1 fp1 fn0 -> 2/3 ; class1: tp0 fp0 fn1 -> 0
    assert res["per_class_f1"][0] == pytest.approx(2 / 3, rel=1e-15)
    assert res["per_class_f1"][1] == 0.0
    assert res["macro_f1"] == pytest.approx((2 / 3 + 0.0) / 2, rel=1e-15)
    assert len(res["per_class_f1"]) == 2


def test_attrs_metrics_exact():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())["attrs"]
    # test 三行 true contact=[1,0,0] pred=[1,0,1] -> class0 tp1 fn1, class1 tp1 fp1 均 2/3
    # support: true=[0,1,0] pred=[0,1,0] -> 1.0
    assert res["n"] == 3
    assert res["contact_f1"] == pytest.approx(2 / 3, rel=1e-15)
    assert res["support_f1"] == 1.0
    assert res["attrs"] == pytest.approx((2 / 3 + 1.0) / 2, rel=1e-15)


def test_train_split_rows_are_excluded():
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())
    assert res["change"]["n"] == 4  # 6 slot 中排除 2 个 train
    assert res["relation"]["n"] == 3  # 4 采样行排除 1 个 train
    # 单看 train split 也应算得出来（换个 split 参数即可）
    train_res = evaluate_head_metrics(_z(), _meta(), _head(), _relation(), split="train")
    assert train_res["change"]["n"] == 2
    assert train_res["relation"]["n"] == 1
    # train 行 true=[3,0] pred=[0,0] -> 1/2
    assert train_res["change"]["top1"] == pytest.approx(0.5)


def test_split_as_fourth_positional_is_tolerated():
    """兼容把 split 写在第 4 个位置参数、relation 写在第 5 个的调用习惯。"""
    res = evaluate_head_metrics(_z(), _meta(), _head(), "test", _relation())
    assert res["change"]["n"] == 4
    assert res["relation"]["n"] == 3

    res_kw = evaluate_head_metrics(_z(), _meta(), _head(), _relation(), split="test")
    assert res_kw["relation"]["macro_f1"] == res["relation"]["macro_f1"]


def test_relation_is_required():
    with pytest.raises(ValueError, match="relation 采样结果"):
        evaluate_head_metrics(_z(), _meta(), _head())


def test_head_length_mismatch_raises():
    bad = HeadPrediction(
        change=np.array([0, 0, 0], dtype=np.int64),
        type=np.array([0, 0, 0], dtype=np.int64),
        relation=np.array([0, 0, 0, 0], dtype=np.int64),
        rel_pos=np.array([0, 0, 0, 0], dtype=np.int64),
        attrs=np.zeros((4, 2), dtype=np.int64),
        n_relation_positives=2,
    )
    with pytest.raises(ValueError, match="change"):
        evaluate_head_metrics(_z(), _meta(), bad, _relation())


def test_float_values_are_not_rounded():
    """全精度输出：不能被 round 到 4 位（0.5556 这种）。"""
    res = evaluate_head_metrics(_z(), _meta(), _head(), _relation())
    assert res["relation"]["macro_f1"] == pytest.approx(5 / 9, rel=1e-15)
    # 5/9 = 0.5555555555555556，若被 round(,4) 会变成 0.5556
    assert res["relation"]["macro_f1"] != 0.5556
