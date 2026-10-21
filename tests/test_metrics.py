"""指标单测：手算数值断言（不只断言"不报错"）。

覆盖 ARCHITECTURE §5.1 / §5.2 的关键差异：macro_f1_train 跳过 tp+fp+fn==0 的类，
macro_f1_compare 跳过 tp==0 的类。两类口径在同一组输入上必须给出不同数值。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from starlife_eval.verifiers.metrics import (
    MetricSet,
    accuracy,
    confusion_matrix,
    json_safe,
    macro_f1_compare,
    macro_f1_train,
    per_class_f1,
    top1,
)


# ---------------------------------------------------------------------------
# 手算例子 A（两种 macro-F1 语义差异的来源）
#
#   y_true = [0, 0, 0, 1, 1]
#   y_pred = [0, 0, 0, 1, 2]     n_classes=4
#
# class0: tp=3 fp=0 fn=0               -> f1 = 1.0
# class1: tp=1 fp=0 fn=1 (row4)        -> f1 = 2*1/(2*1+0+1) = 2/3
# class2: tp=0 fp=1 fn=0 (row4)        -> 0/1 = 0.0  ← train 计入、compare 跳过
# class3: tp+fp+fn=0                   -> nan       ← 两者都跳过
#
# train macro  = (1.0 + 2/3 + 0) / 3   = 0.5555...
# compare macro= (1.0 + 2/3) / 2       = 0.8333...
# top1         = 4/5                   = 0.8
# ---------------------------------------------------------------------------
Y_TRUE_A = np.array([0, 0, 0, 1, 1])
Y_PRED_A = np.array([0, 0, 0, 1, 2])


def test_macro_f1_train_skips_empty_classes_and_keeps_zero_f1_class():
    score, per = macro_f1_train(Y_TRUE_A, Y_PRED_A, 4)

    # per-class 长度锁定为 4，缺类（class3）保持 nan
    assert len(per) == 4
    assert per[0] == 1.0
    assert per[1] == pytest.approx(2 / 3, rel=1e-15)
    assert per[2] == 0.0
    assert math.isnan(per[3])

    # 等权平均 (1.0 + 2/3 + 0.0) / 3
    assert score == pytest.approx((1.0 + 2.0 / 3.0 + 0.0) / 3.0, rel=1e-15)


def test_macro_f1_compare_skips_zero_tp_class():
    score = macro_f1_compare(Y_TRUE_A, Y_PRED_A, 4)
    # class2 的 tp==0 被整类跳过 -> 只平均 class0/class1
    assert score == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, rel=1e-15)


def test_the_two_macro_f1_semantics_differ_on_same_input():
    """硬要求：构造 tp==0 但 fp>0 的类，验证两种口径给出不同数值。

    云端 change macro-F1 0.7781 依赖 compare 口径；这里 compare 明显更高。
    """
    train, _ = macro_f1_train(Y_TRUE_A, Y_PRED_A, 4)
    compare = macro_f1_compare(Y_TRUE_A, Y_PRED_A, 4)
    assert compare > train
    # 差异来自 class2 的 0 分是否计入分母：1.667/3 vs 1.667/2
    assert train == pytest.approx(0.5555555555555556, rel=1e-15)
    assert compare == pytest.approx(0.8333333333333334, rel=1e-15)


def test_macro_f1_all_nan_returns_zero():
    # 只有类 5 有样本，n_classes=2 -> 两类都是 tp+fp+fn==0
    score, per = macro_f1_train([5, 5], [5, 5], 2)
    assert score == 0.0
    assert math.isnan(per[0]) and math.isnan(per[1])
    # compare 全跳过也是 0.0
    assert macro_f1_compare([5, 5], [5, 5], 2) == 0.0


def test_macro_f1_input_length_mismatch_raises():
    with pytest.raises(ValueError, match="长度不一致"):
        macro_f1_train([0, 1], [0], 2)
    with pytest.raises(ValueError, match="长度不一致"):
        macro_f1_compare([0, 1], [0], 2)


# ---------------------------------------------------------------------- top1


def test_top1_exact_and_alias():
    # 4/5 命中（row4: true=1, pred=2）
    assert top1(Y_TRUE_A, Y_PRED_A) == pytest.approx(4 / 5)
    assert accuracy(Y_TRUE_A, Y_PRED_A) == pytest.approx(4 / 5)
    assert top1([], []) == 0.0


def test_top1_counts_out_of_range_prediction_as_wrong():
    assert top1([0, 1], [0, 99]) == 0.5


# ------------------------------------------------------------ per-class / locks


def test_per_class_f1_locks_n_classes():
    # 只有 class0 出现；锁定 6 类后仍然返回 6 个位置
    per = per_class_f1([0, 0], [0, 0], 6)
    assert len(per) == 6
    assert per[0] == 1.0
    assert all(math.isnan(v) for v in per[1:])

    # 同样的预测不锁的话只有 1 类，会掩盖"test 缺类"
    per_locked = per_class_f1([0, 1, 1], [0, 1, 0], 6)
    assert len(per_locked) == 6
    assert per_locked[0] == pytest.approx(2 / 3, rel=1e-15)
    assert per_locked[1] == pytest.approx(2 / 3, rel=1e-15)
    assert all(math.isnan(v) for v in per_locked[2:])


def test_per_class_f1_requires_positive_n_classes():
    with pytest.raises(ValueError):
        per_class_f1([0], [0], 0)


# ------------------------------------------------------------- confusion matrix


def test_confusion_matrix_exact():
    cm = confusion_matrix(Y_TRUE_A, Y_PRED_A, 4)
    assert cm == [
        [3, 0, 0, 0],  # true = 0
        [0, 1, 1, 0],  # true = 1
        [0, 0, 0, 0],  # true = 2
        [0, 0, 0, 0],  # true = 3
    ]
    # 行和 = 该真实类的样本数
    assert [sum(row) for row in cm] == [3, 2, 0, 0]


def test_confusion_matrix_ignores_out_of_range():
    cm = confusion_matrix([0, 1, 5], [0, 1, 9], 2)
    assert cm == [[1, 0], [0, 1]]


# -------------------------------------------------------------------- MetricSet


def test_metric_set_chunked_equals_single_shot():
    t = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    p = np.array([0, 1, 1, 1, 2, 3, 2, 2])
    ms = MetricSet(n_classes=4)
    ms.accumulate(t[:3], p[:3]).accumulate(t[3:5], p[3:5]).accumulate(t[5:], p[5:])
    got = ms.compute()

    macro, per = macro_f1_train(t, p, 4)
    assert ms.n == 8
    assert got["n"] == 8
    assert got["top1"] == pytest.approx(top1(t, p))
    assert got["macro_f1"] == pytest.approx(macro)
    for got_v, exp_v in zip(got["per_class"], per, strict=True):
        if math.isnan(exp_v):
            assert math.isnan(got_v)
        else:
            assert got_v == pytest.approx(exp_v)
    assert got["confusion"] == confusion_matrix(t, p, 4)
    # compare 口径一并给出，方便对照协议复用
    assert got["macro_f1_compare"] == pytest.approx(macro_f1_compare(t, p, 4))


def test_metric_set_empty_is_well_defined():
    ms = MetricSet(n_classes=3)
    got = ms.compute()
    assert got["n"] == 0
    assert got["top1"] == 0.0
    assert got["macro_f1"] == 0.0
    assert len(got["per_class"]) == 3
    assert all(math.isnan(v) for v in got["per_class"])
    assert got["confusion"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_metric_set_reset():
    ms = MetricSet(n_classes=2)
    ms.accumulate([0, 1], [0, 1])
    assert ms.n == 2
    ms.reset()
    assert ms.n == 0


def test_metric_set_requires_positive_n_classes():
    with pytest.raises(ValueError):
        MetricSet(n_classes=0)


# ------------------------------------------------------------------ json_safe


def test_json_safe_converts_nan_to_null():
    _, per = macro_f1_train(Y_TRUE_A, Y_PRED_A, 4)
    payload = {"per_class": per, "score": 0.5, "confusion": confusion_matrix(Y_TRUE_A, Y_PRED_A, 4)}
    safe = json_safe(payload)
    # nan -> None，且 allow_nan=False 能写出合法 JSON
    assert safe["per_class"][3] is None
    assert safe["per_class"][0] == 1.0
    text = json.dumps(safe, allow_nan=False)
    assert "NaN" not in text
    assert "null" in text


# ---------------------------------------------------------------- runtime stats
# （runtime.py 也归本任务；测试放在本文件以保持验证命令只有 3 个测试文件）


def test_from_latencies_mean_p50_max():
    from starlife_eval.verifiers.runtime import from_latencies

    assert from_latencies([1.0, 2.0, 3.0, 4.0]) == (2.5, 2.5, 4.0)
    # 全精度：不能被 round 到 1 位/整数
    mean, p50, vmax = from_latencies([0.123456789])
    assert mean == 0.123456789
    assert p50 == 0.123456789
    assert vmax == 0.123456789


def test_from_latencies_rejects_unmeasured():
    from starlife_eval.verifiers.runtime import from_latencies

    with pytest.raises(ValueError, match="没有实测延迟"):
        from_latencies([])
    with pytest.raises(ValueError, match="非有限"):
        from_latencies([1.0, float("nan")])
    with pytest.raises(ValueError, match="负值"):
        from_latencies([-1.0, 2.0])


def test_from_latencies_seconds_to_ms():
    from starlife_eval.verifiers.runtime import from_latencies_s

    mean_ms, p50_ms, max_ms = from_latencies_s([0.1, 0.2, 0.3])
    assert mean_ms == pytest.approx(200.0)
    assert p50_ms == pytest.approx(200.0)
    assert max_ms == pytest.approx(300.0)


def test_summarize_tokens_real_values():
    from starlife_eval.verifiers.runtime import summarize_tokens

    usages = [
        {"prompt_tokens": 10, "completion_tokens": 3},
        {"prompt_tokens": 5, "completion_tokens": 7},
    ]
    assert summarize_tokens(usages) == (15, 10)
    # 没有 usage 时是 None（不是伪造的 0）
    assert summarize_tokens([]) == (None, None)
    assert summarize_tokens([None, {}]) == (None, None)
    # 只有一种 token 字段时另一种保持 None
    assert summarize_tokens([{"prompt_tokens": 10}, {"prompt_tokens": 5}]) == (15, None)


def test_build_runtime_stats():
    from starlife_eval.verifiers.runtime import build_runtime_stats

    stats = build_runtime_stats(
        latencies_ms=[100.0, 200.0, 300.0],
        usages=[{"prompt_tokens": 7, "completion_tokens": 2}],
        source="measured",
    )
    assert stats.latency_ms_mean == pytest.approx(200.0)
    assert stats.latency_ms_p50 == pytest.approx(200.0)
    assert stats.latency_ms_max == pytest.approx(300.0)
    assert stats.n_calls == 3
    assert stats.tokens_prompt == 7
    assert stats.tokens_completion == 2
    # 无计价口径 -> cost 保持 None
    assert stats.cost_usd is None
    assert stats.cost_basis is None

    # 云端回放缓存的 latency_s
    cloud = build_runtime_stats(latencies_s=[6.0, 6.5])
    assert cloud.latency_ms_mean == pytest.approx(6250.0)
    assert cloud.n_calls == 2

    # 本地自有硬件：0 成本必须带口径说明
    local = build_runtime_stats(
        latencies_ms=[10.0],
        cost_usd=0.0,
        cost_basis="self-hosted hardware; no cloud billing",
    )
    assert local.cost_usd == 0.0
    assert local.cost_basis == "self-hosted hardware; no cloud billing"


def test_build_runtime_stats_refuses_fabricated_cost():
    from starlife_eval.verifiers.runtime import build_runtime_stats

    with pytest.raises(ValueError, match="计价口径"):
        build_runtime_stats(cost_usd=0.123)  # 没有 cost_basis 就是编造
    with pytest.raises(ValueError, match="只能给一个"):
        build_runtime_stats(latencies_ms=[1.0], latencies_s=[0.001])
