"""运行时统计汇总（ARCHITECTURE §5.5）：只汇总**真实测量值**，禁止编数。

规则（与 ARCHITECTURE §5.5 / §8 一致）：

* 延迟：本地头推理实测、云端回放缓存里的 ``latency_s``，都是实测值；
  本模块只做聚合，保留全精度（不 round，报告层再决定显示位数）；
* token：用缓存 ``usage`` 的真实 ``prompt_tokens``/``completion_tokens`` 求和；
  没有 usage 的调用返回 ``None`` 而不是 0；
* 成本：**只有给出真实计价口径（``cost_basis``）时才允许填 ``cost_usd``**，
  否则保持 ``None`` 并进 ``na``（云端无公开计费口径；本地自有硬件可填 0 + 口径说明）。

本模块不依赖 torch / dataset / models；返回 :class:`starlife_eval.types.RuntimeStats`
（共享契约，只读使用）。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..types import RuntimeStats

__all__ = [
    "from_latencies",
    "from_latencies_s",
    "summarize_tokens",
    "build_runtime_stats",
    "RuntimeStats",
]


# ------------------------------------------------------------------- latencies


def from_latencies(latencies_ms: Sequence[float]) -> tuple[float, float, float]:
    """``(mean, p50, max)``，单位毫秒，全精度、不 round。

    空列表 / 非有限值 / 负值一律 ``ValueError``：没有实测数据就不该产出数字。
    """
    arr = np.asarray(list(latencies_ms), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("没有实测延迟样本，不能产出延迟指标")
    if not np.all(np.isfinite(arr)):
        raise ValueError("延迟里有非有限值（nan/inf），测量数据不合法")
    if np.any(arr < 0):
        raise ValueError("延迟里有负值，测量数据不合法")
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


def from_latencies_s(latencies_s: Sequence[float]) -> tuple[float, float, float]:
    """同 :func:`from_latencies`，输入单位秒（云端缓存 ``latency_s``），输出毫秒。"""
    return from_latencies([float(x) * 1000.0 for x in latencies_s])


# ---------------------------------------------------------------------- tokens


def summarize_tokens(
    usages: Iterable[Mapping[str, Any] | None],
) -> tuple[int | None, int | None]:
    """汇总 ``usage`` 里的 prompt/completion token；完全没有该字段时返回 ``None``。

    与 ``eval_compare_cloud.main`` 第 318-321 行语义一致（逐条 ``get(key, 0)`` 求和），
    但没有 usage 时**不伪造成 0**。
    """
    prompt = completion = 0
    seen_prompt = seen_completion = False
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        value = usage.get("prompt_tokens")
        if value is not None:
            prompt += int(value)
            seen_prompt = True
        value = usage.get("completion_tokens")
        if value is not None:
            completion += int(value)
            seen_completion = True
    return (
        prompt if seen_prompt else None,
        completion if seen_completion else None,
    )


# ---------------------------------------------------------------- RuntimeStats


def build_runtime_stats(
    *,
    latencies_ms: Sequence[float] | None = None,
    latencies_s: Sequence[float] | None = None,
    usages: Iterable[Mapping[str, Any] | None] | None = None,
    tokens_prompt: int | None = None,
    tokens_completion: int | None = None,
    cost_usd: float | None = None,
    cost_basis: str | None = None,
    n_calls: int | None = None,
    source: str = "measured",
) -> RuntimeStats:
    """组装 :class:`RuntimeStats`；无法测的字段保持 ``None``。

    Parameters
    ----------
    latencies_ms / latencies_s:
        二选一的实测延迟序列。``latencies_s`` 用于云端回放缓存的 ``latency_s``。
    usages:
        原始 ``usage`` dict 序列；未显式给 ``tokens_*`` 时由 :func:`summarize_tokens` 汇总。
    cost_usd / cost_basis:
        成本必须带真实计价口径；只给金额不给口径会报错（防止编造成本）。
        本地头：``cost_usd=0.0, cost_basis="self-hosted hardware; no cloud billing"``。
        云端无计费口径：两者都不传。
    n_calls:
        调用次数；默认取延迟样本数。
    source:
        数据来源标记，默认 ``"measured"``；引用第三方基准时可标 ``"bench"``。
    """
    if latencies_ms is not None and latencies_s is not None:
        raise ValueError("latencies_ms 与 latencies_s 只能给一个")
    stats = RuntimeStats(source=source)

    latencies = None
    if latencies_ms is not None:
        latencies = list(latencies_ms)
    elif latencies_s is not None:
        latencies = [float(x) * 1000.0 for x in latencies_s]
    if latencies is not None:
        mean_ms, p50_ms, max_ms = from_latencies(latencies)
        stats.latency_ms_mean = mean_ms
        stats.latency_ms_p50 = p50_ms
        stats.latency_ms_max = max_ms
        if n_calls is None:
            n_calls = len(latencies)

    if usages is not None and tokens_prompt is None and tokens_completion is None:
        tokens_prompt, tokens_completion = summarize_tokens(usages)
    stats.tokens_prompt = None if tokens_prompt is None else int(tokens_prompt)
    stats.tokens_completion = None if tokens_completion is None else int(tokens_completion)

    if cost_usd is not None and not cost_basis:
        raise ValueError(
            "cost_usd 必须同时给出真实计价口径 cost_basis（否则就是编造成本）；"
            "无公开计费口径时应保持 None 并在报告里标 na"
        )
    stats.cost_usd = None if cost_usd is None else float(cost_usd)
    stats.cost_basis = cost_basis if cost_usd is not None else None
    stats.n_calls = None if n_calls is None else int(n_calls)
    return stats
