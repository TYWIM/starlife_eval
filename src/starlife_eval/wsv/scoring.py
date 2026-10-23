"""OpenWSV 确定性判分与统计（无 LLM-judge、无网络）。

口径（冻结在 ``openwsv-1``）：

* **覆盖率 ≠ 错误**：模型调用报错、空输出、或选项字母解析失败都记为
  *未覆盖*（``covered=False``），从准确率分母中剔除，而不是按错题计；
* 准确率只在覆盖题上计算（``n_covered`` 为分母），同时报告
  ``coverage = n_covered / n``；
* 95% 置信区间用任务级 bootstrap（默认 1,000 draws，分位法），
  种子固定（框架 ``bootstrap_seed``，默认 20261022），结果确定；
* 两模型对比用精确 McNemar 检验（不一致对上做二项精确检验）。

``tasks`` 是 :func:`starlife_eval.wsv.tasks.load_suite` 的规范化 dict 列表，
``preds`` 是 ``{task_id: {"text": str | None, "error": str | None,
"latency_s": float | None}}``。
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "BOTH_SIDED_LEVEL",
    "parse_choice",
    "score_choices",
    "bootstrap_ci",
    "mcnemar",
]

BOTH_SIDED_LEVEL = 0.95
DEFAULT_BOOT = 1000
DEFAULT_BOOTSTRAP_SEED = 20261022

# ------------------------------------------------------------------- parsing


# 显式 "Answer/答案" 形式；要求后面紧跟一个独立字母（不是另一个单词的首字母），
# 避免从解释性文字里乱抓。
_ANSWER_KEYWORD = re.compile(
    r"(?:final answer|answer|option|答案|选择|选项|选)\s*(?:is|是|为)?\s*[:：]?\s*"
    r"[\(\[（【]?\s*([A-Za-z])(?![A-Za-z])\s*[\)\]）】]?",
    re.IGNORECASE,
)
# 整个回复就是一个字母（可带括号/句点/全角标点），例如 "A"、"(C)"、"D."。
_BARE_OPTION = re.compile(r"^[\(\[（【]?\s*([A-Za-z])\s*[\)\]）】]?[.。]?$")
# 代码围栏/加粗标记等：只去掉包装字符，不改变字母本身。
_WRAPPERS = "`*_~#"


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("\u200b", "").replace("\ufeff", "")
    return t.strip()


def _strip_wrappers(text: str) -> str:
    t = text.strip()
    while t and t[0] in _WRAPPERS:
        t = t[1:].lstrip()
    while t and t[-1] in _WRAPPERS:
        t = t[:-1].rstrip()
    return t


def parse_choice(text: str | None, n_choices: int) -> int | None:
    """严格解析模型回复里的选项字母，返回 0-based 下标；无法解析返回 ``None``。

    只接受以下形式（NFKC 归一化后）：

    * ``"A"`` / ``"b"`` / ``"(C)"`` / ``"D."``（整条回复就是一个字母）；
    * ``"Answer: D"`` / ``"answer is (c)" / ``"Final answer: B"``；
    * 中文 ``"选 A"`` / ``"选择：B"`` / ``"答案：C"``；
    * ``"Option D"`` / 反引号或加粗包裹的单个字母（```B```、``**C**``）。

    解析失败（自由文本、多字母、字母越界）返回 ``None``——按口径这是
    *未覆盖*，不是错题。绝不从解释性句子里猜字母。
    """
    if text is None:
        return None
    n = int(n_choices)
    if n <= 0:
        return None
    raw = _normalize(text)
    if not raw:
        return None

    def to_index(letter: str) -> int | None:
        idx = ord(letter.upper()) - ord("A")
        return idx if 0 <= idx < n else None

    candidate = _strip_wrappers(raw)
    # 1) 整条回复就是一个选项字母
    m = _BARE_OPTION.match(candidate)
    if m:
        return to_index(m.group(1))

    # 2) 显式 answer/答案/选 关键词
    m = _ANSWER_KEYWORD.search(candidate)
    if m:
        return to_index(m.group(1))

    return None


# ------------------------------------------------------------------- scoring


def _pred_record(pred: Mapping[str, Any] | None) -> dict[str, Any]:
    if pred is None:
        return {"text": None, "error": "missing prediction", "latency_s": None}
    return {
        "text": pred.get("text"),
        "error": pred.get("error"),
        "latency_s": pred.get("latency_s"),
    }


def score_choices(
    tasks: Sequence[Mapping[str, Any]],
    preds: Mapping[str, Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """对 choice 套件按"覆盖率"口径打分。

    Returns
    -------
    dict
        ``{n, n_covered, coverage, accuracy, ci95, n_correct, n_wrong,
        n_unparsable, n_error, latency_s_mean, per_task}``。

        * ``accuracy`` 以覆盖题为分母（``n_covered==0`` 时为 ``0.0``）；
        * ``ci95`` 是覆盖题正确性向量的 bootstrap 分位区间；
        * ``per_task`` 每项固定 ``{task_id, covered, pred, gold, error}``，
          其中 ``pred`` 是解析出的 0-based 下标（未覆盖为 ``None``），
          ``gold`` 是任务答案下标，``error`` 记录未覆盖原因（覆盖题为 ``None``）。
    """
    per_task: list[dict[str, Any]] = []
    flags: list[bool] = []
    n_wrong = n_unparsable = n_error = 0
    latencies: list[float] = []

    for task in tasks:
        gold = int(task["answer"])
        rec = _pred_record(preds.get(str(task["task_id"])))
        if rec["latency_s"] is not None:
            latencies.append(float(rec["latency_s"]))

        error = rec["error"]
        text = rec["text"]
        pred = None
        covered = False
        if not error:
            pred = parse_choice(text if text is None else str(text), len(task.get("choices") or ()))
            covered = pred is not None

        if covered:
            flags.append(pred == gold)
            if pred != gold:
                n_wrong += 1
            reason = None
        elif error:
            n_error += 1
            reason = str(error)
        else:
            n_unparsable += 1
            reason = "unparsable choice"

        per_task.append(
            {
                "task_id": str(task["task_id"]),
                "covered": covered,
                "pred": pred,
                "gold": gold,
                "error": reason,
            }
        )

    n = len(tasks)
    n_covered = len(flags)
    point, lo, hi = bootstrap_ci(flags, n_boot=n_boot, seed=seed)
    accuracy = point if n_covered else 0.0
    coverage = (n_covered / n) if n else 0.0
    return {
        "n": n,
        "n_covered": n_covered,
        "coverage": coverage,
        "accuracy": accuracy,
        "ci95": [lo, hi] if n_covered else [0.0, 0.0],
        "n_correct": int(sum(flags)),
        "n_wrong": n_wrong,
        "n_unparsable": n_unparsable,
        "n_error": n_error,
        "latency_s_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "per_task": per_task,
    }


# ---------------------------------------------------------------- statistics


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    level: float = BOTH_SIDED_LEVEL,
) -> tuple[float, float, float]:
    """任务级 bootstrap 分位置信区间，返回 ``(point, lo, hi)``。

    * 输入是 0/1（或任意数值）列表，point 是均值；
    * 空输入返回 ``(0.0, 0.0, 0.0)``；
    * 用 ``numpy.random.default_rng(seed)``，同一 seed 结果逐位确定；
    * 重采样单位是"任务"（每条数据一次有放回抽样），不是模型调用。
    """
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，实际 {n_boot}")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level 必须在 (0,1) 内，实际 {level}")
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    n = int(arr.size)
    draws = rng.integers(0, n, size=(int(n_boot), n))
    means = arr[draws].mean(axis=1)
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return float(arr.mean()), float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def mcnemar(
    flags_a: Sequence[bool],
    flags_b: Sequence[bool],
) -> dict[str, Any]:
    """对齐两个正确性向量，做精确 McNemar 检验（双侧）。

    ``b`` = A 错 B 对的数量，``c`` = A 对 B 错的数量；在 ``n=b+c`` 个不一致
    样本上做 ``p = 2 * P(X <= min(b,c))``，``X ~ Binomial(n, 0.5)``，截断到 1。
    无不一致对时 ``p_value=1.0``。
    """
    a = list(flags_a)
    b_flags = list(flags_b)
    if len(a) != len(b_flags):
        raise ValueError(f"mcnemar 需要等长向量，实际 {len(a)} vs {len(b_flags)}")
    b = sum(1 for x, y in zip(a, b_flags) if (not x) and y)
    c = sum(1 for x, y in zip(a, b_flags) if x and (not y))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "note": "no discordant pairs"}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** n))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}