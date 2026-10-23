"""OpenWSV 冻结提示词模板（prompt version ``openwsv-1``）。

**冻结语义**：``PROMPT_VERSION`` 一旦发布，``build_prompt`` 对同一 ``(task,
protocol)`` 必须逐字稳定——不依赖时间、路径、随机数、环境变量。框架五件套里的
"提示词版本"就是这个常量；改动模板必须同时升级版本号，否则不同轮次结果不可比。

三种 protocol：

* ``choice``：单图/多图多选，要求模型**只输出一个选项字母**；
* ``short``：短答，要求一行 ``Answer: <短答>``；
* ``json``：结构化输出，给出 JSON schema（用于结构化世界模型/需要 JSON 的模型）。

注意：``spatialmqa`` / ``spatialmqa_relation`` 用 ``choice``；``pairs`` 套件走
:mod:`starlife_eval.wsv.state` 的结构化协议，不经过本模块的文本提示词。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = ["PROMPT_VERSION", "PROTOCOLS", "DEFAULT_JSON_SCHEMA", "build_prompt"]

#: 冻结提示词版本（进入结果 JSON 的 ``framework.prompt_version``）
PROMPT_VERSION = "openwsv-1"

PROTOCOLS: tuple[str, ...] = ("choice", "short", "json")

#: ``json`` protocol 未在 task.meta 指定 schema 时使用的默认结构（与
#: :class:`starlife_eval.types.PairPrediction` 对齐，slot 用整数编号）。
DEFAULT_JSON_SCHEMA: dict[str, Any] = {
    "changes": {"<slot>": "none|appeared|displaced|disappeared"},
    "relations": [
        {
            "from_slot": 0,
            "to_slot": 1,
            "type": "<relation type>",
            "rel_pos": "<spatial position or null>",
            "contact": False,
            "support": False,
        }
    ],
}

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _image_order_line(n_images: int) -> str:
    """多图顺序说明。

    只写序号（不写文件路径），保证同一任务在任何机器上提示词逐字相同。
    """
    if n_images <= 1:
        return ""
    n = min(n_images, len(_LETTERS))
    labels = ", ".join(f"image {i + 1}" for i in range(n))
    return (
        f"The images are provided in this order: {labels}. "
        "Each image is an independent view unless stated otherwise."
    )


def _choice_block(task: Mapping[str, Any]) -> str:
    choices: Sequence[str] = task.get("choices") or ()
    if not choices:
        raise ValueError(f"choice protocol 需要 choices，task {task.get('task_id')!r} 没有")
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    lines = ["Options:"]
    for letter, text in zip(letters, choices):
        lines.append(f"{letter}. {text}")
    return "\n".join(lines)


def _schema_for(task: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = task.get("meta") or {}
    schema = meta.get("schema")
    return schema if isinstance(schema, Mapping) else DEFAULT_JSON_SCHEMA


def build_prompt(task: Mapping[str, Any], *, protocol: str) -> str:
    """构造一条任务的用户提示词（图像另行以 data URL 传入）。

    Parameters
    ----------
    task:
        :func:`starlife_eval.wsv.tasks.load_suite` 规范化后的任务 dict。
    protocol:
        ``choice`` | ``short`` | ``json``。

    Returns
    -------
    str
        逐字稳定的提示词（同一 task/protocol 多次调用结果完全相同）。

    Raises
    ------
    ValueError
        未知 protocol，或 ``choice`` protocol 缺选项。
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"未知 protocol {protocol!r}；可用: {', '.join(PROTOCOLS)}")

    parts: list[str] = ["Look at the image(s) and answer the following spatial question."]

    order = _image_order_line(len(task.get("images") or ()))
    if order:
        parts.append(order)

    parts.append("Question: " + str(task.get("question", "")).strip())

    if protocol == "choice":
        parts.append(_choice_block(task))
        parts.append(
            'Reply with the option letter only (for example "A"). '
            "Do not write the option text and do not explain."
        )
    elif protocol == "short":
        parts.append('Reply with a short phrase on a single line: "Answer: <your answer>".')
    else:  # json
        schema = json.dumps(_schema_for(task), ensure_ascii=False, sort_keys=True)
        parts.append(
            "Reply with one JSON object only, no prose and no code fences. "
            "Expected JSON schema: " + schema
        )

    return "\n\n".join(parts)
