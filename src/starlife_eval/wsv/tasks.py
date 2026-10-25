"""OpenWSV 冻结任务套件加载（frozen data version）。

七套 suite（名字与 ``SUITES`` 一致）：

======================  ======  ===========================================
suite                    n      说明
======================  ======  ===========================================
``spatialmqa``           1612    SpatialMQA 单图多选（T1 spatial relations）
``spatialmqa_relation``  305     上者中用正则筛出的 "Where is/are the X
                                 located relative to the Y?" 关系题
``pairs``                1000    ``dataset_spatial_v3`` split=test 的 before/after
                                 帧对（结构化世界模型输入，T4–T6）
``change``               1000    由 pairs GT 派生的 T4 空间变化文本题（一 pair 一题）
``relation_change``      600     T5 关系变化（合成数据 openwsv-data-1.1）
``state_replay``         1000    由 pairs GT 派生的 T6 回放后状态事实题
``cross_frame``          400     T7 跨帧不相交空间关系（合成数据 openwsv-data-1.1）
======================  ======  ===========================================

任务文件位于本包 ``suites/``，属于**冻结框架**的一部分：内容一旦发布不得修改，
数据版本用文件 sha256 标识（见 :func:`dataset_sha256`）。四套文本套件是
openwsv-1 的**增量**套件（不改动已有 spatialmqa / pairs 数据），其图像位于
``openwsv-data-1.1`` 或复用 ``openwsv-data-1.0`` 的 pairs 帧。

``load_suite(name)`` 返回规范化后的 ``list[dict]``，每个 dict 固定字段：

``{task_id, suite, question, images, choices, answer, answers, meta}``

* 文本多选：``answer`` 是 0-based choice 下标，``choices`` 是选项文本；
* ``pairs``：``answer=None``、``choices=[]``，真值放在 ``meta``（entities /
  changes / relations / before / after），``question`` 是给站点展示的固定描述。

本模块只读文件、不做任何网络访问。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "SUITES",
    "SUITE_SIZES",
    "SUITE_FILES",
    "DERIVED_TEXT_SUITES",
    "SYNTHETIC_TEXT_SUITES",
    "RELATION_PATTERN",
    "SUITE_DIR",
    "load_suite",
    "suite_path",
    "dataset_sha256",
    "describe_suites",
]

SUITE_DIR = Path(__file__).resolve().parent / "suites"

#: suite 名 -> 任务文件名（关系子集与 spatialmqa 共用同一份冻结文件）
SUITE_FILES: Mapping[str, str] = {
    "spatialmqa": "spatialmqa.jsonl",
    "spatialmqa_relation": "spatialmqa.jsonl",
    "pairs": "pairs_test.jsonl",
    "change": "change_test.jsonl",
    "relation_change": "relation_change_test.jsonl",
    "state_replay": "state_replay_test.jsonl",
    "cross_frame": "cross_frame_test.jsonl",
}

#: 冻结样本量（与发布数据版本绑定；加载时校验，防静默漂移）
SUITE_SIZES: Mapping[str, int] = {
    "spatialmqa": 1612,
    "spatialmqa_relation": 305,
    "pairs": 1000,
    "change": 1000,
    "relation_change": 600,
    "state_replay": 1000,
    "cross_frame": 400,
}

SUITES: tuple[str, ...] = (
    "spatialmqa",
    "spatialmqa_relation",
    "pairs",
    "change",
    "relation_change",
    "state_replay",
    "cross_frame",
)

#: 由 pairs GT 派生的文本套件（直接按行加载，不做额外过滤）
DERIVED_TEXT_SUITES: tuple[str, ...] = ("change", "state_replay")

#: 合成文本套件（openwsv-data-1.1；直接按行加载）
SYNTHETIC_TEXT_SUITES: tuple[str, ...] = ("relation_change", "cross_frame")

#: T1-relation 子集定义（冻结）：SpatialMQA 中形如
#: "Where is/are the X located relative to the Y?" 的 305 题。
RELATION_PATTERN = re.compile(
    r"^Where (?:is|are) the (?P<x>.+?) located relative to the (?P<y>.+?)\?$"
)

PAIRS_QUESTION = (
    "What changed between the before and the after frame, and how are the objects related?"
)

TASK_FIELDS: tuple[str, ...] = (
    "task_id",
    "suite",
    "question",
    "images",
    "choices",
    "answer",
    "answers",
    "meta",
)


# --------------------------------------------------------------------- helpers


def suite_path(name: str) -> Path:
    """返回 suite 对应的冻结 jsonl 路径（未知 suite 抛 ``ValueError``）。"""
    if name not in SUITE_FILES:
        raise ValueError(
            f"未知 suite {name!r}；可用: {', '.join(SUITES)}"
        )
    return SUITE_DIR / SUITE_FILES[name]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少冻结任务文件：{path}（OpenWSV 包需要 suites/ 与包一起分发）"
        )
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - 数据损坏时给出定位
            raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno} 期望 JSON 对象，实际 {type(row).__name__}")
        rows.append(row)
    return rows


def _text_task(row: Mapping[str, Any], suite: str, match: re.Match[str] | None = None) -> dict[str, Any]:
    question = str(row.get("question", ""))
    raw_choices = row.get("choices") or ()
    choices = [str(c) for c in raw_choices]
    answer = row.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int):
        raise ValueError(f"task {row.get('task_id')!r}: answer 必须是 0-based 整数，实际 {answer!r}")
    if choices and not (0 <= answer < len(choices)):
        raise ValueError(
            f"task {row.get('task_id')!r}: answer={answer} 越界 (n_choices={len(choices)})"
        )
    meta: dict[str, Any] = dict(row.get("meta") or {})
    meta.update(
        {
            "dataset": str(row.get("dataset", "spatialmqa")),
            "capability": str(row.get("capability", "")),
            "split": str(row.get("split", "test")),
            "answer_type": str(row.get("answer_type", "choice")),
        }
    )
    if match is not None:
        meta["relation_x"] = match.group("x")
        meta["relation_y"] = match.group("y")
    return {
        "task_id": str(row["task_id"]),
        "suite": suite,
        "question": question,
        "images": [str(p) for p in (row.get("images") or ())],
        "choices": choices,
        "answer": int(answer),
        "answers": [str(a) for a in (row.get("answers") or ())],
        "meta": meta,
    }


def _pairs_task(row: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = str(row["pair_id"])
    entities = []
    for pos, e in enumerate(row.get("entities") or ()):
        slot = int(e["slot"])
        if slot != pos:
            raise ValueError(f"pair {pair_id}: entities slot 不连续（期望 {pos}，实际 {slot}）")
        entities.append(
            {
                "slot": slot,
                "type_key": str(e["type_key"]),
                "bbox": [float(v) for v in e["bbox"]],
                "present_after": bool(e.get("present_after", True)),
            }
        )
    changes = {
        int(c["slot"]): str(c["type"]) for c in (row.get("changes") or ())
    }
    relations = []
    for rel in row.get("relations") or ():
        item = {
            "from_slot": int(rel["from_slot"]),
            "to_slot": int(rel["to_slot"]),
            "type": str(rel["type"]),
            "rel_pos": rel.get("rel_pos"),
            # 保留"GT 是否声明了 rel_pos 字段"的信息：pair_protocol 用字段是否存在
            # 决定 rel_pos 指标的分母，不能用 rel_pos is not None 代替。
            "rel_pos_present": "rel_pos" in rel,
            "contact": bool(rel.get("contact", False)),
            "support": bool(rel.get("support", False)),
            "dist_m": rel.get("dist_m"),
        }
        relations.append(item)
    meta = {
        "pair_id": pair_id,
        "dataset": "dataset_spatial_v3",
        "split": str(row.get("split", "test")),
        "before": str(row["before"]),
        "after": str(row["after"]),
        "entities": entities,
        "changes": changes,
        "relations": relations,
        "negative": bool(row.get("negative", False)),
        "raw_meta": dict(row.get("meta") or {}),
    }
    return {
        "task_id": f"pairs-{pair_id}",
        "suite": "pairs",
        "question": PAIRS_QUESTION,
        "images": [str(row["before"]), str(row["after"])],
        "choices": [],
        "answer": None,
        "answers": [],
        "meta": meta,
    }


def _check_unique(rows: Iterable[Mapping[str, Any]], where: str) -> None:
    seen: set[str] = set()
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in seen:
            raise ValueError(f"{where}: 重复 task_id {task_id!r}")
        seen.add(task_id)


# ----------------------------------------------------------------- public API


def load_suite(name: str) -> list[dict[str, Any]]:
    """加载并规范化一套冻结 suite。

    Parameters
    ----------
    name:
        ``spatialmqa`` | ``spatialmqa_relation`` | ``pairs``。

    Returns
    -------
    list[dict]
        每个任务固定 ``{task_id, suite, question, images, choices, answer,
        answers, meta}``；顺序与冻结文件一致（确定性）。

    Raises
    ------
    ValueError
        未知 suite、任务数据不合法，或样本量与冻结版本不一致。
    FileNotFoundError
        ``suites/`` 下缺少冻结文件。
    """
    if name not in SUITE_FILES:
        raise ValueError(f"未知 suite {name!r}；可用: {', '.join(SUITES)}")
    path = suite_path(name)
    rows = _read_jsonl(path)

    if name == "pairs":
        tasks = [_pairs_task(row) for row in rows]
    elif name == "spatialmqa":
        tasks = [_text_task(row, "spatialmqa") for row in rows]
    elif name == "spatialmqa_relation":
        tasks = []
        for row in rows:
            match = RELATION_PATTERN.match(str(row.get("question", "")))
            if match is not None:
                tasks.append(_text_task(row, "spatialmqa_relation", match))
    else:  # change / relation_change / state_replay / cross_frame：行即规范化文本任务
        tasks = [_text_task(row, name) for row in rows]

    expected = SUITE_SIZES[name]
    if len(tasks) != expected:
        raise ValueError(
            f"suite {name!r} 样本量 {len(tasks)} != 冻结版本 {expected}；"
            "冻结数据被改动或版本不匹配（检查 suites/ 下文件 sha256）"
        )
    _check_unique(tasks, f"suite {name!r}")
    return tasks


def dataset_sha256(name: str) -> str:
    """返回 suite 冻结文件内容的 sha256（十六进制小写，数据版本标识）。"""
    path = suite_path(name)
    if not path.exists():
        raise FileNotFoundError(f"缺少冻结任务文件：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_suites() -> list[dict[str, Any]]:
    """``--list-suites`` 用的套件清单（size + sha256 + 定义）。"""
    descriptions = {
        "spatialmqa": "SpatialMQA single-image multiple choice (T1 spatial relations)",
        "spatialmqa_relation": "T1-relation subset: 'Where is/are the X located relative to the Y?'",
        "pairs": "dataset_spatial_v3 test before/after pairs (structured world model, T4-T6)",
        "change": "T4 spatial change text probes derived from the pairs GT (one per pair)",
        "relation_change": "T5 relation change, synthetic before/after pairs (openwsv-data-1.1)",
        "state_replay": "T6 after-state fact probes derived from the pairs GT (text projection)",
        "cross_frame": "T7 cross-frame disjoint spatial relations, synthetic pairs (openwsv-data-1.1)",
    }
    out = []
    for name in SUITES:
        out.append(
            {
                "suite": name,
                "n": SUITE_SIZES[name],
                "file": str(suite_path(name).relative_to(SUITE_DIR.parent)),
                "sha256": dataset_sha256(name),
                "description": descriptions[name],
            }
        )
    return out