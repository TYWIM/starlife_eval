"""WSV 结构化响应 -> :class:`PairPrediction`，以及 pairs 套件的指标聚合。

**方向约定（与 ``docs/RESULTS_STRUCTURED_TRACK.md`` 一致）**

WSV 服务把关系对按 ``combinations(slots, 2)`` 枚举，即只输出"检测列表顺序"下的
升序对。因此每对帧要跑两次：``forward``（manifest slot 升序）和 ``reversed``
（检测列表逆序）。对每条 GT 关系，选**与 GT 方向一致**的那次响应作为预测：

* ``from_slot < to_slot`` -> 用 forward 响应里的 ``(from, to)``；
* ``from_slot > to_slot`` -> 用 reversed 响应里的 ``(from, to)``。

GT 方向只决定**输入顺序**，标签不进入模型；两次调用都是纯推断。

**指标复用**：change/relation/rel_pos/attrs 的定义与 60 对协议完全一致，直接调用
:func:`starlife_eval.verifiers.pair_protocol.evaluate_pair_protocol`（只读复用，
不重复实现）。本模块只负责：响应解析、方向匹配、manifest 重建、按 pair 的
bootstrap CI 与逐对明细。

指标（point + 95% CI，bootstrap 重采样单位 = pair）：

* ``change_top1``：所有 entity slot 的 change 逐槽 top-1；
* ``change_macro_f1``：``macro_f1_compare`` 口径（与云端协议一致）；
* ``relation_type_acc``：GT 关系上的关系类型命中率；
* ``rel_pos_acc``：关系类型命中且 GT 声明了 rel_pos 时的方位命中率；
* ``attrs_acc``：关系类型命中时 contact/support 两项的命中率。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import DatasetManifest, Entity, PairPrediction, PairSample, Relation, RelationPrediction
from ..verifiers.pair_protocol import evaluate_pair_protocol

__all__ = [
    "DETECTION_KEYS",
    "build_manifest",
    "pair_prediction",
    "evaluate_pairs",
    "slot_id",
]

#: detector.detect 返回 dict 里的非 phrase 元数据键
DETECTION_KEYS = ("scores", "meta")

#: 需要 CI 的指标（分母为 pair 的 bootstrap）
CI_METRICS: tuple[str, ...] = (
    "change_top1",
    "change_macro_f1",
    "relation_type_acc",
    "rel_pos_acc",
    "attrs_acc",
)


# --------------------------------------------------------------- response -> pred


def slot_id(value: Any) -> int:
    """把 ``"slot3"`` / ``"slot:slot3"`` / ``3`` / ``"3"`` 统一成整数 slot。"""
    if isinstance(value, bool):
        raise ValueError(f"非法 slot id: {value!r}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("slot:"):
        text = text[len("slot:"):]
    if text.startswith("slot"):
        text = text[len("slot"):]
    if text.startswith("_"):
        text = text[1:]
    return int(text)


def _raw_response(value: Any) -> Mapping[str, Any] | None:
    """兼容 adapter 结果 ``{"response":..., "error":...}`` 与裸响应。"""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    if "response" in value and isinstance(value.get("response"), Mapping):
        return value["response"]
    return value


def _response_slot_changes(response: Mapping[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    if not response:
        return out
    for item in response.get("slots") or ():
        if not isinstance(item, Mapping):
            continue
        out[slot_id(item.get("id"))] = str(item.get("change", "none"))
    return out


def _response_relations(response: Mapping[str, Any] | None) -> dict[tuple[int, int], Mapping[str, Any]]:
    out: dict[tuple[int, int], Mapping[str, Any]] = {}
    if not response:
        return out
    pairs = response.get("pairs") or ()
    if isinstance(pairs, Mapping):  # 容错：极少数服务把 pairs 返回成 dict
        pairs = list(pairs.values())
    for item in pairs:
        if not isinstance(item, Mapping):
            continue
        try:
            key = (slot_id(item.get("from")), slot_id(item.get("to")))
        except (TypeError, ValueError):
            continue
        out[key] = item
    return out


def _unpack(value: Any) -> Mapping[str, Any] | None:
    return _raw_response(value)


def pair_prediction(
    task: Mapping[str, Any],
    forward: Any,
    reversed_: Any,
) -> PairPrediction:
    """把 forward/reversed 两次 WSV 响应合成一个逐 pair 的 :class:`PairPrediction`。

    * change 取 forward 响应（两次输入只有检测排序不同，slot 变化应一致）；
      forward 不可用时退回 reversed；
    * 每条 GT 关系按 GT 方向从对应响应取候选（见模块 docstring）。
    """
    meta = task.get("meta") or {}
    fwd = _unpack(forward)
    rev = _unpack(reversed_)
    changes = _response_slot_changes(fwd) or _response_slot_changes(rev)
    fwd_rel = _response_relations(fwd)
    rev_rel = _response_relations(rev)

    relations: list[RelationPrediction] = []
    for gt in meta.get("relations") or ():
        frm, to = int(gt["from_slot"]), int(gt["to_slot"])
        key = (frm, to)
        # 方向匹配：GT from<to 用 forward；否则用 reversed。兜底与
        # scripts/structured_track/run_structured_track.py 的 score_pair 一致：
        # 再试同一方向 key 与"排序后 key"在 reversed 响应里的条目。
        cand = fwd_rel.get(key) if frm < to else rev_rel.get(key)
        if cand is None:
            cand = fwd_rel.get(key) or rev_rel.get(tuple(sorted(key)))
        if cand is None:
            continue  # 缺预测 -> evaluate_pair_protocol 记为该关系类型不命中
        relations.append(
            RelationPrediction(
                from_slot=frm,
                to_slot=to,
                type=str(cand.get("relation", "none")),
                rel_pos=None if cand.get("rel_pos") is None else str(cand.get("rel_pos")),
                contact=bool(cand.get("contact", False)),
                support=bool(cand.get("support", False)),
            )
        )

    # evaluate_pair_protocol 需要 pair_index 与 sample_indices 对齐；这里用任务在
    # 本次评测中的序号（稳定），由 evaluate_pairs 在调用前重写。
    return PairPrediction(pair_index=int(meta.get("_index", 0)), changes=changes, relations=relations)


# ------------------------------------------------------------------- manifest


def build_manifest(tasks: Sequence[Mapping[str, Any]], *, root: str | Path | None = None) -> DatasetManifest:
    """由 pairs 套件任务重建 :class:`DatasetManifest`（供 protocol 指标复用）。

    只重构 GT 字段（entities/changes/relations/before/after）；预测仍来自模型。
    ``pair_index`` = 任务在 ``tasks`` 中的序号（0..n-1），与 ``sample_indices`` 一致。
    """
    pairs: list[PairSample] = []
    offsets: list[int] = [0]
    for index, task in enumerate(tasks):
        meta = task.get("meta") or {}
        entities = tuple(
            Entity(
                slot=int(e["slot"]),
                bbox=tuple(float(v) for v in e["bbox"]),  # type: ignore[arg-type]
                type_key=str(e["type_key"]),
                present_after=bool(e.get("present_after", True)),
            )
            for e in meta.get("entities") or ()
        )
        changes = {int(k): str(v) for k, v in (meta.get("changes") or {}).items()}
        relations: list[Relation] = []
        # raw 保留字段存在性，供 pair_protocol._gt_has_rel_pos 使用
        raw_relations: list[dict[str, Any]] = []
        for rel in meta.get("relations") or ():
            relations.append(
                Relation(
                    from_slot=int(rel["from_slot"]),
                    to_slot=int(rel["to_slot"]),
                    type=str(rel["type"]),
                    rel_pos=None if rel.get("rel_pos") is None else str(rel["rel_pos"]),
                    contact=bool(rel.get("contact", False)),
                    support=bool(rel.get("support", False)),
                    dist_m=None if rel.get("dist_m") is None else float(rel["dist_m"]),
                )
            )
            raw_rel: dict[str, Any] = {"from_slot": rel["from_slot"], "to_slot": rel["to_slot"]}
            if rel.get("rel_pos_present", rel.get("rel_pos") is not None):
                raw_rel["rel_pos"] = rel.get("rel_pos")
            raw_relations.append(raw_rel)
        pairs.append(
            PairSample(
                index=index,
                pair_id=str(meta.get("pair_id", task.get("task_id", index))),
                split=str(meta.get("split", "test")),
                before=str(meta.get("before", "")),
                after=str(meta.get("after", "")),
                entities=entities,
                changes=changes,
                relations=tuple(relations),
                raw={"relations": raw_relations},
            )
        )
        offsets.append(offsets[-1] + len(entities))
    return DatasetManifest(
        root=Path(root) if root is not None else Path("."),
        pairs=tuple(pairs),
        meta={"source": "openwsv/pairs"},
        slot_offsets=tuple(offsets),
    )


# ------------------------------------------------------------------- evaluation


def _empty_metrics() -> dict[str, Any]:
    return {
        "change_slots": 0,
        "change_top1": 0.0,
        "change_macro_f1": 0.0,
        "relation_pairs": 0,
        "relation_type_acc": 0.0,
        "rel_pos_acc": 0.0,
        "attrs_acc": 0.0,
    }


def _pair_detail(
    task: Mapping[str, Any],
    prediction: PairPrediction,
) -> dict[str, Any]:
    """单对明细（用于 per_task / 站点下钻）：change 与 relation 计数。"""
    meta = task.get("meta") or {}
    gt_changes = {int(k): str(v) for k, v in (meta.get("changes") or {}).items()}
    change_total = len(meta.get("entities") or ())
    change_hit = 0
    for entity in meta.get("entities") or ():
        slot = int(entity["slot"])
        gold = gt_changes.get(slot, "none")
        pred = prediction.changes.get(slot, "none")
        change_hit += int(pred == gold)
    rel_map = {(r.from_slot, r.to_slot): r for r in prediction.relations}
    rel_total = len(meta.get("relations") or ())
    rel_hit = sum(
        1
        for gt in meta.get("relations") or ()
        if (cand := rel_map.get((int(gt["from_slot"]), int(gt["to_slot"])))) is not None
        and cand.type == str(gt["type"])
    )
    return {
        "change_correct": change_hit,
        "change_slots": change_total,
        "relation_type_correct": rel_hit,
        "relation_pairs": rel_total,
        "change_top1": (change_hit / change_total) if change_total else 0.0,
    }


def evaluate_pairs(
    tasks: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 20261022,
) -> dict[str, Any]:
    """在 pairs 套件上算结构化指标 + bootstrap CI + 逐对明细。

    Parameters
    ----------
    tasks:
        :func:`starlife_eval.wsv.tasks.load_suite` 的 ``pairs`` 任务列表。
    responses:
        ``{task_id: {"forward": <adapter 结果或裸响应>,
        "reversed": <...>, "error": str | None}}``；只有两次响应都成功
        且无 ``error`` 的 pair 才计入覆盖（分母）。
    n_boot, seed:
        bootstrap 次数与种子（框架冻结 ``bootstrap_seed``）。

    Returns
    -------
    dict
        ``{n_pairs, n_covered, coverage, metrics, ci95, n_boot, seed, per_pair}``。

        ``metrics`` 为 :func:`evaluate_pair_protocol` 的 point 估计（缺预测时按
        空预测计）；覆盖 pair 为 0 时指标全 0 且 ``ci95`` 全 ``None``。
    """
    manifest = build_manifest(tasks)
    covered_indices: list[int] = []
    preds: dict[int, PairPrediction] = {}
    per_pair: list[dict[str, Any]] = []

    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        entry = responses.get(task_id)
        forward = None
        reversed_ = None
        error: str | None = None
        if entry is None:
            error = "missing prediction"
        else:
            error = entry.get("error") if isinstance(entry, Mapping) else None
            forward = entry.get("forward") if isinstance(entry, Mapping) else None
            reversed_ = entry.get("reversed") if isinstance(entry, Mapping) else None
        ok = (
            not error
            and _unpack(forward) is not None
            and _unpack(reversed_) is not None
        )
        if ok:
            pred = pair_prediction(task, forward, reversed_)
            pred.pair_index = index
            preds[index] = pred
            covered_indices.append(index)
        detail = (
            _pair_detail(task, preds[index])
            if ok
            else {"change_correct": 0, "change_slots": 0,
                  "relation_type_correct": 0, "relation_pairs": 0, "change_top1": 0.0}
        )
        per_pair.append(
            {
                "task_id": task_id,
                "covered": ok,
                "error": None if ok else (error or "incomplete forward/reversed response"),
                **detail,
            }
        )

    n_pairs = len(tasks)
    n_covered = len(covered_indices)
    if n_covered:
        metrics = evaluate_pair_protocol(manifest, covered_indices, preds, strict=False)
    else:
        metrics = _empty_metrics()

    ci95: dict[str, list[float] | None] = {name: None for name in CI_METRICS}
    if n_covered:
        rng = np.random.default_rng(int(seed))
        draws: dict[str, list[float]] = {name: [] for name in CI_METRICS}
        idx = np.asarray(covered_indices, dtype=np.int64)
        for _ in range(int(n_boot)):
            picked = rng.integers(0, idx.size, size=idx.size)
            sample = [int(v) for v in idx[picked]]
            boot = evaluate_pair_protocol(manifest, sample, preds, strict=False)
            for name in CI_METRICS:
                value = boot.get(name)
                if value is not None and math.isfinite(float(value)):
                    draws[name].append(float(value))
        for name, values in draws.items():
            if values:
                ci95[name] = [
                    round(float(np.quantile(values, 0.025)), 4),
                    round(float(np.quantile(values, 0.975)), 4),
                ]

    return {
        "n_pairs": n_pairs,
        "n_covered": n_covered,
        "coverage": (n_covered / n_pairs) if n_pairs else 0.0,
        "metrics": metrics,
        "ci95": ci95,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "per_pair": per_pair,
    }