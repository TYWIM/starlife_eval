"""综合能力分（composite score）：把「所有模型都能测」的 7 个维度等权平均成一个 0-100 的分。

为什么要单独一层，而不是直接改原来的 `score`：
  * `score`（= 事件级 StateChanged + RelationAdded 的 macroF1）是为**横向可比**设计的，
    只用两类事件、不给"变化识别/方位/交互属性/状态一致性"任何权重；
  * 用户看榜单需要一个"总分"，但**总分依赖权重**。所以这里的做法是：
      - 只取 6 个模型 100% 都能测的维度（避免用 N/A 维度给某些模型白送分/白扣分）；
      - **等权**（不预设"关系比变化更重要"），权重写进产物里，谁都能按自己场景重算；
      - 同时输出逐维分数，总分只是它们的均值，不是黑箱。

`state_consistency` 用 objects/changes/relations/attrs 四维的重放一致率均值，
**不含 types**——types 只有本地模型能测，混进去会让云端模型被扣一个自己根本没有的维度。

缺值处理：某维为 None 时**从分母里剔除**并记录到 `missing`，不会当 0 分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPOSITE_METHOD = "equal-weight"
COMPOSITE_FORMULA = "7 个维度（0-100）等权平均；缺项自动从分母剔除，不当 0 分"
STATE_DIMS_COMMON = ("objects", "changes", "relations", "attrs")  # types 仅本地可测，排除


@dataclass(frozen=True)
class CompositeDim:
    key: str
    group: str
    label: str
    source: str


COMPOSITE_DIMS: tuple[CompositeDim, ...] = (
    CompositeDim("change_top1", "变化识别", "变化识别 top1", "pair_protocol.change_top1"),
    CompositeDim("change_balance", "变化平衡性", "变化平衡性 (macroF1)", "pair_protocol.change_macro_f1"),
    CompositeDim("relation_type", "关系类型", "关系类型准确率", "pair_protocol.relation_type_acc"),
    CompositeDim("rel_pos", "空间方位", "空间方位准确率", "pair_protocol.rel_pos_acc"),
    CompositeDim("attrs", "接触支撑", "接触/支撑准确率", "pair_protocol.attrs_acc"),
    CompositeDim("event_f1", "事件级 F1", "事件级 F1", "event_metrics.macro_f1_common"),
    CompositeDim("state_consistency", "世界状态一致性", "世界状态一致性", "state_replay.state_acc_common"),
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None


def state_acc_common(state_replay: dict[str, Any] | None) -> float | None:
    """四维（objects/changes/relations/attrs）一致率均值；全缺返回 None。"""
    sr = state_replay or {}
    vals = [_num(sr.get(k)) for k in STATE_DIMS_COMMON]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def composite_parts(model: dict[str, Any]) -> dict[str, float | None]:
    """从单个模型的结果字典里取 7 个维度的原始值（0-1 比例）。"""
    pair = model.get("pair_protocol") or {}
    ev = model.get("event_metrics") or {}
    return {
        "change_top1": _num(pair.get("change_top1")),
        "change_balance": _num(pair.get("change_macro_f1")),
        "relation_type": _num(pair.get("relation_type_acc")),
        "rel_pos": _num(pair.get("rel_pos_acc")),
        "attrs": _num(pair.get("attrs_acc")),
        "event_f1": _num(ev.get("macro_f1_common")),
        "state_consistency": state_acc_common(model.get("state_replay")),
    }


def composite_score(model: dict[str, Any]) -> dict[str, Any]:
    """返回 {score, available, total, missing[], dims[{key,group,label,value,source}]}。

    `value` 一律换成 0-100 便于展示；`score` 是可用维度的等权平均（0-100）。
    """
    parts = composite_parts(model)
    dims: list[dict[str, Any]] = []
    values: list[float] = []
    missing: list[str] = []
    for dim in COMPOSITE_DIMS:
        raw = parts.get(dim.key)
        if raw is None:
            missing.append(dim.key)
            dims.append({"key": dim.key, "group": dim.group, "label": dim.label,
                         "value": None, "source": dim.source, "available": False})
            continue
        pct = round(100.0 * raw, 1)
        values.append(pct)
        dims.append({"key": dim.key, "group": dim.group, "label": dim.label,
                     "value": pct, "source": dim.source, "available": True})
    score = round(sum(values) / len(values), 1) if values else None
    n = len(values)
    return {
        "score": score,
        "method": COMPOSITE_METHOD,
        "formula": COMPOSITE_FORMULA,
        "weights": [round(1.0 / n, 6)] * n if n else [],
        "available": n,
        "total": len(COMPOSITE_DIMS),
        "missing": missing,
        "dims": dims,
        "state_dims_used": list(STATE_DIMS_COMMON),
    }


def add_composite_score(report: dict[str, Any]) -> None:
    """给 report 的每个模型与 meta 追加综合分（幂等，可重复调用）。"""
    meta = report.setdefault("meta", {})
    meta["composite"] = {
        "name": "综合能力分",
        "method": COMPOSITE_METHOD,
        "formula": COMPOSITE_FORMULA,
        "n_dims": len(COMPOSITE_DIMS),
        "dims": [
            {"key": d.key, "group": d.group, "label": d.label, "source": d.source}
            for d in COMPOSITE_DIMS
        ],
        "note": "总分依赖权重；这里用等权并公开逐维分数，任何人都能按自己的场景重算。",
    }
    for m in (report.get("models") or {}).values():
        comp = composite_score(m)
        m["composite_score"] = comp["score"]
        m["composite"] = comp
