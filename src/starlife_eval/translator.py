"""L1 事件契约翻译器：预测/GT 的 pair 级判别结果 → `WorldEvent` 四类事件流。

口径来源（只读参考，不在运行时依赖）：
- `starlife/starlife_store/Core/WorldEvent.cs`：四类事件与 payload 形状
  （EntityCreated{entity_id,type} / StateChanged{entity_id,valid_from,properties}
  / RelationAdded{from,to,relation_type,valid_from,properties?}
  / RelationRemoved{from,to,relation_type,valid_at}），时间戳 Unix ms、append-only；
- `starlife/starlife_webAPI/Vision/ObservationTranslator.cs`：
  VLM 观察实体 ID 以 `1_000_000` 为基准递增、同一 (picture, id) 稳定复用、
  同帧关系去重、上一快照有而新快照没有的关系发 RelationRemoved；
- `starlife_eval/ARCHITECTURE.md` §5.3（L3）与 `测试套件设计.md` L1。

本模块只依赖 `starlife_eval.types` 与标准库，**不依赖 torch / dataset / models**，
也不 import 任何 .NET 代码，仅用 Python 最小实现对齐语义。

设计约定（与 L3/L4 指标共享）：
- `entity_id` 从 `1_000_000` 起，按首次出现的 (pair_index, slot) 递增分配；
  同一 (pair_index, slot) 永远复用同一 ID，不同 pair 的 ID 不混用；
- 事件额外带 `pair_index` / `slot` / `from_slot` / `to_slot` 元数据（不进 payload），
  供事件匹配与状态回放按 **per-pair slot 编号** 对齐，不依赖 ID 分配顺序；
- 一个 pair 的 before/after 两帧视为同一帧时刻，pair 之间时间戳按 step 递增，
  时间戳保证 Unix ms 单调不减；同帧重复关系去重。
- **v2 数据集只有 before/after 两帧、只标注 after 帧关系**：RelationRemoved 只在
  调用方提供/翻译器记住“前一次快照关系”时才可能产生；v2 单帧对默认不产生
  RelationRemoved（指标侧据此标 N/A，见 `verifiers/event_metrics.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .types import CHANGE_LABELS, DatasetManifest, PairPrediction, PairSample, RelationPrediction

# --------------------------------------------------------------------- constants

EVENT_KINDS: tuple[str, ...] = (
    "EntityCreated",
    "StateChanged",
    "RelationAdded",
    "RelationRemoved",
)

#: VLM 观察实体 ID 基准（WorldEvent.cs / ObservationTranslator.cs 的约定）。
ENTITY_ID_BASE: int = 1_000_000
#: 固定起点的 Unix ms，便于单测复现；真实运行可传 ts_ms。
DEFAULT_TS_START_MS: int = 1_700_000_000_000
#: 相邻 pair 的时间步长（ms）。v2 每个 pair 是独立场景帧对，该值只用于排序。
DEFAULT_TS_STEP_MS: int = 1_000

#: 非 none 的变化标签（StateChanged 才会有事件）。
_ACTIVE_CHANGES = tuple(c for c in CHANGE_LABELS if c != "none")

RELATION_REMOVED_NA_REASON: str = (
    "v2 数据集只标注 after 帧关系（manifest.relations），没有 before 关系标注、"
    "也不是连续帧序列，无法构造 RelationRemoved 的 GT；该事件类不可测，按 N/A 处理"
    "（不编造 0 分，也不计入 macro-F1）。"
)

ID_SWITCH_NA_REASON: str = (
    "v2 数据集每个 pair 只有 before/after 两帧、不是连续帧序列，无法跨帧追踪同一物体，"
    "id_switch_rate 不可测（L1 单测只验证同一 (pair, slot) 的 entity_id 稳定复用）；"
    "不以 0 分或编造值上报。"
)


# ------------------------------------------------------------------- data model


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class WorldEvent:
    """`WorldEvent.cs` 四类事件的 Python 最小实现（append-only 事件流中的一个事件）。

    payload 形状严格对齐（`to_payload()`）：
    - EntityCreated   -> {"entity_id": int, "type": str}
    - StateChanged    -> {"entity_id": int, "valid_from": int, "properties": {...}}
    - RelationAdded   -> {"from": int, "to": int, "relation_type": str,
                          "valid_from": int, "properties": {...}}（properties 可选但总是给出）
    - RelationRemoved -> {"from": int, "to": int, "relation_type": str, "valid_at": int}

    `pair_index` / `slot` / `from_slot` / `to_slot` 是评测侧元数据（不进 payload），
    让事件匹配按 per-pair slot 编号对齐。
    """

    kind: str
    ts_ms: int
    entity_id: int | None = None
    entity_type: str | None = None
    pair_index: int | None = None
    slot: int | None = None
    from_id: int | None = None
    to_id: int | None = None
    from_slot: int | None = None
    to_slot: int | None = None
    relation_type: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    # -- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"非法事件类型 {self.kind!r}，只允许 {EVENT_KINDS}")
        if not _is_int(self.ts_ms) or self.ts_ms < 0:
            raise ValueError(f"ts_ms 必须是非负整数 Unix ms，实际 {self.ts_ms!r}")
        if self.kind == "EntityCreated":
            if not _is_int(self.entity_id):
                raise ValueError("EntityCreated 需要整数 entity_id")
            if not isinstance(self.entity_type, str) or not self.entity_type:
                raise ValueError("EntityCreated 需要非空字符串 type（无类型时不应发该事件）")
        elif self.kind == "StateChanged":
            if not _is_int(self.entity_id):
                raise ValueError("StateChanged 需要整数 entity_id")
            if not isinstance(self.properties, Mapping):
                raise ValueError("StateChanged.properties 必须是 mapping")
            present = self.properties.get("present")
            if not isinstance(present, bool):
                raise ValueError("StateChanged.properties 必须包含 bool 字段 present")
            change = self.properties.get("change")
            if change not in _ACTIVE_CHANGES:
                raise ValueError(
                    f"StateChanged 的 change 必须是 {_ACTIVE_CHANGES} 之一，实际 {change!r}"
                    "（none 不发 StateChanged）"
                )
        elif self.kind in ("RelationAdded", "RelationRemoved"):
            for name, value in (("from", self.from_id), ("to", self.to_id)):
                if not _is_int(value):
                    raise ValueError(f"{self.kind} 需要整数 {name}")
            if not isinstance(self.relation_type, str) or not self.relation_type:
                raise ValueError(f"{self.kind} 需要非空 relation_type")
            if not isinstance(self.properties, Mapping):
                raise ValueError(f"{self.kind}.properties 必须是 mapping")

    # -- serialization ---------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        """返回与 `WorldEvent.cs` 注释一致的 payload（不含评测元数据）。"""
        if self.kind == "EntityCreated":
            return {"entity_id": self.entity_id, "type": self.entity_type}
        if self.kind == "StateChanged":
            return {
                "entity_id": self.entity_id,
                "valid_from": self.ts_ms,
                "properties": dict(self.properties),
            }
        if self.kind == "RelationAdded":
            return {
                "from": self.from_id,
                "to": self.to_id,
                "relation_type": self.relation_type,
                "valid_from": self.ts_ms,
                "properties": dict(self.properties),
            }
        # RelationRemoved
        return {
            "from": self.from_id,
            "to": self.to_id,
            "relation_type": self.relation_type,
            "valid_at": self.ts_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        """报告/日志用的完整事件（含评测元数据）。"""
        return {
            "type": self.kind,
            "entity_id": self.entity_id,
            "recorded_at": self.ts_ms,
            "payload": self.to_payload(),
            "pair_index": self.pair_index,
            "slot": self.slot,
            "from_slot": self.from_slot,
            "to_slot": self.to_slot,
        }

    def __hash__(self) -> int:  # properties 不参与 hash，事件标注性字段参与
        return hash(
            (
                self.kind,
                self.ts_ms,
                self.entity_id,
                self.entity_type,
                self.pair_index,
                self.slot,
                self.from_id,
                self.to_id,
                self.from_slot,
                self.to_slot,
                self.relation_type,
            )
        )


# -- factories ----------------------------------------------------------------
# 用构造函数集中校验字段，避免调用方直接拼错 payload 形状。


def entity_created(
    entity_id: int,
    entity_type: str,
    ts_ms: int,
    *,
    pair_index: int | None = None,
    slot: int | None = None,
) -> WorldEvent:
    return WorldEvent(
        kind="EntityCreated",
        ts_ms=ts_ms,
        entity_id=entity_id,
        entity_type=entity_type,
        pair_index=pair_index,
        slot=slot,
    )


def state_changed(
    entity_id: int,
    change_type: str,
    ts_ms: int,
    *,
    present: bool,
    pair_index: int | None = None,
    slot: int | None = None,
) -> WorldEvent:
    return WorldEvent(
        kind="StateChanged",
        ts_ms=ts_ms,
        entity_id=entity_id,
        pair_index=pair_index,
        slot=slot,
        properties={"present": bool(present), "change": change_type},
    )


def relation_added(
    from_id: int,
    to_id: int,
    relation_type: str,
    ts_ms: int,
    *,
    properties: Mapping[str, Any] | None = None,
    pair_index: int | None = None,
    from_slot: int | None = None,
    to_slot: int | None = None,
) -> WorldEvent:
    return WorldEvent(
        kind="RelationAdded",
        ts_ms=ts_ms,
        from_id=from_id,
        to_id=to_id,
        relation_type=relation_type,
        pair_index=pair_index,
        from_slot=from_slot,
        to_slot=to_slot,
        properties=dict(properties or {}),
    )


def relation_removed(
    from_id: int,
    to_id: int,
    relation_type: str,
    ts_ms: int,
    *,
    pair_index: int | None = None,
    from_slot: int | None = None,
    to_slot: int | None = None,
) -> WorldEvent:
    return WorldEvent(
        kind="RelationRemoved",
        ts_ms=ts_ms,
        from_id=from_id,
        to_id=to_id,
        relation_type=relation_type,
        pair_index=pair_index,
        from_slot=from_slot,
        to_slot=to_slot,
    )


# --------------------------------------------------------------------- helpers


def as_prediction_map(
    preds: Mapping[int, PairPrediction] | Iterable[PairPrediction],
) -> dict[int, PairPrediction]:
    """把 `dict[pair_index, PairPrediction]` 或可迭代预测统一成 dict。"""
    if isinstance(preds, Mapping):
        out = {}
        for key, value in preds.items():
            if not isinstance(value, PairPrediction):
                raise TypeError(f"preds[{key!r}] 不是 PairPrediction")
            if value.pair_index != key:
                raise ValueError(
                    f"preds 键 {key!r} 与 PairPrediction.pair_index={value.pair_index} 不一致"
                )
            out[int(key)] = value
        return out
    out = {}
    for value in preds:
        if not isinstance(value, PairPrediction):
            raise TypeError("preds 元素不是 PairPrediction")
        out[int(value.pair_index)] = value
    return out


def select_pairs(
    manifest: DatasetManifest | Sequence[PairSample],
    *,
    split: str | None = None,
    sample_indices: Iterable[int] | None = None,
) -> list[PairSample]:
    """按 split / 抽样 pair 下标筛出 pair，按 manifest 行号升序返回。

    兼容 `DatasetManifest`（`manifest.pairs`）与直接的 `Sequence[PairSample]`，
    方便 runner 用真实的 manifest、测试用玩具 manifest。
    """
    if hasattr(manifest, "pairs"):
        pairs = list(manifest.pairs)  # type: ignore[arg-type]
    else:
        pairs = list(manifest)  # type: ignore[arg-type]
    if sample_indices is not None:
        wanted = {int(i) for i in sample_indices}
        pairs = [p for p in pairs if p.index in wanted]
    if split is not None:
        pairs = [p for p in pairs if p.split == split]
    return sorted(pairs, key=lambda p: p.index)


def validate_event_stream(events: Sequence[WorldEvent]) -> None:
    """校验事件流：schema 合法、时间戳 Unix ms 单调不减、同帧不重复。

    逐事件调用 `WorldEvent.__post_init__` 已保证 schema；此函数额外检查流级约束。
    失败抛 `ValueError`（不静默吞掉）。
    """
    last_ts: int | None = None
    seen: set[tuple[Any, ...]] = set()
    for i, ev in enumerate(events):
        if not isinstance(ev, WorldEvent):
            raise ValueError(f"events[{i}] 不是 WorldEvent")
        if last_ts is not None and ev.ts_ms < last_ts:
            raise ValueError(
                f"events[{i}] 时间戳回退：{ev.ts_ms} < {last_ts}（要求 Unix ms 单调不减）"
            )
        last_ts = ev.ts_ms
        if ev.kind == "StateChanged":
            key = (ev.ts_ms, "StateChanged", ev.pair_index, ev.slot)
        elif ev.kind == "EntityCreated":
            key = (ev.ts_ms, "EntityCreated", ev.pair_index, ev.slot)
        else:
            key = (
                ev.ts_ms,
                ev.kind,
                ev.pair_index,
                ev.from_slot,
                ev.to_slot,
                ev.relation_type,
            )
        if key in seen:
            raise ValueError(f"events[{i}] 同帧重复事件 {key!r}")
        seen.add(key)


def events_to_dicts(events: Sequence[WorldEvent]) -> list[dict[str, Any]]:
    return [ev.to_dict() for ev in events]


def _normalize_relation_set(
    relations: Iterable[tuple[int, int, str]] | Iterable[RelationPrediction],
) -> frozenset[tuple[int, int, str]]:
    """把关系集合（元组或 RelationPrediction）规范化成 (from_slot, to_slot, type)。"""
    out: set[tuple[int, int, str]] = set()
    for rel in relations:
        if isinstance(rel, RelationPrediction):
            key = (int(rel.from_slot), int(rel.to_slot), str(rel.type))
        else:
            try:
                frm, to, rtype = rel  # type: ignore[misc]
            except (TypeError, ValueError) as exc:  # noqa: BLE001
                raise TypeError(f"无法解析关系 {rel!r}") from exc
            key = (int(frm), int(to), str(rtype))
        if key[2] and key[2] != "none":
            out.add(key)
    return frozenset(out)


def _relation_properties(rel: RelationPrediction) -> dict[str, Any]:
    """RelationAdded.properties：只放真实给出值的属性，不编造默认值。"""
    props: dict[str, Any] = {}
    if rel.rel_pos is not None:
        props["rel_pos"] = rel.rel_pos
    if rel.contact is not None:
        props["contact"] = bool(rel.contact)
    if rel.support is not None:
        props["support"] = bool(rel.support)
    return props


# ------------------------------------------------------------------- translator


class EventTranslator:
    """把 pair 级预测翻译成 append-only `WorldEvent` 事件流。

    用法::

        tr = EventTranslator()
        events = tr.translate(preds_by_pair_index)           # 全部 pair
        events = tr.translate(preds, pair_indices=[0, 3])     # 只翻抽样子集
        events = tr.translate_pair(pred, ts_ms=...)           # 单 pair

    ID 分配：`entity_id(pair_index, slot)` 首次调用时从 `entity_id_base`（默认 1_000_000）
    递增分配，之后稳定复用；不同 pair 之间不混用。
    """

    def __init__(
        self,
        *,
        entity_id_base: int = ENTITY_ID_BASE,
        ts_start_ms: int = DEFAULT_TS_START_MS,
        ts_step_ms: int = DEFAULT_TS_STEP_MS,
    ) -> None:
        if not _is_int(entity_id_base) or entity_id_base < 0:
            raise ValueError("entity_id_base 必须是非负整数")
        if not _is_int(ts_step_ms) or ts_step_ms <= 0:
            raise ValueError("ts_step_ms 必须是正整数")
        self.entity_id_base = int(entity_id_base)
        self.ts_start_ms = int(ts_start_ms)
        self.ts_step_ms = int(ts_step_ms)
        self._next_id = int(entity_id_base)
        self._id_map: dict[tuple[int, int], int] = {}
        self._slot_of_id: dict[int, tuple[int, int]] = {}
        self._last_relations: dict[int, frozenset[tuple[int, int, str]]] = {}
        self._last_ts: int | None = None

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        self._next_id = self.entity_id_base
        self._id_map.clear()
        self._slot_of_id.clear()
        self._last_relations.clear()
        self._last_ts = None

    @property
    def id_map(self) -> dict[tuple[int, int], int]:
        """(pair_index, slot) -> entity_id 的只读快照。"""
        return dict(self._id_map)

    def entity_id(self, pair_index: int, slot: int) -> int:
        """同一 (pair_index, slot) 稳定复用；首次出现时递增分配。"""
        key = (int(pair_index), int(slot))
        eid = self._id_map.get(key)
        if eid is None:
            eid = self._next_id
            self._next_id += 1
            self._id_map[key] = eid
            self._slot_of_id[eid] = key
        return eid

    def entity_slot(self, entity_id: int) -> tuple[int, int]:
        """entity_id -> (pair_index, slot) 的反向映射。"""
        try:
            return self._slot_of_id[int(entity_id)]
        except KeyError as exc:
            raise KeyError(f"未分配的 entity_id: {entity_id!r}") from exc

    # -- timestamps ------------------------------------------------------
    def _next_ts(self) -> int:
        if self._last_ts is None:
            ts = self.ts_start_ms
        else:
            ts = self._last_ts + self.ts_step_ms
        self._last_ts = ts
        return ts

    def _accept_ts(self, ts_ms: int) -> int:
        ts = int(ts_ms)
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"时间戳回退：{ts} < 上一次 {self._last_ts}（要求 Unix ms 单调不减）"
            )
        self._last_ts = ts
        return ts

    # -- pair -> events --------------------------------------------------
    def translate_pair(
        self,
        pred: PairPrediction,
        *,
        pair_index: int | None = None,
        ts_ms: int | None = None,
        previous_relations: (
            Iterable[tuple[int, int, str]] | Iterable[RelationPrediction] | None
        ) = None,
    ) -> list[WorldEvent]:
        """把一个 pair 的预测翻译成事件流（同帧去重；时间戳为该 pair 的帧时刻）。

        Args:
            pred: 逐 pair 预测（slot 编号）。
            pair_index: 显式 pair 下标；缺省用 `pred.pair_index`。
            ts_ms: 该 pair 的帧时刻（Unix ms）；缺省由翻译器递增分配。
            previous_relations: 该 pair“前一次快照”的关系集合。给出时会比较出
                RelationRemoved；缺省时若翻译器此前翻译过同一 pair 则复用其上次快照。
                v2 单帧对没有前序快照，因此不会产生 RelationRemoved（除非显式传入）。
        """
        pid = int(pred.pair_index if pair_index is None else pair_index)
        ts = self._next_ts() if ts_ms is None else self._accept_ts(ts_ms)
        events: list[WorldEvent] = []

        # 1) 逐 slot 变化：appeared -> EntityCreated + StateChanged；其余 -> StateChanged
        for raw_slot in sorted(pred.changes, key=lambda s: int(s)):
            slot = int(raw_slot)
            change = str(pred.changes[raw_slot])
            if change == "none":
                continue
            if change not in _ACTIVE_CHANGES:
                raise ValueError(f"pair {pid} slot {slot} 非法 change 标签 {change!r}")
            eid = self.entity_id(pid, slot)
            if change == "appeared":
                etype = pred.types.get(slot)
                if etype is not None and str(etype):
                    events.append(
                        entity_created(eid, str(etype), ts, pair_index=pid, slot=slot)
                    )
                events.append(
                    state_changed(
                        eid, change, ts, present=True, pair_index=pid, slot=slot
                    )
                )
            elif change == "displaced":
                events.append(
                    state_changed(
                        eid, change, ts, present=True, pair_index=pid, slot=slot
                    )
                )
            else:  # disappeared
                events.append(
                    state_changed(
                        eid, change, ts, present=False, pair_index=pid, slot=slot
                    )
                )

        # 2) 关系：同帧去重（同 (from,to,type) 只发一次），再与上一快照 diff
        current: list[tuple[int, int, str, RelationPrediction]] = []
        seen: set[tuple[int, int, str]] = set()
        for rel in pred.relations:
            key = (int(rel.from_slot), int(rel.to_slot), str(rel.type))
            if not key[2] or key[2] == "none" or key in seen:
                continue
            seen.add(key)
            current.append((key[0], key[1], key[2], rel))

        for from_slot, to_slot, relation_type, rel in current:
            events.append(
                relation_added(
                    self.entity_id(pid, from_slot),
                    self.entity_id(pid, to_slot),
                    relation_type,
                    ts,
                    properties=_relation_properties(rel),
                    pair_index=pid,
                    from_slot=from_slot,
                    to_slot=to_slot,
                )
            )

        if previous_relations is not None:
            prev = _normalize_relation_set(previous_relations)
        else:
            prev = self._last_relations.get(pid)
        if prev is not None:
            for from_slot, to_slot, relation_type in sorted(prev - frozenset(seen)):
                events.append(
                    relation_removed(
                        self.entity_id(pid, from_slot),
                        self.entity_id(pid, to_slot),
                        relation_type,
                        ts,
                        pair_index=pid,
                        from_slot=from_slot,
                        to_slot=to_slot,
                    )
                )
        self._last_relations[pid] = frozenset(seen)

        return events

    def translate(
        self,
        preds: Mapping[int, PairPrediction] | Iterable[PairPrediction],
        *,
        pair_indices: Iterable[int] | None = None,
    ) -> list[WorldEvent]:
        """翻译一组 pair（按 pair 下标升序），返回整条事件流并做流级校验。"""
        pred_map = as_prediction_map(preds)
        if pair_indices is None:
            order = sorted(pred_map)
        else:
            order = sorted({int(i) for i in pair_indices})
            missing = [i for i in order if i not in pred_map]
            if missing:
                raise ValueError(f"缺少 pair 预测：{missing}")
        events: list[WorldEvent] = []
        for pid in order:
            events.extend(self.translate_pair(pred_map[pid], pair_index=pid))
        validate_event_stream(events)
        return events

    # -- GT -> events ----------------------------------------------------
    def translate_gt_pair(
        self,
        pair: PairSample,
        *,
        ts_ms: int | None = None,
        previous_relations: (
            Iterable[tuple[int, int, str]] | Iterable[RelationPrediction] | None
        ) = None,
    ) -> list[WorldEvent]:
        """把 manifest 的一个 pair 翻译成 GT 事件流（规则与预测端完全对称）。

        规则：
        - `changes` 只列非 none；appeared -> EntityCreated + StateChanged{present:true}；
          displaced -> StateChanged{present:true}；disappeared -> StateChanged{present:false}；
        - 每条 after 帧 GT 关系 -> RelationAdded（payload 带 rel_pos/contact/support）；
        - 只有提供/记住前一次快照时才会产生 RelationRemoved（v2 无 before 标注 → 不产生）。
        """
        pid = int(pair.index)
        ts = self._next_ts() if ts_ms is None else self._accept_ts(ts_ms)
        events: list[WorldEvent] = []

        changes = {int(k): str(v) for k, v in pair.changes.items()}
        for slot in sorted(changes):
            change = changes[slot]
            if change == "none":
                continue
            if change not in _ACTIVE_CHANGES:
                raise ValueError(f"pair {pid} slot {slot} 非法 GT change 标签 {change!r}")
            entity = pair.entity(slot)
            eid = self.entity_id(pid, slot)
            if change == "appeared":
                events.append(
                    entity_created(
                        eid, entity.type_key, ts, pair_index=pid, slot=slot
                    )
                )
                events.append(
                    state_changed(
                        eid, change, ts, present=True, pair_index=pid, slot=slot
                    )
                )
            elif change == "displaced":
                events.append(
                    state_changed(
                        eid, change, ts, present=True, pair_index=pid, slot=slot
                    )
                )
            else:  # disappeared
                events.append(
                    state_changed(
                        eid, change, ts, present=False, pair_index=pid, slot=slot
                    )
                )

        current: list[tuple[int, int, str, dict[str, Any]]] = []
        seen: set[tuple[int, int, str]] = set()
        for rel in pair.relations:
            key = (int(rel.from_slot), int(rel.to_slot), str(rel.type))
            if not key[2] or key[2] == "none" or key in seen:
                continue
            seen.add(key)
            props: dict[str, Any] = {"contact": bool(rel.contact), "support": bool(rel.support)}
            if rel.rel_pos is not None:
                props["rel_pos"] = rel.rel_pos
            current.append((key[0], key[1], key[2], props))

        for from_slot, to_slot, relation_type, props in current:
            events.append(
                relation_added(
                    self.entity_id(pid, from_slot),
                    self.entity_id(pid, to_slot),
                    relation_type,
                    ts,
                    properties=props,
                    pair_index=pid,
                    from_slot=from_slot,
                    to_slot=to_slot,
                )
            )

        if previous_relations is not None:
            prev = _normalize_relation_set(previous_relations)
        else:
            prev = self._last_relations.get(pid)
        if prev is not None:
            for from_slot, to_slot, relation_type in sorted(prev - frozenset(seen)):
                events.append(
                    relation_removed(
                        self.entity_id(pid, from_slot),
                        self.entity_id(pid, to_slot),
                        relation_type,
                        ts,
                        pair_index=pid,
                        from_slot=from_slot,
                        to_slot=to_slot,
                    )
                )
        self._last_relations[pid] = frozenset(seen)
        return events


# ---------------------------------------------------------------- module-level GT


def translate_gt(
    manifest: DatasetManifest | Sequence[PairSample],
    *,
    split: str | None = None,
    sample_indices: Iterable[int] | None = None,
    translator: EventTranslator | None = None,
    previous_relations: (
        Mapping[int, Iterable[tuple[int, int, str]]] | None
    ) = None,
) -> list[WorldEvent]:
    """由 manifest（GT）生成事件流，规则与预测端完全对称，保证可比性。

    Args:
        manifest: `DatasetManifest` 或 `Sequence[PairSample]`。
        split: 只翻译该 split。
        sample_indices: 只翻译这些 pair 下标（如 60 对抽样）；与 split 同时给出取交集。
        translator: 复用外部翻译器（决定 ID/时间戳基准）；缺省新建。
        previous_relations: 可选 {pair_index: 前一次快照关系}，用于构造 RelationRemoved
            的合成用例；v2 真实数据不提供（没有 before 关系标注）。
    """
    pairs = select_pairs(manifest, split=split, sample_indices=sample_indices)
    tr = translator if translator is not None else EventTranslator()
    events: list[WorldEvent] = []
    for pair in pairs:
        prev = None
        if previous_relations is not None:
            prev = previous_relations.get(pair.index)
        events.extend(tr.translate_gt_pair(pair, previous_relations=prev))
    validate_event_stream(events)
    return events


__all__ = [
    "EVENT_KINDS",
    "ENTITY_ID_BASE",
    "DEFAULT_TS_START_MS",
    "DEFAULT_TS_STEP_MS",
    "RELATION_REMOVED_NA_REASON",
    "ID_SWITCH_NA_REASON",
    "WorldEvent",
    "EventTranslator",
    "entity_created",
    "state_changed",
    "relation_added",
    "relation_removed",
    "as_prediction_map",
    "select_pairs",
    "validate_event_stream",
    "events_to_dicts",
    "translate_gt",
]
