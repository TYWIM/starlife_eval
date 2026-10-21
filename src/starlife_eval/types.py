"""共享数据模型。签名由 Lead 维护，teammate 只读不改（需要加字段先报）。

术语：
- slot 编号：数据集内 per-pair 的实体编号（manifest `entities[].slot`）。
- 全局 slot 下标：特征 npz 的行号，等于 `manifest.slot_offsets[pair_index] + slot`。
- 头级指标用全局下标（与 npz 的 slot_change/slot_type/rel_from/rel_to 直接对齐）；
- 事件级/状态回放/云端协议用 per-pair slot 编号（与 manifest 对齐）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CHANGE_LABELS: tuple[str, ...] = ("none", "appeared", "displaced", "disappeared")
CHANGE_INDEX: Mapping[str, int] = {k: i for i, k in enumerate(CHANGE_LABELS)}


# --------------------------------------------------------------------------- GT


@dataclass(frozen=True)
class Entity:
    slot: int
    bbox: tuple[float, float, float, float]
    type_key: str
    present_after: bool


@dataclass(frozen=True)
class Change:
    slot: int
    type: str  # none 只出现在 GT 缺省时，manifest 不列


@dataclass(frozen=True)
class Relation:
    from_slot: int
    to_slot: int
    type: str
    rel_pos: str | None
    contact: bool
    support: bool
    dist_m: float | None


@dataclass(frozen=True)
class PairSample:
    """一个 before/after 帧对（manifest 一行）。"""

    index: int  # manifest 行号
    pair_id: str
    split: str
    before: str  # 相对数据集根
    after: str
    entities: tuple[Entity, ...]
    changes: Mapping[int, str]
    relations: tuple[Relation, ...]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(e.slot for e in self.entities)

    def entity(self, slot: int) -> Entity:
        for e in self.entities:
            if e.slot == slot:
                return e
        raise KeyError(slot)


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    pairs: tuple[PairSample, ...]
    meta: Mapping[str, Any]
    slot_offsets: tuple[int, ...]  # len == len(pairs) + 1

    def split_indices(self, split: str) -> list[int]:
        return [p.index for p in self.pairs if p.split == split]

    def global_slot(self, pair_index: int, slot: int) -> int:
        return self.slot_offsets[pair_index] + slot

    def pair_of_global_slot(self, global_slot: int) -> int:
        lo, hi = 0, len(self.pairs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.slot_offsets[mid] <= global_slot:
                lo = mid
            else:
                hi = mid - 1
        return lo


# --------------------------------------------------------------------- features


@dataclass(frozen=True)
class FeatureBundle:
    """特征 npz 的只读视图（键名见 ARCHITECTURE.md §3）。"""

    path: Path
    arrays: Mapping[str, np.ndarray]
    meta: Mapping[str, Any]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    @property
    def type_keys(self) -> list[str]:
        return list(self.meta["type_keys"])

    @property
    def rel_types(self) -> list[str]:
        return list(self.meta["rel_types"])

    @property
    def rel_pos_keys(self) -> list[str]:
        return list(self.meta["rel_pos_keys"])


# ---------------------------------------------------------------- predictions


@dataclass(frozen=True)
class HeadPrediction:
    """与 npz 行对齐的本地头预测（含关系负样本）。"""

    change: np.ndarray  # (N_slot,) int64
    type: np.ndarray  # (N_slot,) int64
    relation: np.ndarray  # (N_rel_model,) int64, 末 n_none 个是负样本
    rel_pos: np.ndarray  # (N_rel_model,) int64
    attrs: np.ndarray  # (N_rel_model, 2) int64
    n_relation_positives: int  # = len(z["rel_from"])
    latency_s: float | None = None

    def __post_init__(self) -> None:
        n = len(self.change)
        if len(self.type) != n:
            raise ValueError(f"change/type 长度不一致: {len(self.change)} vs {len(self.type)}")
        if len(self.relation) != len(self.rel_pos) or len(self.relation) != len(self.attrs):
            raise ValueError("relation/rel_pos/attrs 长度不一致")


@dataclass(frozen=True)
class RelationPrediction:
    from_slot: int
    to_slot: int
    type: str
    rel_pos: str | None = None
    contact: bool | None = None
    support: bool | None = None


@dataclass
class PairPrediction:
    """逐 pair 结构化预测（云端/事件/状态回放用）。"""

    pair_index: int
    changes: dict[int, str] = field(default_factory=dict)
    types: dict[int, str] = field(default_factory=dict)
    relations: list[RelationPrediction] = field(default_factory=list)

    def relation_map(self) -> dict[tuple[int, int], RelationPrediction]:
        out: dict[tuple[int, int], RelationPrediction] = {}
        for r in self.relations:
            out.setdefault((r.from_slot, r.to_slot), r)
        return out


# --------------------------------------------------------------------- runtime


@dataclass
class RuntimeStats:
    """真实测量值；无法测的字段保持 None 并进 `na`。"""

    latency_ms_mean: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_max: float | None = None
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    cost_usd: float | None = None
    cost_basis: str | None = None
    n_calls: int | None = None
    source: str = "measured"


@dataclass
class ModelRun:
    """一个模型在一次评测里的全部产物。"""

    model_id: str
    kind: str  # local_heads | replay_cloud | chat_vlm | unavailable
    mode: str = "LOCAL-HEADS"
    family: str = "StarLife"
    provider: str = "StarLife AI"
    head: HeadPrediction | None = None
    pairs: dict[int, PairPrediction] | None = None
    runtime: RuntimeStats = field(default_factory=RuntimeStats)
    na: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- utilities


def as_int64(a: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
