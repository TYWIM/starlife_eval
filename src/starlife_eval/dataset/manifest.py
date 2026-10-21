"""数据集 manifest 加载与校验。

契约见 ``ARCHITECTURE.md`` §2。这里的解析结果必须与真实产物
``starlife/starlife_scripts/extract_features.py`` (``run_spatial``) 以及
``train_heads.py`` (``load_manifest``) 对同一份 ``manifest.jsonl`` 的理解完全一致：

- ``PairSample.index`` 就是 manifest 行号（与 npz 的 ``slot_pair`` 值域一致）；
- 全局 slot 下标 = ``slot_offsets[pair_index] + slot``，其中
  ``slot_offsets`` 是 entities 数量前缀和（train_heads.py:130-140）；
- 因此要求每个 pair 的 ``entities[].slot`` 从 0 开始连续，否则全局下标会错位。

本模块只做“读 + 严格校验”，绝不补造缺失字段：数据不合法直接抛错。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from ..types import CHANGE_LABELS, DatasetManifest, Entity, PairSample, Relation

MANIFEST_NAME = "manifest.jsonl"
META_NAME = "meta.json"


class ManifestError(ValueError):
    """manifest 结构不合法（消息里带文件名/行号，便于定位真实产物问题）。"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 manifest：{path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ManifestError(f"{path}:{lineno} 期望 JSON 对象，实际 {type(row).__name__}")
        rows.append(row)
    if not rows:
        raise ManifestError(f"{path} 为空：没有任何 pair")
    return rows


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ManifestError(msg)


def _parse_bbox(raw_bbox: Any, where: str) -> tuple[float, float, float, float]:
    _require(
        isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4,
        f"{where} bbox 必须是长度 4 的数组，实际 {raw_bbox!r}",
    )
    vals = tuple(float(v) for v in raw_bbox)
    _require(all(math.isfinite(v) for v in vals), f"{where} bbox 含非有限值 {raw_bbox!r}")
    x0, y0, x1, y1 = vals
    _require(x1 >= x0 and y1 >= y0, f"{where} bbox 需满足 x1>=x0,y1>=y0，实际 {raw_bbox!r}")
    return vals  # type: ignore[return-value]


def _parse_entities(raw_entities: Any, pair_id: str) -> tuple[Entity, ...]:
    _require(isinstance(raw_entities, list), f"pair {pair_id}: entities 必须是数组")
    out: list[Entity] = []
    for pos, raw in enumerate(raw_entities):
        where = f"pair {pair_id} entities[{pos}]"
        _require(isinstance(raw, dict), f"{where} 必须是对象")
        _require("slot" in raw, f"{where} 缺少 slot")
        slot = int(raw["slot"])
        _require(slot == pos, f"{where} slot 必须从 0 连续（期望 {pos}，实际 {slot}）")
        type_key = raw.get("type_key")
        _require(isinstance(type_key, str) and type_key, f"{where} type_key 必须是非空字符串")
        out.append(
            Entity(
                slot=slot,
                bbox=_parse_bbox(raw.get("bbox"), where),
                type_key=type_key,
                present_after=bool(raw.get("present_after", True)),
            )
        )
    return tuple(out)


def _parse_changes(raw_changes: Any, entities: tuple[Entity, ...], pair_id: str) -> dict[int, str]:
    if raw_changes is None:
        return {}
    _require(isinstance(raw_changes, list), f"pair {pair_id}: changes 必须是数组")
    slots = {e.slot for e in entities}
    out: dict[int, str] = {}
    for pos, raw in enumerate(raw_changes):
        where = f"pair {pair_id} changes[{pos}]"
        _require(isinstance(raw, dict), f"{where} 必须是对象")
        _require("slot" in raw, f"{where} 缺少 slot")
        slot = int(raw["slot"])
        _require(slot in slots, f"{where} slot={slot} 不在 entities 中")
        _require(slot not in out, f"{where} slot={slot} 重复")
        ctype = raw.get("type")
        _require(
            isinstance(ctype, str) and ctype in CHANGE_LABELS,
            f"{where} type={ctype!r} 不在 {list(CHANGE_LABELS)} 中",
        )
        out[slot] = ctype
    return out


def _parse_relations(raw_relations: Any, entities: tuple[Entity, ...], pair_id: str) -> tuple[Relation, ...]:
    if raw_relations is None:
        return ()
    _require(isinstance(raw_relations, list), f"pair {pair_id}: relations 必须是数组")
    slots = {e.slot for e in entities}
    out: list[Relation] = []
    for pos, raw in enumerate(raw_relations):
        where = f"pair {pair_id} relations[{pos}]"
        _require(isinstance(raw, dict), f"{where} 必须是对象")
        _require("from_slot" in raw and "to_slot" in raw, f"{where} 缺少 from_slot/to_slot")
        frm, to = int(raw["from_slot"]), int(raw["to_slot"])
        _require(frm in slots, f"{where} from_slot={frm} 不在 entities 中")
        _require(to in slots, f"{where} to_slot={to} 不在 entities 中")
        rtype = raw.get("type")
        _require(isinstance(rtype, str) and rtype, f"{where} type 必须是非空字符串")
        rel_pos = raw.get("rel_pos")
        _require(
            rel_pos is None or isinstance(rel_pos, str),
            f"{where} rel_pos 必须是字符串或 null，实际 {rel_pos!r}",
        )
        dist = raw.get("dist_m")
        _require(dist is None or isinstance(dist, (int, float)), f"{where} dist_m 必须是数字或 null")
        out.append(
            Relation(
                from_slot=frm,
                to_slot=to,
                type=rtype,
                rel_pos=rel_pos,
                contact=bool(raw.get("contact", False)),
                support=bool(raw.get("support", False)),
                dist_m=None if dist is None else float(dist),
            )
        )
    return tuple(out)


def _load_dataset_meta(root: Path) -> dict[str, Any]:
    path = root / META_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"{path} 期望 JSON 对象")
    return raw


def load_manifest(dataset_dir: str | Path) -> DatasetManifest:
    """读取 ``<dataset_dir>/manifest.jsonl`` + ``meta.json`` → :class:`DatasetManifest`。

    校验：pair_id 唯一、slot 从 0 连续、changes/relations 的 slot 引用有效、
    before/after 非空、split 非空。任何不合法都抛 :class:`ManifestError`
    （继承 ``ValueError``），绝不填补默认值。
    """
    root = Path(dataset_dir)
    path = root / MANIFEST_NAME
    rows = _read_jsonl(path)

    pairs: list[PairSample] = []
    offsets: list[int] = [0]
    seen_pair_ids: dict[str, int] = {}

    for lineno, raw in enumerate(rows):
        where = f"{path} 第 {lineno + 1} 行"
        pair_id = raw.get("pair_id")
        _require(isinstance(pair_id, str) and pair_id, f"{where} pair_id 必须是非空字符串")
        _require(pair_id not in seen_pair_ids, f"{where} pair_id={pair_id!r} 重复（首次出现在第 {seen_pair_ids.get(pair_id, -1) + 1} 行）")
        seen_pair_ids[pair_id] = lineno

        split = raw.get("split")
        _require(isinstance(split, str) and split, f"{where} split 必须是非空字符串")
        before, after = raw.get("before"), raw.get("after")
        _require(isinstance(before, str) and before, f"{where} before 必须是非空相对路径")
        _require(isinstance(after, str) and after, f"{where} after 必须是非空相对路径")

        entities = _parse_entities(raw.get("entities", []), pair_id)
        changes = _parse_changes(raw.get("changes"), entities, pair_id)
        relations = _parse_relations(raw.get("relations"), entities, pair_id)

        pairs.append(
            PairSample(
                index=lineno,
                pair_id=pair_id,
                split=split,
                before=before,
                after=after,
                entities=entities,
                changes=changes,
                relations=relations,
                raw=raw,
            )
        )
        offsets.append(offsets[-1] + len(entities))

    manifest = DatasetManifest(
        root=root,
        pairs=tuple(pairs),
        meta=_load_dataset_meta(root),
        slot_offsets=tuple(offsets),
    )
    _validate_global_layout(manifest)
    return manifest


def _validate_global_layout(manifest: DatasetManifest) -> None:
    """二次校验全局下标映射（长度/单调性），防止 0-entity pair 造成静默错位。"""
    if len(manifest.slot_offsets) != len(manifest.pairs) + 1:
        raise ManifestError(
            f"slot_offsets 长度 {len(manifest.slot_offsets)} != pairs+1 ({len(manifest.pairs) + 1})"
        )
    if manifest.slot_offsets[0] != 0:
        raise ManifestError(f"slot_offsets 必须从 0 开始，实际 {manifest.slot_offsets[0]}")
    for i in range(len(manifest.pairs)):
        if manifest.slot_offsets[i + 1] < manifest.slot_offsets[i]:
            raise ManifestError(f"slot_offsets 非单调：{manifest.slot_offsets}")


def split_pairs(manifest: DatasetManifest, split: str) -> tuple[PairSample, ...]:
    """按 split 过滤（返回 PairSample，``.index`` 仍是原 manifest 行号）。"""
    return tuple(p for p in manifest.pairs if p.split == split)


def manifest_pair_map(manifest: DatasetManifest) -> dict[int, PairSample]:
    return {p.index: p for p in manifest.pairs}


def iter_global_slots(manifest: DatasetManifest) -> Iterable[tuple[int, PairSample, Entity]]:
    """按全局下标顺序 yield ``(global_slot, pair, entity)``（与 npz 行序一致）。"""
    for pair in manifest.pairs:
        base = manifest.slot_offsets[pair.index]
        for entity in pair.entities:
            yield base + entity.slot, pair, entity
