#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集验收脚本（数据方交付前 / 模型侧收货时运行）。

支持两种数据集：
    spatial  —— 空间/关系数据：每行一对 before/after（见《数据意见》§3.2）
    object   —— 物品数据：每行一张图 + objects

用法：
    python starlife_scripts/validate_dataset.py --manifest dataset/manifest.jsonl
    python starlife_scripts/validate_dataset.py --manifest dataset/manifest.jsonl \
        --catalog catalog.csv --vocab-version 1.0.0 --report report.csv
    python starlife_scripts/validate_dataset.py --manifest dataset/manifest.jsonl --full
    python starlife_scripts/validate_dataset.py --manifest dataset/manifest.jsonl --mode object

退出码：0 = 无错误；1 = 有错误；2 = 无法运行。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

RELATIONS = {
    "on", "contains", "held_by", "in", "near", "next_to",
    "part_of", "belongs_to", "facing", "above", "below",
}
REL_POS = {
    "top", "inside", "front", "back", "left", "right",
    "front_left", "front_right", "back_left", "back_right",
}
CHANGES = {"appeared", "displaced", "disappeared"}
SPLITS = {"train", "val", "test"}
TIER_MIN = {"A": 500, "B": 20}
NEG_RATIO = (0.30, 0.50)


class Report:
    def __init__(self) -> None:
        self.errors: list[dict] = []
        self.warnings: list[dict] = []

    def err(self, line: int, code: str, msg: str) -> None:
        self.errors.append({"line": line, "level": "error", "code": code, "message": msg})

    def warn(self, line: int, code: str, msg: str) -> None:
        self.warnings.append({"line": line, "level": "warning", "code": code, "message": msg})

    def emit(self, level: str, line: int, code: str, msg: str) -> None:
        if level == "error":
            self.err(line, code, msg)
        else:
            self.warn(line, code, msg)


def sha256_file(path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    cache[path] = h.hexdigest()
    return cache[path]


def image_size(path: Path, cache: dict[Path, tuple]) -> tuple | None:
    if path in cache:
        return cache[path]
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            size = im.size
    except Exception as exc:  # noqa: BLE001
        cache[path] = None
        print(f"  ! 无法解码图片 {path}: {exc}", file=sys.stderr)
        return None
    cache[path] = size
    return size


def resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else base / p


def parse_bbox(value, where: str, rep: Report, line: int, img_w: int | None, img_h: int | None) -> bool:
    ok = True
    if not isinstance(value, list) or len(value) != 4:
        rep.err(line, "bbox", f"{where}: bbox 必须是 4 个数字")
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        rep.err(line, "bbox", f"{where}: bbox 含非数字")
        return False
    x1, y1, x2, y2 = value
    if not (x1 < x2 and y1 < y2):
        rep.err(line, "bbox", f"{where}: bbox 必须 x1<x2 且 y1<y2")
        ok = False
    if x1 < 0 or y1 < 0:
        rep.err(line, "bbox", f"{where}: bbox 不能为负")
        ok = False
    if img_w is not None and img_h is not None:
        if x2 > img_w or y2 > img_h:
            rep.err(line, "bbox", f"{where}: bbox 超出图像 {img_w}x{img_h}")
            ok = False
    return ok


def load_catalog(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.exists():
        print(f"[ERROR] catalog 不存在: {path}", file=sys.stderr)
        sys.exit(2)
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("key") or "").strip()
            if key:
                out[key] = row
    return out


def check_slots(entities: list[dict], rep: Report, line: int) -> set[int]:
    slots = []
    for i, ent in enumerate(entities):
        if not isinstance(ent, dict):
            rep.err(line, "entity", f"entities[{i}] 不是对象")
            continue
        slot = ent.get("slot")
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            rep.err(line, "slot", f"entities[{i}].slot 必须是非负整数")
            continue
        slots.append(slot)
    if slots and sorted(slots) != list(range(len(slots))):
        rep.err(line, "slot", f"slot 必须从 0 开始连续，实际 {sorted(slots)}")
    if len(set(slots)) != len(slots):
        rep.err(line, "slot", "slot 重复")
    return set(slots)


def validate_spatial(rows: list[tuple[int, str]], base: Path, args, rep: Report) -> dict:
    counts = {
        "pairs": 0, "negatives": 0,
        "relations": Counter(), "rel_pos": Counter(), "changes": Counter(),
        "types": Counter(), "splits": Counter(),
    }
    img_cache: dict[Path, tuple] = {}
    hash_cache: dict[Path, str] = {}
    pair_hash: dict[str, int] = {}
    scene_splits: dict[str, set] = defaultdict(set)
    vocab_versions: set[str] = set()
    catalog = load_catalog(Path(args.catalog) if args.catalog else None)

    for lineno, raw in rows:
        counts["pairs"] += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            rep.err(lineno, "json", f"JSON 解析失败: {exc}")
            continue
        if not isinstance(obj, dict):
            rep.err(lineno, "json", "每行必须是 JSON 对象")
            continue

        for k in ("pair_id", "split", "vocab_version", "before", "after", "entities", "relations", "changes", "negative"):
            if k not in obj:
                rep.err(lineno, "field", f"缺少字段 {k}")
        split = obj.get("split")
        if split not in SPLITS:
            rep.err(lineno, "split", f"split 不合法: {split!r}")
        else:
            counts["splits"][split] += 1
        if isinstance(obj.get("vocab_version"), str):
            vocab_versions.add(obj["vocab_version"])

        before_p = resolve(base, str(obj.get("before", "")))
        after_p = resolve(base, str(obj.get("after", "")))
        size_b = size_a = None
        for label, p in (("before", before_p), ("after", after_p)):
            if not p.exists():
                rep.err(lineno, "image", f"{label} 图片不存在: {p}")
                continue
            size = image_size(p, img_cache)
            if size is None:
                rep.err(lineno, "image", f"{label} 图片无法解码: {p}")
                continue
            if label == "before":
                size_b = size
            else:
                size_a = size
        if size_b and size_a and size_b != size_a:
            rep.err(lineno, "image", f"before/after 尺寸不一致: {size_b} vs {size_a}")
        if before_p.exists() and after_p.exists():
            ph = sha256_file(before_p, hash_cache) + ":" + sha256_file(after_p, hash_cache)
            if ph in pair_hash:
                rep.err(lineno, "duplicate", f"与第 {pair_hash[ph]} 行的图片对重复")
            else:
                pair_hash[ph] = lineno

        entities = obj.get("entities")
        if not isinstance(entities, list):
            rep.err(lineno, "entities", "entities 必须是数组")
            entities = []
        slots = check_slots(entities, rep, lineno)
        type_of: dict[int, str] = {}
        for i, ent in enumerate(entities):
            if not isinstance(ent, dict):
                continue
            slot = ent.get("slot")
            tk = ent.get("type_key")
            if not isinstance(tk, str) or not tk.strip():
                rep.err(lineno, "type_key", f"entities[{i}].type_key 必须是非空字符串")
            else:
                type_of[slot] = tk
                counts["types"][tk] += 1
                if catalog and tk not in catalog:
                    rep.warn(lineno, "catalog", f"type_key 不在 catalog: {tk}")
            if not isinstance(ent.get("present_after"), bool):
                rep.err(lineno, "present_after", f"entities[{i}].present_after 必须是 bool")
            w, h = size_b if size_b else (None, None)
            if "bbox" in ent:
                parse_bbox(ent["bbox"], f"entities[{i}]", rep, lineno, w, h)
            else:
                rep.err(lineno, "bbox", f"entities[{i}] 缺少 bbox")

        changes = obj.get("changes")
        change_seen: set[int] = set()
        if isinstance(changes, list):
            for i, ch in enumerate(changes):
                if not isinstance(ch, dict):
                    rep.err(lineno, "change", f"changes[{i}] 不是对象")
                    continue
                slot = ch.get("slot")
                if slot not in slots:
                    rep.err(lineno, "change", f"changes[{i}].slot={slot} 不存在")
                if slot in change_seen:
                    rep.err(lineno, "change", f"changes 中 slot={slot} 重复")
                change_seen.add(slot)
                ctype = ch.get("type")
                if ctype not in CHANGES:
                    rep.err(lineno, "change", f"changes[{i}].type 非法: {ctype!r}（只写 appeared/displaced/disappeared）")
                else:
                    counts["changes"][ctype] += 1
        else:
            rep.err(lineno, "changes", "changes 必须是数组")

        relations = obj.get("relations")
        rel_seen: dict[tuple, str] = {}
        if isinstance(relations, list):
            for i, rel in enumerate(relations):
                if not isinstance(rel, dict):
                    rep.err(lineno, "relation", f"relations[{i}] 不是对象")
                    continue
                fr, to = rel.get("from_slot"), rel.get("to_slot")
                if fr not in slots or to not in slots:
                    rep.err(lineno, "relation", f"relations[{i}] 引用不存在的 slot: {fr}->{to}")
                if fr == to:
                    rep.err(lineno, "relation", f"relations[{i}] from_slot 与 to_slot 相同")
                rtype = rel.get("type")
                if rtype not in RELATIONS:
                    rep.err(lineno, "relation", f"relations[{i}].type 非法: {rtype!r}（不允许 none）")
                else:
                    counts["relations"][rtype] += 1
                key = (fr, to)
                if key in rel_seen:
                    rep.err(lineno, "relation", f"({fr},{to}) 出现多条关系: {rel_seen[key]} / {rtype}")
                else:
                    rel_seen[key] = rtype
                contact, support = rel.get("contact"), rel.get("support")
                if contact is not None and not isinstance(contact, bool):
                    rep.err(lineno, "relation", f"relations[{i}].contact 必须是 bool")
                if support is not None and not isinstance(support, bool):
                    rep.err(lineno, "relation", f"relations[{i}].support 必须是 bool")
                rp = rel.get("rel_pos")
                if rp is not None:
                    if rp not in REL_POS:
                        rep.err(lineno, "relation", f"relations[{i}].rel_pos 非法: {rp!r}")
                    else:
                        counts["rel_pos"][rp] += 1
                dist = rel.get("dist_m")
                if dist is not None and (not isinstance(dist, (int, float)) or dist < 0):
                    rep.err(lineno, "relation", f"relations[{i}].dist_m 必须是非负数")

                if rtype == "on":
                    if contact is False or support is False:
                        rep.warn(lineno, "rule", f"relations[{i}] on 通常 contact/support 都为 true")
                    if rp == "inside":
                        rep.err(lineno, "rule", f"relations[{i}] on 不应使用 rel_pos=inside")
                elif rtype == "in":
                    if rp is not None and rp != "inside":
                        rep.err(lineno, "rule", f"relations[{i}] in 的 rel_pos 必须是 inside")
                elif rtype in ("above", "below"):
                    if contact is True or support is True:
                        rep.err(lineno, "rule", f"relations[{i}] {rtype} 必须 contact=false, support=false")
                elif rtype in ("near", "next_to"):
                    if contact is True or support is True:
                        rep.err(lineno, "rule", f"relations[{i}] {rtype} 必须 contact=false, support=false")

        # 同一事实的逆关系不应同时出现
        for (a, b), t in rel_seen.items():
            inverse = {"in": "contains", "part_of": "belongs_to"}.get(t)
            if inverse and rel_seen.get((b, a)) == inverse:
                rep.warn(lineno, "rule", f"({a},{b},{t}) 与逆关系同时出现，建议只保留 in/part_of")

        if obj.get("negative") is True:
            counts["negatives"] += 1

        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        scene = meta.get("scene_id") or meta.get("background_id")
        if scene and split in SPLITS:
            scene_splits[str(scene)].add(split)

    for scene, splits in scene_splits.items():
        if len(splits) > 1:
            rep.err(0, "leak", f"场景 {scene} 同时出现在 {sorted(splits)}，存在切分泄漏")

    total = counts["pairs"]
    if total:
        ratio = counts["negatives"] / total
        if not (NEG_RATIO[0] <= ratio <= NEG_RATIO[1]):
            rep.warn(0, "negative", f"负样本比例 {ratio:.1%}，建议 {NEG_RATIO[0]:.0%}~{NEG_RATIO[1]:.0%}")

    if args.vocab_version and vocab_versions and vocab_versions != {args.vocab_version}:
        rep.err(0, "version", f"vocab_version 不一致: {sorted(vocab_versions)}，期望 {args.vocab_version}")

    if args.full and catalog:
        for key, n in counts["types"].items():
            tier = (catalog.get(key, {}).get("tier") or "").upper()
            need = TIER_MIN.get(tier)
            if need and n < need:
                rep.err(0, "coverage", f"类别 {key}（tier {tier}）只有 {n} 条，要求 ≥{need}")

    return {**counts, "vocab_versions": vocab_versions}


def validate_object(rows: list[tuple[int, str]], base: Path, args, rep: Report) -> dict:
    counts = {
        "images": 0, "negatives": 0, "types": Counter(), "splits": Counter(),
        "sizes": Counter(),
    }
    img_cache: dict[Path, tuple] = {}
    hash_cache: dict[Path, str] = {}
    seen_hash: dict[str, int] = {}
    scene_splits: dict[str, set] = defaultdict(set)
    vocab_versions: set[str] = set()
    catalog = load_catalog(Path(args.catalog) if args.catalog else None)

    for lineno, raw in rows:
        counts["images"] += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            rep.err(lineno, "json", f"JSON 解析失败: {exc}")
            continue
        if not isinstance(obj, dict):
            rep.err(lineno, "json", "每行必须是 JSON 对象")
            continue
        for k in ("image_id", "split", "vocab_version", "image", "objects", "negative"):
            if k not in obj:
                rep.err(lineno, "field", f"缺少字段 {k}")
        split = obj.get("split")
        if split not in SPLITS:
            rep.err(lineno, "split", f"split 不合法: {split!r}")
        else:
            counts["splits"][split] += 1
        if isinstance(obj.get("vocab_version"), str):
            vocab_versions.add(obj["vocab_version"])

        img_p = resolve(base, str(obj.get("image", "")))
        size = None
        if not img_p.exists():
            rep.err(lineno, "image", f"图片不存在: {img_p}")
        else:
            size = image_size(img_p, img_cache)
            if size is None:
                rep.err(lineno, "image", f"图片无法解码: {img_p}")
            else:
                counts["sizes"][f"{size[0]}x{size[1]}"] += 1
                h = sha256_file(img_p, hash_cache)
                if h in seen_hash:
                    rep.err(lineno, "duplicate", f"与第 {seen_hash[h]} 行的图片重复")
                else:
                    seen_hash[h] = lineno

        objects = obj.get("objects")
        slots = check_slots(objects if isinstance(objects, list) else [], rep, lineno)
        if not isinstance(objects, list):
            rep.err(lineno, "objects", "objects 必须是数组")
            objects = []
        if obj.get("negative") is True:
            counts["negatives"] += 1
            neg_type = obj.get("negative_type")
            # 难负样本（hard_confusable）故意保留 distractor 的真实 bbox，不算警告
            if objects and neg_type not in ("hard_confusable",):
                rep.warn(lineno, "negative", "negative=true 但 objects 非空（且不是 hard_confusable）")
        elif not objects:
            rep.err(lineno, "objects", "negative=false 时 objects 不能为空")
        w, h = size if size else (None, None)
        for i, o in enumerate(objects):
            if not isinstance(o, dict):
                rep.err(lineno, "object", f"objects[{i}] 不是对象")
                continue
            tk = o.get("type_key")
            if not isinstance(tk, str) or not tk.strip():
                rep.err(lineno, "type_key", f"objects[{i}].type_key 必须是非空字符串")
            else:
                counts["types"][tk] += 1
                if catalog and tk not in catalog:
                    rep.warn(lineno, "catalog", f"type_key 不在 catalog: {tk}")
            if "bbox" not in o:
                rep.err(lineno, "bbox", f"objects[{i}] 缺少 bbox")
            else:
                parse_bbox(o["bbox"], f"objects[{i}]", rep, lineno, w, h)
            if "occlusion" in o and (not isinstance(o["occlusion"], (int, float)) or not (0 <= o["occlusion"] <= 1)):
                rep.err(lineno, "occlusion", f"objects[{i}].occlusion 必须在 0~1")
            if "truncated" in o and not isinstance(o["truncated"], bool):
                rep.err(lineno, "truncated", f"objects[{i}].truncated 必须是 bool")

        scene = obj.get("scene") if isinstance(obj.get("scene"), dict) else {}
        scene_id = scene.get("scene_id") or scene.get("background_id")
        if scene_id and split in SPLITS:
            scene_splits[str(scene_id)].add(split)

    for scene, splits in scene_splits.items():
        if len(splits) > 1:
            rep.err(0, "leak", f"背景/场景 {scene} 同时出现在 {sorted(splits)}，存在切分泄漏")

    if len(counts["sizes"]) > 1:
        rep.warn(0, "image", f"图片尺寸不统一: {dict(counts['sizes'])}")

    if args.vocab_version and vocab_versions and vocab_versions != {args.vocab_version}:
        rep.err(0, "version", f"vocab_version 不一致: {sorted(vocab_versions)}，期望 {args.vocab_version}")

    if args.full and catalog:
        for key, n in counts["types"].items():
            tier = (catalog.get(key, {}).get("tier") or "").upper()
            need = TIER_MIN.get(tier)
            if need and n < need:
                rep.err(0, "coverage", f"类别 {key}（tier {tier}）只有 {n} 条，要求 ≥{need}")

    return {**counts, "vocab_versions": vocab_versions}


def main() -> None:
    ap = argparse.ArgumentParser(description="StarLife 数据集验收")
    ap.add_argument("--manifest", required=True, help="manifest.jsonl 路径")
    ap.add_argument("--mode", choices=["auto", "spatial", "object"], default="auto")
    ap.add_argument("--catalog", default="", help="catalog.csv（可选，用于检查 type_key 与分层数量）")
    ap.add_argument("--vocab-version", default="", help="期望的 vocab_version（可选）")
    ap.add_argument("--full", action="store_true", help="按全量标准检查覆盖率（A≥500 / B≥20）")
    ap.add_argument("--report", default="", help="错误/警告报告 CSV（可选）")
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 行（调试用）")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[ERROR] manifest 不存在: {manifest}", file=sys.stderr)
        sys.exit(2)
    base = manifest.parent

    rows: list[tuple[int, str]] = []
    with manifest.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            rows.append((lineno, text))
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        print("[ERROR] manifest 为空", file=sys.stderr)
        sys.exit(2)

    first = None
    try:
        first = json.loads(rows[0][1])
    except json.JSONDecodeError:
        pass

    mode = args.mode
    if mode == "auto":
        if isinstance(first, dict) and "before" in first and "after" in first:
            mode = "spatial"
        elif isinstance(first, dict) and "image" in first:
            mode = "object"
        else:
            print("[ERROR] 无法识别数据集类型，请用 --mode spatial|object", file=sys.stderr)
            sys.exit(2)

    rep = Report()
    if mode == "spatial":
        stats = validate_spatial(rows, base, args, rep)
    else:
        stats = validate_object(rows, base, args, rep)

    print("=" * 64)
    print(f"模式: {mode}    文件: {manifest}")
    if mode == "spatial":
        print(f"帧对: {stats['pairs']}    负样本: {stats['negatives']}    split: {dict(stats['splits'])}")
        print(f"类型数: {len(stats['types'])}    关系: {dict(stats['relations'])}")
        print(f"方位: {dict(stats['rel_pos'])}    变化: {dict(stats['changes'])}")
    else:
        print(f"图片: {stats['images']}    负样本: {stats['negatives']}    split: {dict(stats['splits'])}")
        print(f"类型数: {len(stats['types'])}    尺寸: {dict(stats['sizes'])}")
    if stats["vocab_versions"]:
        print(f"vocab_version: {sorted(stats['vocab_versions'])}")
    print(f"错误: {len(rep.errors)}    警告: {len(rep.warnings)}")
    print("=" * 64)

    for e in rep.errors[:50]:
        print(f"[ERROR][行 {e['line']}][{e['code']}] {e['message']}")
    if len(rep.errors) > 50:
        print(f"... 其余 {len(rep.errors) - 50} 条错误见报告")
    for w in rep.warnings[:30]:
        print(f"[WARN ][行 {w['line']}][{w['code']}] {w['message']}")
    if len(rep.warnings) > 30:
        print(f"... 其余 {len(rep.warnings) - 30} 条警告见报告")

    if args.report:
        out = Path(args.report)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["line", "level", "code", "message"])
            writer.writeheader()
            writer.writerows(rep.errors + rep.warnings)
        print(f"\n报告已写入: {out}")

    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()
