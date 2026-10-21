#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 1 万类以上实体词表：种子目录 + Open Images 类名 + WordNet 层级/中文名。

设计（见《决策模型计划书》§3.2–3.6）：
- A 热类：有真实/合成图像，闭集头可训（catalog tier=A）；
- B 温类：合成资产库可渲染，嵌入头训练（catalog tier=B）；
- C 冷类：没有图像，只靠冻结文本编码器做零样本。

词表条目来源（都不需要人工标注）：
1. `catalog.csv`：3533 个有 3D 资产的类（A/B 层，带中英文、别名、描述）；
2. Open Images V7 类名（~2 万条，带英文描述）：映射到 WordNet synset，作为 C 层；
3. WordNet：synset 层级（parent）、同义词（aliases）、释义（description）、中文名（OMW）。

产物：entity_vocab.json（id/key/zh/en/parent/tier/aliases/description/index）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STOP_LEMMAS = {"object", "thing", "entity", "physical_entity", "whole", "unit", "artefact",
               "artifact", "instrumentality", "commodity", "item", "matter", "substance",
               "being", "organism", "living_thing", "causal_agent", "agent"}

# ID 分段：A 1–9999，B 10000–29999，C 30000+
SEGMENT = {"A": (1, 9999), "B": (10000, 29999), "C": (30000, 65000)}
PHYSICAL = "physical_entity.n.01"


def snake(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def load_catalog(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("key"):
                rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def load_openimages(path: Path) -> list[tuple[str, str]]:
    """返回 [(类名, 描述)]；文件为 /m/xxxx,"Aircraft",... 或 标签,描述 两种格式。"""
    out = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[1].strip()
            desc = row[2].strip() if len(row) > 2 else ""
            if name:
                out.append((name, desc))
    return out


def under_physical(syn, cache: dict[str, bool]) -> bool:
    key = syn.name()
    if key in cache:
        return cache[key]
    cur, ok = syn, False
    for _ in range(25):
        if cur.name() == PHYSICAL:
            ok = True
            break
        hs = cur.hypernyms()
        if not hs:
            break
        cur = hs[0]
    cache[key] = ok
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 1 万类以上实体词表")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--openimages", default="", help="oidv7-class-descriptions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=1, help="从已收录 synset 向下扩展几层下位词")
    ap.add_argument("--max-entries", type=int, default=30000)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    import nltk
    from nltk.corpus import wordnet as wn

    for pkg in ("wordnet", "omw-1.4", "omw-2.0"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # noqa: BLE001
                pass

    def zh_names(syn) -> list[str]:
        try:
            return [n for n in syn.lemma_names("cmn")]
        except Exception:  # noqa: BLE001
            return []

    entries: dict[str, dict] = {}
    syn_key: dict[str, str] = {}
    key_syn: dict[str, str] = {}
    phys_cache: dict[str, bool] = {}

    def add_synset(syn, tier: str, key_hint: str = "", zh_hint: str = "", en_hint: str = "",
                   aliases_hint: list[str] | None = None, desc_hint: str = "") -> str | None:
        lemmas = [l for l in syn.lemma_names() if l.replace("_", " ") not in STOP_LEMMAS and l not in STOP_LEMMAS]
        if not lemmas:
            return None
        zh = (zh_hint or "").strip() or (zh_names(syn)[0] if zh_names(syn) else "")
        # key 优先用英文/显式 key，避免中文名的首字符（如「8字形」）变成坏 key
        key = snake(key_hint) or snake(en_hint) or snake(lemmas[0]) or snake(zh)
        if not key:
            return None
        if key in entries:
            for alt in [key_hint, en_hint] + lemmas[1:]:
                cand = snake(alt)
                if cand and cand not in entries:
                    key = cand
                    break
            else:
                return None
        aliases = []
        for a in (aliases_hint or []) + lemmas + zh_names(syn):
            a = str(a).replace("_", " ").strip()
            if a and a not in aliases and a != key:
                aliases.append(a)
        entries[key] = {
            "key": key,
            "zh": zh,
            "en": (en_hint or lemmas[0].replace("_", " ")).strip(),
            "parent": "",
            "tier": tier,
            "aliases": ";".join(aliases[:12]),
            "description": (desc_hint or (syn.definition() or ""))[:240],
            "synset": syn.name(),
        }
        syn_key[syn.name()] = key
        key_syn[key] = syn.name()
        return key

    # 1) catalog 种子（A/B）
    catalog = load_catalog(Path(args.catalog))
    for row in catalog:
        name_for_wn = (row.get("en") or row["key"].replace("_", " ")).strip()
        syns = wn.synsets(name_for_wn, pos="n") or wn.synsets(row["key"].replace("_", " "), pos="n")
        syn = syns[0] if syns else None
        if syn is None:
            entries[row["key"]] = {
                "key": row["key"], "zh": row.get("zh", ""), "en": row.get("en", ""),
                "parent": "", "tier": (row.get("tier") or "B").upper(),
                "aliases": row.get("aliases", ""), "description": row.get("description", ""),
                "synset": "",
            }
            continue
        add_synset(syn, (row.get("tier") or "B").upper(), key_hint=row["key"], zh_hint=row.get("zh", ""),
                   en_hint=row.get("en", ""),
                   aliases_hint=[a for a in row.get("aliases", "").split(";") if a],
                   desc_hint=row.get("description", ""))
    print(f"[vocab] catalog 种子: {len(entries)}", flush=True)

    # 2) Open Images 类名 → WordNet（C 层）
    n_oi = 0
    if args.openimages:
        for name, desc in load_openimages(Path(args.openimages)):
            syns = [s for s in wn.synsets(name.replace(" ", "_"), pos="n") if under_physical(s, phys_cache)]
            if not syns:
                continue
            syn = syns[0]
            if syn.name() in syn_key:
                continue
            if add_synset(syn, "C", en_hint=name, desc_hint=desc):
                n_oi += 1
    print(f"[vocab] Open Images 新增: {n_oi}，累计 {len(entries)}", flush=True)

    # 3) 向下扩展下位词
    n_exp = 0
    for _ in range(args.depth):
        frontier = list(syn_key.keys())
        for synname in frontier:
            if len(entries) >= args.max_entries:
                break
            syn = wn.synset(synname)
            for hypo in syn.hyponyms():
                if len(entries) >= args.max_entries:
                    break
                if hypo.name() in syn_key:
                    continue
                if add_synset(hypo, "C"):
                    n_exp += 1
    print(f"[vocab] WordNet 下位词扩展: {n_exp}，累计 {len(entries)}", flush=True)

    # 4) parent 解析：取最近一个已收录的上位词
    for key, e in entries.items():
        if not e["synset"]:
            continue
        cur = wn.synset(e["synset"])
        for _ in range(12):
            hs = cur.hypernyms()
            if not hs:
                break
            cur = hs[0]
            pk = syn_key.get(cur.name())
            if pk and pk != key:
                e["parent"] = pk
                break

    # 5) 分配 index / id
    order = sorted(entries.values(), key=lambda x: ("ABC".index(x["tier"]), x["key"]))
    counters = Counter()
    for i, e in enumerate(order):
        e["index"] = i
        lo, hi = SEGMENT[e["tier"]]
        counters[e["tier"]] += 1
        e["id"] = lo + counters[e["tier"]] - 1
    tiers = Counter(e["tier"] for e in order)
    zh_cov = sum(1 for e in order if e["zh"])
    with_parent = sum(1 for e in order if e["parent"])

    vocab = {
        "version": "2.1.0",
        "description": "catalog 种子 + Open Images 类名 + WordNet 层级（A 闭集 / B 可渲染 / C 文本零样本）",
        "sources": ["synthetic_asset_catalog", "open_images_v7", "wordnet", "omw"],
        "stats": {"total": len(order), "tiers": dict(tiers),
                  "zh_coverage": zh_cov, "with_parent": with_parent},
        "types": order,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "total": len(order),
        "from_catalog": sum(1 for e in order if e["tier"] in ("A", "B")),
        "from_openimages": n_oi,
        "from_wordnet_expand": n_exp,
        "tiers": dict(tiers),
        "zh_coverage": round(zh_cov / max(len(order), 1), 4),
        "with_parent": round(with_parent / max(len(order), 1), 4),
        "depth": args.depth,
        "out": str(out),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
