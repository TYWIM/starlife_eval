"""Structured-input track: score a StarLife-1-WSV service on the 60-pair protocol.

The service consumes ``before``/``after`` frames plus detector boxes and returns structured
events, so it cannot answer the text multiple-choice tasks of the v2 text track. This script runs
the published 60-pair structured protocol against the service and reports the same metric
definitions as ``starlife_eval.verifiers.pair_protocol``.

Two runs per pair
-----------------
The service enumerates relation pairs as ``combinations(slots, 2)`` (ascending slot order only),
so a GT relation whose direction runs against that order cannot be expressed in a single run.
Each pair is therefore sent twice, once with the detections in manifest order and once reversed;
for every GT relation the run whose orientation matches the GT direction is used. GT direction
selects an input ordering only — labels are never fed to the model.

Usage
-----
    export WSV_API_URL=http://127.0.0.1:8890     # e.g. an ssh -L tunnel to the service
    export WSV_API_KEY=...
    export DATASET_DIR=/path/to/dataset_spatial_v3
    export OUT_DIR=./structured_track_out
    python run_structured_track.py

Requires the dataset (`dataset_spatial_v3`, GitHub release v1.5) and the frames of the sampled
pairs; missing frames are reported before any request is sent.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import random
import sys

import numpy as np

try:  # the verifiers ship with the package
    from starlife_eval.dataset.manifest import load_manifest
    from starlife_eval.types import CHANGE_INDEX, CHANGE_LABELS
    from starlife_eval.verifiers.metrics import macro_f1_compare
    from starlife_eval.verifiers.pair_protocol import sample_pairs
except ImportError as exc:  # pragma: no cover - environment guidance
    raise SystemExit(
        "starlife_eval is not importable; install it (pip install -e .) or add src/ to PYTHONPATH"
    ) from exc

API = os.environ.get("WSV_API_URL", "http://127.0.0.1:8890").rstrip("/")
KEY = os.environ.get("WSV_API_KEY", "")
DATASET = pathlib.Path(os.environ.get("DATASET_DIR", "./dataset_spatial_v3"))
OUT = pathlib.Path(os.environ.get("OUT_DIR", "./structured_track_out"))
SPLIT, N_PAIRS, SAMPLE_SEED, BOOT_SEED = "test", 60, 20261020, 20261022
N_BOOT = 1000

import urllib.request  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def detections(pair, *, reverse: bool) -> dict:
    entities = list(reversed(pair.entities)) if reverse else list(pair.entities)
    return {
        "width": 1024,
        "height": 768,
        "before": [{"id": f"slot{e.slot}", "bbox": [float(v) for v in e.bbox]} for e in entities],
        "after": [{"id": f"slot{e.slot}", "bbox": [float(v) for v in e.bbox]}
                  for e in entities if e.present_after],
    }


def call(pair, *, reverse: bool) -> dict:
    body = {"before": b64(DATASET / pair.before), "after": b64(DATASET / pair.after),
            "detections": detections(pair, reverse=reverse)}
    req = urllib.request.Request(
        API + "/predict", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"X-API-Key": KEY} if KEY else {})})
    with OPENER.open(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def sid(value) -> int:
    s = str(value)
    return int(s[4:]) if s.startswith("slot") and s[4:].isdigit() else int(s)


def pred_map(resp: dict) -> dict[tuple[int, int], dict]:
    return {(sid(p["from"]), sid(p["to"])): p for p in resp.get("pairs", [])}


def pred_slots(resp: dict) -> dict[int, dict]:
    return {sid(s.get("id", i)): s for i, s in enumerate(resp.get("slots", []))}


def score_pair(pair, fwd: dict, rev: dict) -> dict:
    slots = pred_slots(fwd)
    ch_true, ch_pred = [], []
    for e in pair.entities:
        ch_true.append(pair.changes.get(e.slot, "none"))
        ch_pred.append(str(slots.get(e.slot, {}).get("change", "none")))
    p_fwd, p_rev = pred_map(fwd), pred_map(rev)
    rel = []
    for g in pair.relations:
        key = (g.from_slot, g.to_slot)
        cand = p_fwd.get(key) if g.from_slot < g.to_slot else p_rev.get(key)
        if cand is None:
            cand = p_fwd.get(key) or p_rev.get(tuple(sorted(key)))
        rel.append((bool(cand and cand.get("relation") == g.type),
                    bool(cand and g.rel_pos is not None and cand.get("rel_pos") == g.rel_pos),
                    g.rel_pos is not None,
                    cand, g.contact, g.support))
    return {"ch_true": ch_true, "ch_pred": ch_pred, "rel": rel}


def aggregate(rows: list[dict]) -> dict:
    ch_true = [x for r in rows for x in r["ch_true"]]
    ch_pred = [x for r in rows for x in r["ch_pred"]]
    t_idx = [CHANGE_INDEX.get(x, -1) for x in ch_true]
    p_idx = [CHANGE_INDEX.get(x, -1) for x in ch_pred]
    rel_total = sum(len(r["rel"]) for r in rows)
    rel_hit = sum(1 for r in rows for h, *_ in r["rel"] if h)
    pos_tot = sum(1 for r in rows for h, ph, has, *_ in r["rel"] if h and has)
    pos_hit = sum(1 for r in rows for h, ph, has, *_ in r["rel"] if h and has and ph)
    attr_tot = attr_hit = 0
    for r in rows:
        for h, ph, has, cand, g_contact, g_support in r["rel"]:
            if h and cand is not None:
                attr_tot += 2
                attr_hit += int(bool(cand.get("contact")) == bool(g_contact))
                attr_hit += int(bool(cand.get("support")) == bool(g_support))
    return {
        "change_slots": len(ch_true),
        "change_top1": round(float(np.mean([a == b for a, b in zip(ch_true, ch_pred)])), 4),
        "change_macro_f1": round(macro_f1_compare(t_idx, p_idx, len(CHANGE_LABELS)), 4),
        "relation_pairs": rel_total,
        "relation_type_acc": round(rel_hit / rel_total, 4) if rel_total else 0.0,
        "rel_pos_acc": round(pos_hit / pos_tot, 4) if pos_tot else None,
        "rel_pos_n": pos_tot,
        "attrs_acc": round(attr_hit / attr_tot, 4) if attr_tot else None,
        "attrs_n": attr_tot // 2,
    }


def ci95(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]


def main() -> int:
    manifest = load_manifest(DATASET)
    idx = sample_pairs(manifest, split=SPLIT, n=N_PAIRS, seed=SAMPLE_SEED)[:N_PAIRS]
    by_index = {p.index: p for p in manifest.pairs}
    missing = [(by_index[i].pair_id, str(path)) for i in idx for path in
               (DATASET / by_index[i].before, DATASET / by_index[i].after) if not path.exists()]
    if missing:
        print("missing frames for:", missing[:5], f"... ({len(missing)} files)")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    fwd, rev = {}, {}
    for k, i in enumerate(idx, 1):
        pair = by_index[i]
        fwd[pair.pair_id] = call(pair, reverse=False)
        rev[pair.pair_id] = call(pair, reverse=True)
        if k % 10 == 0:
            print(f"  {k}/{len(idx)} pairs", flush=True)
    (OUT / "raw_forward.json").write_text(json.dumps(fwd, ensure_ascii=False), encoding="utf-8")
    (OUT / "raw_reversed.json").write_text(json.dumps(rev, ensure_ascii=False), encoding="utf-8")

    rows = [score_pair(by_index[i], fwd[by_index[i].pair_id], rev[by_index[i].pair_id]) for i in idx]
    point = aggregate(rows)

    rng = random.Random(BOOT_SEED)
    boot = {k: [] for k in ("change_top1", "change_macro_f1", "relation_type_acc", "attrs_acc")}
    pos_boot = []
    for _ in range(N_BOOT):
        a = aggregate([rows[rng.randrange(len(rows))] for _ in rows])
        for k in boot:
            if a[k] is not None:
                boot[k].append(a[k])
        if a["rel_pos_acc"] is not None:
            pos_boot.append(a["rel_pos_acc"])

    result = {
        "protocol": "v2-structured-track",
        "dataset": {"name": DATASET.name, "split": SPLIT, "sample_seed": SAMPLE_SEED, "pairs": len(idx)},
        "service": {"base_url": API, "model": (fwd.get(by_index[idx[0]].pair_id) or {}).get("model"),
                    "version": (fwd.get(by_index[idx[0]].pair_id) or {}).get("version")},
        "input_mode": "structured: before/after frames + GT detections (service runs no detector)",
        "orientation": "forward + reversed slot orderings, one per GT direction (2 calls per pair)",
        "metrics": point,
        "ci95": {k: ci95(v) for k, v in boot.items()},
        "rel_pos_ci95": ci95(pos_boot) if pos_boot else None,
        "n_pairs": len(rows),
    }
    (OUT / "structured_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"metrics": point, "ci95": result["ci95"],
                      "rel_pos_ci95": result["rel_pos_ci95"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
