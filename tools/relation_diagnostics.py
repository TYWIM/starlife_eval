"""关系维度诊断：云端低 relation_type_acc 是"槽位索引错位"还是"语义错"？

对每个云端模型在 60 对同批采样上统计预测关系，并把"多预测"的槽对按
反转 (to,from) / 槽位 ±1 / ±2 尝试对齐 GT；同时给出**随机对照基线**——
6 个槽只有 30 个有序槽对，误配率天然不低，不对照就会过度解读。

用法::

    python tools/relation_diagnostics.py --data-root <DATA_ROOT> --config configs/full.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from starlife_eval.config import EvalConfig
from starlife_eval.dataset.features import load_features
from starlife_eval.dataset.manifest import load_manifest
from starlife_eval.models.registry import build_models, execute, live_cache_path
from starlife_eval.verifiers.pair_protocol import sample_pairs, local_pair_predictions
from starlife_eval.dataset.negative_sampling import build_relation_data


def align(kind: str, a: int, b: int, gt_slots: set[tuple[int, int]]) -> int:
    if kind == "swap":
        return int((b, a) in gt_slots)
    if kind == "shift1":
        cand = [(a + da, b + db) for da in (-1, 1) for db in (-1, 1)]
        cand += [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]
        return int(any(c in gt_slots for c in cand))
    cand = [(a + da, b + db) for da in (-2, 2) for db in (-2, 2)]
    return int(any(c in gt_slots for c in cand))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--include-local", action="store_true", help="同时诊断本地头（便于对照）")
    args = ap.parse_args()

    cfg = EvalConfig.from_json(Path(args.config), data_root=Path(args.data_root))
    manifest = load_manifest(cfg.dataset_dir)
    feat = load_features(cfg.features_path)
    rel = build_relation_data(feat)
    sample = sample_pairs(manifest, cfg.split, cfg.n_pairs, cfg.sample_seed)
    pair_map = {p.index: p for p in manifest.pairs}
    prepared = [p for p in build_models(cfg, manifest, feat) if p.kind != "unavailable"]
    if not args.include_local:
        prepared = [p for p in prepared if p.kind == "chat_vlm"]
    rng = random.Random(7)

    print(f"{'model':22s} {'GT':>4s} {'pred':>5s} {'hit':>4s} {'wrongT':>7s} {'miss':>5s} {'extra':>6s} | "
          f"{'swap':>5s} {'±1':>5s} {'±2':>5s} {'unexpl':>6s} | {'ctrl swap':>9s} {'ctrl ±1':>7s} {'ctrl ±2':>7s}")
    for prep in prepared:
        cache = live_cache_path(cfg, prep.spec) if prep.kind == "chat_vlm" else None
        run = execute(prep, manifest, sample, dataset_dir=cfg.dataset_dir, live=False, cache_path=cache)
        pairs = (local_pair_predictions(manifest, run.head, rel, sample, meta=feat.meta)
                 if run.head is not None else run.pairs)
        gt_n = hit = wrong = miss = extra = n_pred = 0
        cnt = {"swap": 0, "shift1": 0, "shift2": 0}
        ctrl = {"swap": 0, "shift1": 0, "shift2": 0}
        for pi in sample:
            pair, pred = pair_map[pi], pairs[pi]
            pm = pred.relation_map()
            n_pred += len(pm)
            gt_pairs = {(r.from_slot, r.to_slot): r.type for r in pair.relations}
            gt_slots = set(gt_pairs)
            gt_n += len(gt_pairs)
            for key, rtype in gt_pairs.items():
                if key not in pm:
                    miss += 1
                elif pm[key].type == rtype:
                    hit += 1
                else:
                    wrong += 1
            extras = [k for k in pm if k not in gt_slots]
            extra += len(extras)
            for a, b in extras:
                for kind in cnt:
                    cnt[kind] += align(kind, a, b, gt_slots)
            slots = list(pair.slots)
            pool = [(a, b) for a in slots for b in slots if a != b and (a, b) not in gt_slots]
            if pool and extras:
                for a, b in rng.sample(pool, min(len(extras), len(pool))):
                    for kind in ctrl:
                        ctrl[kind] += align(kind, a, b, gt_slots)
        unexpl = extra - cnt["swap"] - cnt["shift1"] - cnt["shift2"]
        print(f"{prep.model_id:22s} {gt_n:4d} {n_pred:5d} {hit:4d} {wrong:7d} {miss:5d} {extra:6d} | "
              f"{cnt['swap']:5d} {cnt['shift1']:5d} {cnt['shift2']:5d} {unexpl:6d} | "
              f"{ctrl['swap']:9d} {ctrl['shift1']:7d} {ctrl['shift2']:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
