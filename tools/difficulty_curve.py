"""按「场景复杂度」分档，重算每档的 7 维综合分（折线图数据源）。

复杂度定义（只用 GT，不看模型）：
    difficulty = 2 × 变化物体数 + 关系对数
把 60 对按 difficulty 升序切成 ``--buckets`` 个等量档（默认 4 档，每档 15 对），
在每档内部用与榜单**完全相同**的 7 维定义重算综合分（只是把 60 对限定为子集）。
动机：折线图的横轴必须是有序量，用它回答「场景越复杂，模型掉多快」。

用法（在服务器/任何有 data-root 的机器上，离线、只读缓存，不调 API）：

    python tools/difficulty_curve.py --config configs/full_v14.json \
        --data-root <DATA_ROOT> --out reports/v1.4/difficulty_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from starlife_eval.config import EvalConfig  # noqa: E402
from starlife_eval.dataset.features import load_features  # noqa: E402
from starlife_eval.dataset.manifest import load_manifest  # noqa: E402
from starlife_eval.dataset.negative_sampling import build_relation_data  # noqa: E402
from starlife_eval.models.registry import build_models, execute  # noqa: E402
from starlife_eval.verifiers.event_metrics import evaluate_event_metrics  # noqa: E402
from starlife_eval.verifiers.pair_protocol import (  # noqa: E402
    evaluate_pair_protocol,
    local_pair_predictions,
    sample_pairs,
)
from starlife_eval.verifiers.state_replay import replay_and_compare  # noqa: E402


def _bucket_indices(sample: list[int], difficulty: dict[int, int], buckets: int) -> list[list[int]]:
    ordered = sorted(sample, key=lambda i: (difficulty[i], i))
    size = len(ordered) / buckets
    out: list[list[int]] = [[] for _ in range(buckets)]
    for pos, idx in enumerate(ordered):
        out[min(buckets - 1, int(pos // size))].append(idx)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", type=int, default=4)
    args = ap.parse_args()

    cfg = EvalConfig.from_json(Path(args.config), data_root=Path(args.data_root))
    manifest = load_manifest(cfg.dataset_dir)
    features = load_features(cfg.features_path)
    meta = features.meta
    rel = build_relation_data(features)
    sample = sample_pairs(manifest, cfg.split, cfg.n_pairs, cfg.sample_seed)
    by_index = {p.index: p for p in manifest.pairs}

    difficulty: dict[int, int] = {}
    changed_count: dict[int, int] = {}
    rel_count: dict[int, int] = {}
    for i in sample:
        pair = by_index[i]
        changed = sum(1 for e in pair.entities if pair.changes.get(e.slot, "none") != "none")
        changed_count[i] = changed
        rel_count[i] = len(pair.relations)
        difficulty[i] = 2 * changed + rel_count[i]

    groups = _bucket_indices(sample, difficulty, args.buckets)
    bucket_meta = []
    for k, g in enumerate(groups, 1):
        bucket_meta.append({
            "key": f"b{k}",
            "label": "档 {0}".format(k),
            "rank": k,
            "n": len(g),
            "difficulty": [min(difficulty[i] for i in g), max(difficulty[i] for i in g)],
            "changed_mean": round(sum(changed_count[i] for i in g) / len(g), 2),
            "relations_mean": round(sum(rel_count[i] for i in g) / len(g), 2),
        })

    prepared = build_models(cfg, manifest, features)
    models_out = []
    for prep in prepared:
        run = execute(prep, manifest, sample, dataset_dir=cfg.dataset_dir, live=False)
        if run.head is not None:
            pairs = local_pair_predictions(manifest, run.head, rel, sample, meta=meta)
        else:
            pairs = run.pairs or {}
        has_types = run.head is not None
        points = []
        for g in groups:
            pp = evaluate_pair_protocol(manifest, g, pairs, strict=False)
            ev = evaluate_event_metrics(manifest, pairs, split=cfg.split, sample_indices=g,
                                        pred_has_types=has_types)
            st = replay_and_compare(manifest, g, pairs, split=cfg.split, pred_has_types=has_types)
            dims = {
                "change_top1": pp["change_top1"],
                "change_balance": pp["change_macro_f1"],
                "relation_type": pp["relation_type_acc"],
                "rel_pos": pp["rel_pos_acc"],
                "attrs": pp["attrs_acc"],
                "event_f1": ev.get("macro_f1_common"),
                "state_consistency": st.get("state_acc_common"),
            }
            values = [v for v in dims.values() if isinstance(v, (int, float))]
            composite = round(100.0 * sum(values) / len(values), 1) if values else None
            points.append({
                "bucket": bucket_meta[len(points)]["key"],
                "composite": composite,
                "dims": {k: (round(100.0 * v, 1) if isinstance(v, (int, float)) else None)
                         for k, v in dims.items()},
            })
        models_out.append({
            "id": prep.model_id,
            "name": prep.model_id,
            "family": prep.family,
            "mode": prep.mode,
            "kind": prep.kind,
            "points": points,
        })

    payload = {
        "schemaVersion": 1,
        "metric": "composite-by-difficulty",
        "generatedFrom": "reports/v1.4/report.json",
        "xAxis": {
            "key": "difficulty",
            "label": "场景复杂度（GT：变化物体数 + 关系对数）",
            "definition": "difficulty = 2 × 变化物体数 + 关系对数；按 difficulty 升序等量分档",
            "buckets": bucket_meta,
        },
        "yAxis": {"key": "composite", "label": "分档综合分", "range": [0, 100]},
        "note": "在每档内部用与榜单完全相同的 7 维定义重算综合分（不是榜单主分，是限定子集的诊断视图）。"
                "档内样本只有 60/4 ≈ 15 对，档间差值在 ±5 分内的不要当结论。",
        "models": models_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, out.stat().st_size, "bytes")
    print("buckets:", [(b["key"], b["n"], b["difficulty"]) for b in bucket_meta])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
