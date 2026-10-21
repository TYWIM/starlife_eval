"""可靠性分析：逐 pair 分解 + 按 pair 重采样 bootstrap + 配对比较。

只读真实产物与真实缓存（不重训、不改判分口径）：

* change   : 逐 slot 4 类混淆（GT 缺省=none），top1 与 compare-macroF1（跳 tp==0 类）
* relation : GT 关系按 (from,to) 匹配，type 命中才算对（分母 = GT 关系数）
* score    : 事件级 StateChanged / RelationAdded 的 F1 平均值（= report.macro_f1_common）

重采样单位是 **pair**（60 对里有放回抽 60 对），所以 CI 已包含"同 pair 内 slot 相关"
带来的方差膨胀——这是这类评测最常被忽略的一点。

用法::

    python tools/reliability.py --data-root <DATA_ROOT> \
        --config configs/full.json --report runs/full_v11/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from starlife_eval.config import EvalConfig
from starlife_eval.dataset.features import load_features
from starlife_eval.dataset.manifest import load_manifest
from starlife_eval.dataset.negative_sampling import build_relation_data
from starlife_eval.models.registry import build_models, execute, live_cache_path
from starlife_eval.verifiers.event_metrics import evaluate_event_metrics
from starlife_eval.verifiers.pair_protocol import local_pair_predictions, sample_pairs

N_BOOT = 2000
BOOT_SEED = 7


def collect(cfg: EvalConfig):
    manifest = load_manifest(cfg.dataset_dir)
    feat = load_features(cfg.features_path)
    rel = build_relation_data(feat)
    sample = sample_pairs(manifest, cfg.split, cfg.n_pairs, cfg.sample_seed)
    pair_map = {p.index: p for p in manifest.pairs}
    preds_by_model: dict[str, dict] = {}
    for prep in build_models(cfg, manifest, feat):
        if prep.kind == "unavailable":
            continue
        cache = live_cache_path(cfg, prep.spec) if prep.kind == "chat_vlm" else None
        run = execute(prep, manifest, sample, dataset_dir=cfg.dataset_dir, live=False, cache_path=cache)
        if run.head is not None:
            preds_by_model[prep.model_id] = local_pair_predictions(manifest, run.head, rel, sample,
                                                                  meta=feat.meta)
        else:
            preds_by_model[prep.model_id] = run.pairs
    return manifest, feat, sample, pair_map, preds_by_model


def per_pair_counts(manifest, feat, sample, pair_map, preds):
    """-> (N,K,K) 混淆, (N,4) 关系计数, {kind: (N,3)} 事件计数。"""
    change_keys = list(feat.meta["change_keys"])
    idx_of = {k: i for i, k in enumerate(change_keys)}
    K, N = len(change_keys), len(sample)
    has_types = any(p.types for p in preds.values())
    conf = np.zeros((N, K, K), dtype=np.int64)
    rs = np.zeros((N, 4), dtype=np.int64)  # hit, wrong_type, miss, extra
    ev: dict[str, np.ndarray] = {}
    for pos, pi in enumerate(sample):
        pair, pred = pair_map[pi], preds[pi]
        for e in pair.entities:
            gt = pair.changes.get(e.slot, "none")
            pr = pred.changes.get(e.slot, "none")
            if gt in idx_of and pr in idx_of:
                conf[pos, idx_of[gt], idx_of[pr]] += 1
        pm = pred.relation_map()
        gt_pairs = {(r.from_slot, r.to_slot): r.type for r in pair.relations}
        for key, rtype in gt_pairs.items():
            if key not in pm:
                rs[pos, 2] += 1
            elif pm[key].type == rtype:
                rs[pos, 0] += 1
            else:
                rs[pos, 1] += 1
        rs[pos, 3] += sum(1 for k in pm if k not in gt_pairs)
        one = evaluate_event_metrics(manifest, preds, split="test", sample_indices=[pi],
                                     pred_has_types=has_types)
        for kind, entry in one["per_kind"].items():
            arr = ev.setdefault(kind, np.zeros((N, 3), dtype=np.int64))
            arr[pos] = [int(entry.get("matched") or 0), int(entry.get("predicted") or 0),
                        int(entry.get("support") or 0)]
    return conf, rs, ev


def metrics(conf, rs, ev):
    conf = np.asarray(conf)
    if conf.ndim == 2:
        conf = conf[None, ...]
    rs = np.asarray(rs)
    if rs.ndim == 1:
        rs = rs[None, :]
    tot = conf.sum()
    top1 = float(np.trace(conf, axis1=1, axis2=2).sum() / tot) if tot else float("nan")
    f1s = []
    for c in range(conf.shape[1]):
        tp = conf[:, c, c].sum()
        fp = conf[:, :, c].sum() - tp
        fn = conf[:, c, :].sum() - tp
        if tp == 0:  # 与 eval_compare_cloud.macro_f1 一致：跳 tp==0 的类
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    macro = float(np.mean(f1s)) if f1s else float("nan")
    gt_rel = rs[:, 0] + rs[:, 1] + rs[:, 2]
    rel_acc = float(rs[:, 0].sum() / gt_rel.sum()) if gt_rel.sum() else float("nan")
    kind_f1 = {}
    for kind, triple in ev.items():
        arr = np.asarray(triple)
        if arr.ndim == 2:
            arr = arr.sum(0)
        m, p, s = int(arr[0]), int(arr[1]), int(arr[2])
        if p + s == 0:
            continue
        kind_f1[kind] = 2 * m / (p + s)
    score = (float(np.mean([kind_f1["StateChanged"], kind_f1["RelationAdded"]]))
             if {"StateChanged", "RelationAdded"} <= set(kind_f1) else float("nan"))
    return dict(top1=top1, macro=macro, rel=rel_acc, score=score,
                gt_rel=int(gt_rel.sum()),
                pred_rel=int(rs[:, 0].sum() + rs[:, 1].sum() + rs[:, 3].sum()),
                prec=float(rs[:, 0].sum() / max(1, rs[:, 0].sum() + rs[:, 1].sum() + rs[:, 3].sum())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--report", default="runs/full_v11/report.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    cfg = EvalConfig.from_json(Path(args.config), data_root=Path(args.data_root))
    manifest, feat, sample, pair_map, preds_by_model = collect(cfg)
    N = len(sample)

    conf_by, rel_by, ev_by = {}, {}, {}
    for mid, preds in preds_by_model.items():
        conf_by[mid], rel_by[mid], ev_by[mid] = per_pair_counts(manifest, feat, sample, pair_map, preds)

    # 0) 自校验：逐 pair 汇总必须等于 report（report 的 pair_protocol 已 round 到 4 位）
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    print("== 自校验（逐 pair 汇总 vs report.json）==")
    bad = 0
    for mid in preds_by_model:
        got = metrics(conf_by[mid], rel_by[mid], ev_by[mid])
        rp = report["models"][mid]
        exp = {"top1": (rp["pair_protocol"] or {}).get("change_top1"),
               "macro": (rp["pair_protocol"] or {}).get("change_macro_f1"),
               "rel": (rp["pair_protocol"] or {}).get("relation_type_acc"),
               "score": (rp["event_metrics"] or {}).get("macro_f1_common")}
        diffs = {k: abs(got[k] - exp[k]) for k in exp if exp[k] is not None}
        ok = all(v < 1e-4 for v in diffs.values())
        bad += 0 if ok else 1
        print(f"  {mid:24s} {'OK' if ok else 'MISMATCH ' + str(diffs)}")
    print("  ->", "全部一致（差异仅为 report 的 4 位小数舍入）" if bad == 0 else f"{bad} 个不一致")

    rng = np.random.default_rng(BOOT_SEED)
    boot = {}
    for mid in preds_by_model:
        conf, rs, ev = conf_by[mid], rel_by[mid], ev_by[mid]
        rows = []
        for _ in range(args.n_boot):
            pick = rng.integers(0, N, N)
            m = metrics(conf[pick].sum(0), rs[pick].sum(0), {k: v[pick].sum(0) for k, v in ev.items()})
            rows.append([m["top1"], m["macro"], m["rel"], m["score"]])
        boot[mid] = np.array(rows)

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return f"{lo:.2f},{hi:.2f}"

    print(f"\n== 点估计与 95% CI（按 pair bootstrap，n={N}，{args.n_boot} 次）==")
    print(f"{'model':24s} {'change_top1 [CI]':>28s} {'rel_type_acc [CI]':>28s} {'score [CI]':>28s}")
    for mid in preds_by_model:
        p = metrics(conf_by[mid], rel_by[mid], ev_by[mid])
        b = boot[mid]
        print(f"{mid:24s} {p['top1']:.3f} [{ci(b[:,0])}]{'':>7s} "
              f"{p['rel']:.3f} [{ci(b[:,2])}]{'':>7s} {p['score']:.3f} [{ci(b[:,3])}]")

    print("\n== 关系输出覆盖度（低 relation_type_acc 的来源）==")
    print(f"{'model':24s} {'GT':>4s} {'pred':>5s} {'coverage':>9s} {'precision':>10s} {'recall':>7s}")
    for mid in preds_by_model:
        m = metrics(conf_by[mid], rel_by[mid], ev_by[mid])
        cov = m["pred_rel"] / max(1, m["gt_rel"])
        print(f"{mid:24s} {m['gt_rel']:4d} {m['pred_rel']:5d} {cov:9.2f} {m['prec']:10.3f} {m['rel']:7.3f}")

    clouds = [m for m in preds_by_model if not m.startswith("starlife")]
    ref = next((m for m in preds_by_model if m == "starlife-1-wsv-lite-v2"), None)
    if ref and clouds:
        print(f"\n== 配对比较：{ref} - cloud（同一批重采样 pair）==")
        print(f"{'vs':24s} {'Δscore mean [95%CI]':>30s} {'P(local>cloud)':>15s} {'Δrel mean [95%CI]':>28s}")
        for mid in clouds:
            ds = boot[ref][:, 3] - boot[mid][:, 3]
            dr = boot[ref][:, 2] - boot[mid][:, 2]
            print(f"{mid:24s} {np.mean(ds):+.3f} [{np.percentile(ds,2.5):+.2f},{np.percentile(ds,97.5):+.2f}]{'':>8s} "
                  f"{(ds>0).mean():15.3f} {np.mean(dr):+.3f} [{np.percentile(dr,2.5):+.2f},{np.percentile(dr,97.5):+.2f}]")

    if len(clouds) > 1:
        print("\n== 云端内部排序稳定性（P(A>B) on score）==")
        for i, a in enumerate(clouds):
            for b in clouds[i + 1:]:
                print(f"  {a:16s} > {b:16s}: P={(boot[a][:,3] > boot[b][:,3]).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
