"""给榜单每个指标算 95% bootstrap 置信区间（对 60 对 pair 重采样）。

用途：前端柱状图在柱子末端画浮动区间（点估计 + CI），避免把 74.4 和 71.6 这种
差 2.8 分的名次当成确定结论。

方法：pair-level bootstrap —— 有放回地重抽 60 个 pair（B 次、固定种子），
每次重算 7 个维度和综合分，取 2.5% / 97.5% 分位。重抽时**每个 pair 的贡献按重数累加**
（不能用集合去重，否则区间会假性变窄）。

7 个维度全部由「逐 pair 充分统计量」重算，口径与 `report_json` 完全一致，
并用全样本（权重全 1）对拍官方 `reports/*/report.json`，不一致直接报错。

用法：

    python tools/confidence_intervals.py --config configs/full_v14.json \
        --data-root <DATA_ROOT> --out reports/v1.4/confidence_intervals.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from starlife_eval.config import EvalConfig  # noqa: E402
from starlife_eval.dataset.features import load_features  # noqa: E402
from starlife_eval.dataset.manifest import load_manifest  # noqa: E402
from starlife_eval.dataset.negative_sampling import build_relation_data  # noqa: E402
from starlife_eval.models.registry import build_models, execute  # noqa: E402
from starlife_eval.types import CHANGE_INDEX, CHANGE_LABELS  # noqa: E402
from starlife_eval.verifiers.event_metrics import evaluate_event_metrics  # noqa: E402
from starlife_eval.verifiers.pair_protocol import (  # noqa: E402
    _gt_has_rel_pos,
    local_pair_predictions,
    sample_pairs,
)
from starlife_eval.verifiers.state_replay import replay  # noqa: E402

N_CLASSES = len(CHANGE_LABELS)
# 事件维度只用与官方一致的固定类集合（见 report/report_json.SCORE_KINDS）
SCORE_KINDS = ("StateChanged", "RelationAdded")
# state_acc_common 四维（types 仅本地可测，官方不计入）
STATE_DIMS = ("objects", "changes", "relations", "attrs")
DIM_KEYS = (
    "change_top1",
    "change_balance",
    "relation_type",
    "rel_pos",
    "attrs",
    "event_f1",
    "state_consistency",
)

PCT = 100.0


# ------------------------------------------------------------------ per-pair stats


def pair_change_confusion(pair: Any, pred: Any) -> np.ndarray:
    """逐 pair 的 change 混淆计数，形状 (n_classes, n_classes + 1)。

    最后一列是「非法预测标签」（官方按 -1 计入 top1 错误、且不参与 macro 的类）。
    """
    gt = pair.changes
    pred_changes = {int(k): str(v) for k, v in (getattr(pred, "changes", None) or {}).items()}
    mat = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    for entity in pair.entities:
        slot = entity.slot
        g = CHANGE_INDEX.get(gt.get(slot, "none"), -1)
        p = CHANGE_INDEX.get(pred_changes.get(slot, "none"), -1)
        mat[max(g, 0), p if p >= 0 else N_CLASSES] += 1
    return mat


def pair_relation_counts(pair: Any, pred: Any) -> np.ndarray:
    """逐 pair 的关系计数：[rel, rel_hit, pos, pos_hit, attr, attr_hit]。"""
    pred_relations = getattr(pred, "relations", None) or []
    p_rel = {(int(r.from_slot), int(r.to_slot)): r for r in pred_relations}
    rel_total = rel_hit = pos_total = pos_hit = attr_total = attr_hit = 0
    for rel_idx, g in enumerate(pair.relations):
        cand = p_rel.get((int(g.from_slot), int(g.to_slot)))
        rel_total += 1
        if getattr(cand, "type", None) == g.type:
            rel_hit += 1
            if _gt_has_rel_pos(pair, rel_idx):
                pos_total += 1
                if getattr(cand, "rel_pos", None) == g.rel_pos:
                    pos_hit += 1
            attr_total += 2
            attr_hit += int(bool(getattr(cand, "contact", None)) == bool(g.contact))
            attr_hit += int(bool(getattr(cand, "support", None)) == bool(g.support))
    return np.array([rel_total, rel_hit, pos_total, pos_hit, attr_total, attr_hit], dtype=np.int64)


def pair_event_counts(manifest: Any, preds: Mapping[int, Any], pair_index: int, split: str,
                      pred_has_types: bool) -> np.ndarray:
    """逐 pair 事件计数，2 类 x [support, predicted, matched]。"""
    ev = evaluate_event_metrics(
        manifest, preds, split=split, sample_indices=[pair_index], pred_has_types=pred_has_types
    )
    row = []
    for kind in SCORE_KINDS:
        entry = (ev.get("per_kind") or {}).get(kind) or {}
        row.extend([int(entry.get("support") or 0), int(entry.get("predicted") or 0),
                    int(entry.get("matched") or 0)])
    return np.array(row, dtype=np.int64)


def pair_state_counts(manifest: Any, preds: Mapping[int, Any], pair_index: int, split: str,
                      pred_has_types: bool) -> np.ndarray:
    """逐 pair 状态计数：[objects_ok,total, changes_ok,total, rel_matched,rel_total, attrs_ok,total]。"""
    st = replay(preds, manifest, split=split, sample_indices=[pair_index],
                pred_has_types=pred_has_types)
    ok = st["n_ok"]
    tot = st["n_total"]
    return np.array([ok["objects"], tot["objects"], ok["changes"], tot["changes"],
                     ok["relations"], tot["relations"], ok["attrs"], tot["attrs"]], dtype=np.int64)


# ------------------------------------------------------------------ weighted metrics


def dims_from_stats(change: np.ndarray, rel: np.ndarray, ev: np.ndarray,
                    st: np.ndarray) -> dict[str, float | None]:
    """由（加权后的）充分统计量算 7 个维度，口径与官方一致，返回 0-1 或 None。"""
    change = change.reshape(N_CLASSES, N_CLASSES + 1)
    total = int(change.sum())
    top1 = float(change[np.arange(N_CLASSES), np.arange(N_CLASSES)].sum() / total) if total else None

    f1s = []
    for c in range(N_CLASSES):
        tp = int(change[c, c])
        if tp == 0:
            continue
        fp = int(change[:, c].sum()) - tp
        fn = int(change[c, :].sum()) - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    change_balance = float(np.mean(f1s)) if f1s else 0.0

    rel_total, rel_hit, pos_total, pos_hit, attr_total, attr_hit = (int(x) for x in rel)
    relation_type = rel_hit / rel_total if rel_total else None
    rel_pos = pos_hit / pos_total if pos_total else None
    attrs = attr_hit / attr_total if attr_total else None

    f1s = []
    for k in range(len(SCORE_KINDS)):
        support, predicted, matched = (int(x) for x in ev[3 * k: 3 * k + 3])
        if support + predicted == 0:
            continue
        f1s.append(2.0 * matched / (support + predicted))
    event_f1 = float(np.mean(f1s)) if f1s else None

    o_ok, o_tot, c_ok, c_tot, r_m, r_tot, a_ok, a_tot = (int(x) for x in st)
    values = []
    if o_tot:
        values.append(o_ok / o_tot)
    if c_tot:
        values.append(c_ok / c_tot)
    if r_tot:
        values.append(2.0 * r_m / r_tot)
    if a_tot:
        values.append(a_ok / a_tot)
    state_acc = float(np.mean(values)) if values else None

    return {
        "change_top1": top1,
        "change_balance": change_balance,
        "relation_type": relation_type,
        "rel_pos": rel_pos,
        "attrs": attrs,
        "event_f1": event_f1,
        "state_consistency": state_acc,
    }


def composite_of(dims: Mapping[str, float | None]) -> float | None:
    values = [v for v in dims.values() if v is not None]
    return float(np.mean(values)) if values else None


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--report", default="", help="官方 report.json，用于对拍（可选）")
    ap.add_argument("--tol", type=float, default=0.2,
                    help="对拍容差（0-100 分制；官方数值有 0.1 舍入）")
    args = ap.parse_args()

    cfg = EvalConfig.from_json(Path(args.config), data_root=Path(args.data_root))
    manifest = load_manifest(cfg.dataset_dir)
    features = load_features(cfg.features_path)
    meta = features.meta
    rel = build_relation_data(features)
    sample = sorted(sample_pairs(manifest, cfg.split, cfg.n_pairs, cfg.sample_seed))
    by_index = {p.index: p for p in manifest.pairs}
    n_pairs = len(sample)
    print(f"pairs: {n_pairs}, draws: {args.draws}, seed: {args.seed}")

    report = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))

    rng = np.random.default_rng(args.seed)
    weights = np.zeros((args.draws, n_pairs), dtype=np.float64)
    for b in range(args.draws):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        np.add.at(weights[b], idx, 1.0)

    prepared = build_models(cfg, manifest, features)
    models_out = []
    worst = 0.0
    for prep in prepared:
        run = execute(prep, manifest, sample, dataset_dir=cfg.dataset_dir, live=False)
        if run.head is not None:
            preds = local_pair_predictions(manifest, run.head, rel, sample, meta=meta)
        else:
            preds = run.pairs or {}
        has_types = run.head is not None

        change_mats, rel_rows, ev_rows, st_rows = [], [], [], []
        for i in sample:
            pair = by_index[i]
            pred = preds.get(i)
            change_mats.append(pair_change_confusion(pair, pred))
            rel_rows.append(pair_relation_counts(pair, pred))
            ev_rows.append(pair_event_counts(manifest, preds, i, cfg.split, has_types))
            st_rows.append(pair_state_counts(manifest, preds, i, cfg.split, has_types))

        change_mat = np.stack(change_mats).reshape(n_pairs, -1).astype(np.float64)   # (60, 20)
        rel_mat = np.stack(rel_rows).astype(np.float64)                              # (60, 6)
        ev_mat = np.stack(ev_rows).astype(np.float64)                                # (60, 6)
        st_mat = np.stack(st_rows).astype(np.float64)                                # (60, 8)

        # ---------------- 全样本对拍（权重全 1） ----------------
        exact = dims_from_stats(change_mat.sum(axis=0), rel_mat.sum(axis=0),
                                ev_mat.sum(axis=0), st_mat.sum(axis=0))
        exact_comp = composite_of(exact)
        if report is not None:
            entry = (report.get("models") or {}).get(prep.model_id) or {}
            dims = {d["key"]: d["value"] for d in ((entry.get("composite") or {}).get("dims") or [])
                    if d.get("available")}
            for key in DIM_KEYS:
                want = dims.get(key)
                got = exact.get(key)
                if want is None or got is None:
                    continue
                diff = abs(want - got * PCT)
                worst = max(worst, diff)
                if diff > args.tol:
                    raise SystemExit(f"对拍失败 {prep.model_id}.{key}: 官方 {want} vs 本工具 {got * PCT:.4f}")
            want_comp = (entry.get("composite") or {}).get("score")
            if want_comp is not None and exact_comp is not None:
                diff = abs(want_comp - exact_comp * PCT)
                worst = max(worst, diff)
                if diff > args.tol:
                    raise SystemExit(f"对拍失败 {prep.model_id}.composite: 官方 {want_comp} vs {exact_comp * PCT:.4f}")

        # ---------------- bootstrap ----------------
        draws_dims: dict[str, list[float | None]] = {k: [] for k in DIM_KEYS}
        draws_comp: list[float | None] = []
        change_sums = weights @ change_mat
        rel_sums = weights @ rel_mat
        ev_sums = weights @ ev_mat
        st_sums = weights @ st_mat
        for b in range(args.draws):
            dims = dims_from_stats(change_sums[b], rel_sums[b], ev_sums[b], st_sums[b])
            for key in DIM_KEYS:
                draws_dims[key].append(dims[key])
            draws_comp.append(composite_of(dims))

        def interval(values: list[float | None], exact_value: float | None) -> dict[str, Any]:
            arr = np.array([v for v in values if v is not None], dtype=np.float64) * PCT
            if arr.size == 0 or exact_value is None:
                return {"value": None, "lo": None, "hi": None}
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return {"value": round(exact_value * PCT, 2), "lo": round(float(lo), 2),
                    "hi": round(float(hi), 2)}

        intervals = {key: interval(draws_dims[key], exact[key]) for key in DIM_KEYS}
        intervals["composite"] = interval(draws_comp, exact_comp)
        models_out.append({
            "id": prep.model_id,
            "family": prep.family,
            "compositeScore": intervals["composite"]["value"],
            "intervals": intervals,
        })
        print(f"  {prep.model_id:26s} composite {intervals['composite']['value']:6.2f} "
              f"[{intervals['composite']['lo']:6.2f}, {intervals['composite']['hi']:6.2f}]")

    payload = {
        "schemaVersion": 1,
        "method": (f"pair-level bootstrap：对 {n_pairs} 对 pair 有放回重抽 {args.draws} 次，"
                   "每次按重数累加逐 pair 充分统计量并重算 7 维与综合分；区间为 2.5%/97.5% 分位（95%）。"),
        "draws": args.draws,
        "seed": args.seed,
        "nPairs": n_pairs,
        "metricKeys": list(DIM_KEYS) + ["composite"],
        "metricLabels": {
            "composite": "综合分",
            "change_top1": "变化识别",
            "change_balance": "变化平衡",
            "relation_type": "关系类型",
            "rel_pos": "空间方位",
            "attrs": "接触支撑",
            "event_f1": "事件F1",
            "state_consistency": "状态一致性",
        },
        "generatedFrom": "reports/v1.4/report.json",
        "note": "区间只反映 60 对抽样误差，不含模型/网关波动；两个模型的区间重叠时不要宣称名次差。",
        "models": models_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} {out.stat().st_size} bytes; worst report diff = {worst:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
