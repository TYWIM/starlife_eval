"""Runner：把真实产物喂给适配器，跑全部 verifier，产出 report。

流程（每一步都只消费真实数据，缺失即 N/A）：
  manifest + features + relation 采样
    → 逐模型适配器（本地头 / 云端回放 / 云端实时）
    → 头级指标（仅本地头）
    → 60 对同批协议
    → 事件级 P/R/F1（L3）
    → 世界状态回放（L4）
    → report.json / report.md / leaderboard.generated.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import EvalConfig
from ..dataset.features import load_features
from ..dataset.manifest import load_manifest
from ..dataset.negative_sampling import build_relation_data
from ..models.local_heads import head_predictions_to_pairs  # noqa: F401  (保留兼容)
from ..models.registry import build_models, execute, resolve_cache_path
from ..verifiers.pair_protocol import local_pair_predictions
from ..report.leaderboard import build_leaderboard_payload, write_leaderboard
from ..report.report_json import ModelEvaluation, build_report, write_report
from ..report.report_md import write_markdown
from ..translator import translate_gt
from ..verifiers.event_metrics import evaluate_event_metrics
from ..verifiers.head_metrics import evaluate_head_metrics
from ..verifiers.pair_protocol import evaluate_pair_protocol, sample_pairs
from ..verifiers.state_replay import replay_and_compare


def _log(msg: str) -> None:
    print(f"[starlife_eval] {msg}", flush=True)


def run_evaluation(cfg: EvalConfig, *, out_dir: Path | None = None, live: bool = False) -> dict[str, Any]:
    t_start = time.time()
    cfg.require_data()
    out = out_dir or cfg.out or Path("runs/latest")
    out.mkdir(parents=True, exist_ok=True)

    _log(f"data_root={cfg.data_root}")
    manifest = load_manifest(cfg.dataset_dir)
    _log(f"manifest: {len(manifest.pairs)} pairs, split={cfg.split} -> {len(manifest.split_indices(cfg.split))} pairs")
    needs_features = any(m.kind == "local_heads" for m in cfg.models)
    features = None
    meta: dict[str, Any] = {}
    rel: dict[str, Any] = {}
    if needs_features:
        features = load_features(cfg.features_path)
        meta = features.meta
        _log(f"features: {cfg.features_path.name}, slots={features['slot_before'].shape[0]}, dim={features['slot_before'].shape[1]}")
        rel = build_relation_data(features)
    else:
        _log("features: not required (no local_heads model in this config)")
    sample = sample_pairs(manifest, cfg.split, cfg.n_pairs, cfg.sample_seed)
    _log(f"sample_pairs: n={len(sample)} seed={cfg.sample_seed}")

    prepared = build_models(cfg, manifest, features)
    _log(f"models: {[p.model_id + ('(unavailable)' if p.kind == 'unavailable' else '') for p in prepared]}")

    gt_events = translate_gt(manifest, split=cfg.split, sample_indices=sample)
    evaluations: list[ModelEvaluation] = []

    for prep in prepared:
        if prep.kind == "unavailable":
            evaluations.append(
                ModelEvaluation(
                    model_id=prep.model_id, kind=prep.kind, mode=prep.mode,
                    family=prep.family, provider=prep.provider,
                    run=prep.empty_run() if hasattr(prep, "empty_run") else _empty_run(prep),
                    na=dict(prep.na or {}),
                )
            )
            _log(f"  - {prep.model_id}: unavailable ({prep.na})")
            continue

        cache_path = resolve_cache_path(cfg, prep.spec) if prep.kind == "chat_vlm" else None
        try:
            run = execute(prep, manifest, sample, dataset_dir=cfg.dataset_dir, live=live, cache_path=cache_path)
        except Exception as exc:  # noqa: BLE001 —— 单个模型失败不许拖垮整份报告
            _log(f"  - {prep.model_id}: FAILED ({type(exc).__name__}: {exc})")
            evaluations.append(
                ModelEvaluation(
                    model_id=prep.model_id, kind=prep.kind, mode=prep.mode, family=prep.family,
                    provider=prep.provider, run=_empty_run(prep), name=prep.model_id,
                    na={"run_error": f"{type(exc).__name__}: {exc}"},
                    notes=["该模型本次运行失败，整行指标 N/A（不填 0、不猜数）"],
                )
            )
            continue
        ev = ModelEvaluation(
            model_id=run.model_id, kind=run.kind, mode=run.mode, family=run.family,
            provider=run.provider, run=run, name=prep.model_id,
            provider_color=getattr(prep.spec, "provider_color", None),
            na=dict(run.na or {}), notes=list(run.notes or []),
        )

        if run.head is not None:
            ev.head_metrics = evaluate_head_metrics(features, meta, run.head, rel, split=cfg.split)
            pairs = local_pair_predictions(manifest, run.head, rel, sample, meta=meta)
            ev.notes.append("本地头：头级指标在全 test split 上计算；60 对协议与云端同批可比。")
        else:
            pairs = run.pairs or {}
            ev.na.setdefault("head_metrics", "云端零样本不输出头级概率分布，头级指标 N/A")
            ev.na.setdefault("type", "云端输出 schema 不含实体类型，type 维度 N/A")

        try:
            ev.pair_protocol = evaluate_pair_protocol(manifest, sample, pairs)
            pred_has_types = run.head is not None
            ev.event_metrics = evaluate_event_metrics(
                manifest, pairs, split=cfg.split, sample_indices=sample, pred_has_types=pred_has_types
            )
            ev.state_replay = replay_and_compare(
                manifest, sample, pairs, split=cfg.split, pred_has_types=pred_has_types
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"  - {run.model_id}: 指标计算失败 ({type(exc).__name__}: {exc})")
            ev.na["metrics_error"] = f"{type(exc).__name__}: {exc}"
            evaluations.append(ev)
            continue
        if ev.state_replay.get("types") is None:
            ev.na.setdefault("state.types", (ev.state_replay.get("na") or {}).get("types", "无类型预测，types 维度 N/A"))
        evaluations.append(ev)
        _log(f"  - {run.model_id}: done ({run.kind}), event_f1={ev.event_metrics.get('macro_f1')}")

    report = build_report(cfg, evaluations)
    if features is not None:
        slots = int(features["slot_before"].shape[0])
        relations = int(len(features["rel_from"]))
    else:
        slots = sum(len(pair.slots) for pair in manifest.pairs)
        relations = sum(len(pair.relations) for pair in manifest.pairs)
    counts = {
        "pairs": len(manifest.pairs),
        "slots": slots,
        "relations": relations,
        "heads": 5 if needs_features else 0,
    }
    report["meta"]["n_test_pairs"] = len(manifest.split_indices(cfg.split))
    report["meta"]["counts"] = counts
    report["meta"]["elapsed_s"] = round(time.time() - t_start, 3)
    report["leaderboard"] = _recompute_leaderboard(report)

    report_path = write_report(report, out)
    md_path = write_markdown(report, out)
    lb_payload = build_leaderboard_payload(
        report, n_test_pairs=report["meta"]["n_test_pairs"], source_run=str(out)
    )
    lb_path = write_leaderboard(lb_payload, out / "leaderboard.generated.json")
    _log(f"wrote: {report_path}  {md_path}  {lb_path}")
    return report


def _empty_run(prep: Any):  # noqa: ANN001
    from ..types import ModelRun

    return ModelRun(model_id=prep.model_id, kind=prep.kind, mode=prep.mode,
                    family=prep.family, provider=prep.provider, na=dict(prep.na or {}))


def _recompute_leaderboard(report: dict[str, Any]) -> list[dict[str, Any]]:
    from ..report.report_json import build_leaderboard_rows

    return build_leaderboard_rows(report)


# ----------------------------------------------------------------- acceptance


def check_acceptance(report: dict[str, Any], acceptance_path: Path, tol: float = 1e-3) -> list[dict[str, Any]]:
    """和验收卡比对（默认 tol=±0.001；latency 这类测量量允许相对容差 tol_rel）。"""
    spec = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    tol_rel = float(spec.get("tolerance_rel", 0.0))
    results: list[dict[str, Any]] = []
    for model_id, checks in spec.get("models", {}).items():
        m = (report.get("models") or {}).get(model_id)
        for path, expected in checks.items():
            actual = _lookup(m, path) if m else None
            if actual is None:
                ok, diff = False, None
            else:
                diff = abs(float(actual) - float(expected))
                allowed = max(tol, tol_rel * abs(float(expected)))
                ok = diff <= allowed
            results.append({
                "model": model_id, "metric": path, "expected": expected,
                "actual": actual, "abs_diff": diff, "ok": bool(ok),
            })
    return results


def _lookup(model: dict[str, Any] | None, path: str) -> Any:
    cur: Any = model
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
