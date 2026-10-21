"""starlife_eval CLI：唯一真后端的入口。

用法：
  python -m starlife_eval run   --config configs/v2g_real.json --data-root <dir> --out runs/v2g_real [--live]
  python -m starlife_eval leaderboard --run runs/v2g_real --out <openwsv>/src/data/leaderboard.generated.json
  python -m starlife_eval acceptance   --run runs/v2g_real [--spec configs/acceptance.json] [--tol 0.001]
  python -m starlife_eval validate     --config configs/v2g_real.json --data-root <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import EvalConfig


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="starlife_eval", description="StarLife-1-WSV 真实评测后端")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="跑完整评测，输出 report.json/report.md/leaderboard.generated.json")
    run.add_argument("--config", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--out", default=None)
    run.add_argument("--live", action="store_true", help="允许云端模型实时调用 API（key 只读环境变量）")
    run.add_argument("--n-pairs", type=int, default=None, help="覆盖 60 对抽样数")
    run.add_argument("--split", default=None)
    run.add_argument("--only", default=None, help="只跑逗号分隔的 model id")

    lb = sub.add_parser("leaderboard", help="从 report.json 生成前端契约文件")
    lb.add_argument("--run", required=True, help="runs/<name> 目录（含 report.json）")
    lb.add_argument("--out", required=True, help="目标 leaderboard.generated.json 路径")

    acc = sub.add_parser("acceptance", help="和验收卡逐项比对")
    acc.add_argument("--run", required=True)
    acc.add_argument("--spec", default=None)
    acc.add_argument("--tol", type=float, default=1e-3)
    acc.add_argument("--json", action="store_true")

    val = sub.add_parser("validate", help="只检查产物是否存在、形状是否自洽（不跑模型）")
    val.add_argument("--config", required=True)
    val.add_argument("--data-root", required=True)

    return ap


def _load_cfg(args: argparse.Namespace) -> EvalConfig:
    cfg = EvalConfig.from_json(Path(args.config), data_root=Path(args.data_root),
                               out=Path(args.out) if getattr(args, "out", None) else None)
    if getattr(args, "n_pairs", None):
        cfg.n_pairs = args.n_pairs
    if getattr(args, "split", None):
        cfg.split = args.split
    if getattr(args, "live", False):
        cfg.allow_live = True
    only = getattr(args, "only", None)
    if only:
        keep = {s.strip() for s in only.split(",") if s.strip()}
        cfg.models = [m for m in cfg.models if m.id in keep]
        if not cfg.models:
            raise SystemExit(f"--only {only} 没有匹配到任何模型")
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from .runner.executor import run_evaluation

    cfg = _load_cfg(args)
    report = run_evaluation(cfg, out_dir=Path(args.out) if args.out else None, live=cfg.allow_live)
    print(json.dumps({"ok": True, "models": list(report["models"]), "leaderboard": report["leaderboard"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .report.leaderboard import build_leaderboard_payload, write_leaderboard
    from .report.composite import add_composite_score
    from .report.report_json import add_common_score, add_pipeline_latency, build_leaderboard_rows

    run_dir = Path(args.run)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    # 从 models 重新推导榜单（保证 isBest / score 口径与当前代码一致，不信任旧文件里的缓存行）
    add_common_score(report)
    add_pipeline_latency(report)
    add_composite_score(report)
    report["leaderboard"] = build_leaderboard_rows(report)
    payload = build_leaderboard_payload(
        report,
        n_test_pairs=(report.get("meta") or {}).get("n_test_pairs"),
        source_run=str(run_dir),
    )
    path = write_leaderboard(payload, Path(args.out))
    print(json.dumps({"ok": True, "path": str(path), "models": len(payload["models"])}, ensure_ascii=False))
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    from .runner.executor import check_acceptance

    run_dir = Path(args.run)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    spec = Path(args.spec) if args.spec else Path(__file__).resolve().parents[2] / "configs" / "acceptance.json"
    results = check_acceptance(report, spec, tol=args.tol)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            status = "PASS" if r["ok"] else "FAIL"
            actual = "None" if r["actual"] is None else f"{float(r['actual']):.6f}"
            diff = "" if r["abs_diff"] is None else f" diff={r['abs_diff']:.2e}"
            print(f"[{status}] {r['model']:34s} {r['metric']:34s} expected={r['expected']:.6f} actual={actual}{diff}")
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed (tol={args.tol})")
    return 0 if not failed else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from .dataset.features import load_features
    from .dataset.manifest import load_manifest

    cfg = _load_cfg(args)
    cfg.require_data()
    manifest = load_manifest(cfg.dataset_dir)
    features = load_features(cfg.features_path)
    slots = features["slot_before"].shape[0]
    offsets = manifest.slot_offsets
    ok = offsets[-1] == slots
    print(json.dumps({
        "ok": ok,
        "pairs": len(manifest.pairs),
        "splits": {s: len(manifest.split_indices(s)) for s in ("train", "val", "test")},
        "slots": slots,
        "slot_offsets_total": offsets[-1],
        "feature_dim": int(features["slot_before"].shape[1]),
        "relations": int(len(features["rel_from"])),
        "checkpoints": sorted(p.name for p in cfg.checkpoint_dir.glob("*.pt")) if cfg.checkpoint_dir.exists() else [],
        "bench": cfg.bench_path.exists(),
        "cloud_cache": cfg.cloud_cache_path.exists(),
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return {"run": cmd_run, "leaderboard": cmd_leaderboard, "acceptance": cmd_acceptance,
            "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
