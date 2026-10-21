"""CLI entry point: python -m starlife_eval.v2 {run,score,compare}"""

from __future__ import annotations

import argparse
import sys

from .runner import cmd_compare, cmd_run, cmd_score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m starlife_eval.v2", description="v2 task-matrix benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="query a model over tasks and score it")
    run.add_argument("--tasks", required=True, help="unified tasks JSONL")
    run.add_argument("--model", required=True, help="model spec JSON (base_url, model, api_key_env)")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    run.add_argument("--model-name", default="", help="display name for the report")
    run.set_defaults(fn=cmd_run)

    score = sub.add_parser("score", help="score existing predictions JSONL")
    score.add_argument("--tasks", required=True)
    score.add_argument("--pred", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--model-name", default="")
    score.set_defaults(fn=cmd_score)

    cmp = sub.add_parser("compare", help="paired McNemar test between two prediction files")
    cmp.add_argument("--tasks", required=True)
    cmp.add_argument("--pred-a", required=True)
    cmp.add_argument("--pred-b", required=True)
    cmp.set_defaults(fn=cmd_compare)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
