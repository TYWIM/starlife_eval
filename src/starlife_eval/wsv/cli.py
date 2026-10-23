"""OpenWSV 命令行入口。

用法::

    # 列 suite（含冻结样本量与数据 sha256）
    python -m starlife_eval.wsv --list-suites

    # 跑 T1（文本 VLM，OpenAI 兼容端点）
    python -m starlife_eval.wsv run --suite spatialmqa \\
        --model qwen3.8-flash --adapter openai \\
        --base-url https://api.example.com/v1 --out reports/openwsv/

    # 冒烟（只跑前 20 题）
    python -m starlife_eval.wsv run --suite spatialmqa_relation \\
        --model qwen3.8-flash --adapter openai \\
        --base-url https://api.example.com/v1 --out reports/openwsv/ --limit 20

    # 结构化世界模型（WSV /predict）
    python -m starlife_eval.wsv run --suite pairs --model starlife-1-wsv-lite \\
        --adapter wsv --api-url http://127.0.0.1:8890 --out reports/openwsv/ --limit 2

密钥只从**环境变量**或一次性命令行参数读入内存，绝不写盘、不写日志、不写结果
JSON：默认读 ``OPENWSV_API_KEY``（回退 ``WSV_API_KEY`` / ``V2_API_KEY``），
也可用 ``--api-key-env NAME`` 指定变量名；``--api-key`` 仅用于本地临时调试。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from .adapters import api_key_from_env
from .prompts import PROTOCOLS
from .runner import run_suite
from .tasks import SUITES, describe_suites

__all__ = ["build_parser", "main"]

_KEY_ENVS = ("OPENWSV_API_KEY", "WSV_API_KEY", "V2_API_KEY")

_ADAPTER_CHOICES = (
    "openai", "http", "vlm", "chat",
    "wsv", "api", "wsv_api", "wsv-api", "structured",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m starlife_eval.wsv",
        description="OpenWSV frozen-framework evaluation runner (T1 text VLM / T4-T6 structured).",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="列出冻结 suite（样本量 + 数据 sha256）后退出",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="跑一套 suite 并写出结果 JSON")
    run.add_argument("--suite", required=True, choices=SUITES, help="冻结 suite 名")
    run.add_argument("--model", required=True, help="模型 id（写入结果 JSON 的 model.id）")
    run.add_argument("--adapter", default=None, choices=_ADAPTER_CHOICES,
                     help="openai/http/vlm（文本 VLM）或 wsv/api/structured（结构化服务）")
    run.add_argument("--base-url", default=None,
                     help="OpenAI 兼容端点根 URL（文本 VLM），如 https://api.example.com/v1")
    run.add_argument("--api-url", default=None, help="WSV 服务根 URL（结构化；缺省用 --base-url）")
    run.add_argument("--api-key", default=None,
                     help="明文密钥（仅内存，仅本地调试；正式评测请用环境变量）")
    run.add_argument("--api-key-env", default=None,
                     help="读取密钥的环境变量名（默认 OPENWSV_API_KEY）")
    run.add_argument("--out", required=True, help="结果输出目录，如 reports/openwsv/")
    run.add_argument("--limit", type=int, default=None, help="只跑前 N 条任务（冒烟）")
    run.add_argument("--data-root", default=None,
                     help="图像相对路径根目录（默认环境变量 OPENWSV_DATA_ROOT 或 .）")
    run.add_argument("--protocol", default=None, choices=PROTOCOLS,
                     help="文本提示词协议（默认 choice）")
    run.add_argument("--temperature", type=float, default=0.0, help="采样温度（默认 0.0）")
    run.add_argument("--max-tokens", type=int, default=1024, help="最大输出 token（默认 1024）")
    run.add_argument("--workers", type=int, default=1, help="并发请求数（默认 1，保持稳定）")
    return parser


def _print_suites() -> None:
    rows = describe_suites()
    print(f"{'suite':22s} {'n':>5s}  {'sha256[:16]':16s}  {'file':28s} description")
    for row in rows:
        print(
            f"{row['suite']:22s} {row['n']:5d}  {row['sha256'][:16]:16s}  "
            f"{row['file']:28s} {row['description']}"
        )


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return str(args.api_key)
    env_name = args.api_key_env or "OPENWSV_API_KEY"
    names = (env_name,) + tuple(n for n in _KEY_ENVS if n != env_name)
    return api_key_from_env(*names)


def _cmd_run(args: argparse.Namespace) -> int:
    adapter = args.adapter or ("wsv" if args.suite == "pairs" else "openai")
    spec: dict[str, Any] = {
        "id": args.model,
        "adapter": adapter,
        "api_key": _resolve_api_key(args),
        "data_root": args.data_root,
    }
    if adapter in ("wsv", "api", "wsv_api", "wsv-api", "structured"):
        spec["api_url"] = args.api_url or args.base_url
    else:
        spec["base_url"] = args.base_url
        spec["model"] = args.model
        spec["temperature"] = args.temperature
        spec["max_tokens"] = args.max_tokens
        spec["workers"] = args.workers
        if args.protocol:
            spec["protocol"] = args.protocol

    results = run_suite(args.suite, [spec], out_dir=args.out, limit=args.limit,
                        data_root=args.data_root)
    print(json.dumps(
        {
            "ok": True,
            "suite": args.suite,
            "n": len(results[0]["per_task"]) if results else 0,
            "out_dir": os.fspath(args.out),
        },
        ensure_ascii=False,
    ))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.list_suites:
        _print_suites()
        return 0
    if args.command == "run":
        try:
            return _cmd_run(args)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"[openwsv] 错误：{exc}", file=sys.stderr)
            return 2
    parser.print_help()
    return 1