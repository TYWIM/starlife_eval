"""OpenWSV 薄封装示例：从环境变量读配置，调用 :func:`starlife_eval.wsv.run_suite`。

这是给 CI / 后台任务用的最小脚本；正式评测请直接用 CLI::

    python -m starlife_eval.wsv run --suite spatialmqa --model ... --out ...

环境变量
--------
``OPENWSV_SUITE``          suite 名，默认 ``spatialmqa``
``OPENWSV_MODEL``          模型 id，默认 ``qwen3.8-flash``
``OPENWSV_ADAPTER``        ``openai``（文本 VLM）或 ``wsv``（结构化服务），默认前者
``OPENWSV_BASE_URL``       OpenAI 兼容端点根 URL（文本 VLM 必填）
``OPENWSV_API_URL``        WSV 服务根 URL，默认 ``http://127.0.0.1:8890``
``OPENWSV_API_KEY``        密钥（只从环境读；也可用 ``WSV_API_KEY`` / ``V2_API_KEY``）
``OPENWSV_SERVED_MODEL``   发给服务端的模型名（默认同 ``OPENWSV_MODEL``）
``OPENWSV_OUT``            输出目录，默认 ``reports/openwsv``
``OPENWSV_LIMIT``          只跑前 N 条（冒烟）
``OPENWSV_DATA_ROOT``      图像相对路径根目录
``OPENWSV_WORKERS``        文本模型并发数（默认 1）
``OPENWSV_TEMPERATURE``    采样温度（默认 0）
``OPENWSV_MAX_TOKENS``     最大输出 token（默认 1024）
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 允许在未 pip install 的源码树里直接运行：python scripts/wsv/run_suite.py
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from starlife_eval.wsv import run_suite  # noqa: E402


def main() -> int:
    suite = os.environ.get("OPENWSV_SUITE", "spatialmqa")
    model = os.environ.get("OPENWSV_MODEL", "qwen3.8-flash")
    adapter = os.environ.get("OPENWSV_ADAPTER", "openai")
    api_key = os.environ.get("OPENWSV_API_KEY") or os.environ.get("WSV_API_KEY")
    out_dir = Path(os.environ.get("OPENWSV_OUT", "reports/openwsv"))
    limit_env = os.environ.get("OPENWSV_LIMIT")
    limit = int(limit_env) if limit_env else None

    spec = {"id": model, "adapter": adapter, "api_key": api_key}
    if adapter in ("wsv", "api", "wsv_api", "wsv-api", "structured"):
        spec["api_url"] = os.environ.get("OPENWSV_API_URL", "http://127.0.0.1:8890")
    else:
        spec["base_url"] = os.environ.get("OPENWSV_BASE_URL", "")
        spec["model"] = os.environ.get("OPENWSV_SERVED_MODEL", model)
        spec["temperature"] = float(os.environ.get("OPENWSV_TEMPERATURE", "0"))
        spec["max_tokens"] = int(os.environ.get("OPENWSV_MAX_TOKENS", "1024"))
        spec["workers"] = int(os.environ.get("OPENWSV_WORKERS", "1"))

    results = run_suite(suite, [spec], out_dir=out_dir, limit=limit)
    print(json.dumps(
        {
            "suite": suite,
            "adapter": adapter,
            "model": model,
            "n": results[0]["n"],
            "metrics": results[0]["metrics"],
            "out_dir": str(out_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())