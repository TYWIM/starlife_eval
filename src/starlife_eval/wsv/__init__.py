"""OpenWSV 冻结评测框架。

公开 API：

* :func:`load_suite` — 加载冻结 suite（``spatialmqa`` / ``spatialmqa_relation``
  / ``pairs``），返回规范化任务 dict 列表；
* :func:`run_suite` — 在冻结框架下跑一套 suite × 一个/多个模型，写出稳定契约的
  结果 JSON。

命令行：``python -m starlife_eval.wsv --list-suites`` /
``python -m starlife_eval.wsv run --suite spatialmqa ...``（见 ``cli.py``）。
"""

from __future__ import annotations

from .prompts import PROMPT_VERSION
from .runner import BOOTSTRAP_SEED, FRAMEWORK_VERSION, SAMPLE_SEED, run_suite
from .tasks import SUITES, load_suite

__all__ = [
    "load_suite",
    "run_suite",
    "SUITES",
    "PROMPT_VERSION",
    "FRAMEWORK_VERSION",
    "SAMPLE_SEED",
    "BOOTSTRAP_SEED",
]

__version__ = "0.1.0"