"""全局测试配置：修 Windows 沙箱下 pytest 内置 tmp_path 建目录后不可访问的问题。

pytest 用 ``mkdir(mode=0o700)`` 造 tmp 目录，在本机 DSH 文件沙箱下会变成 WinError 5
（既不能进入也不能清理）。这里直接覆盖 ``tmp_path_factory`` / ``tmp_path``，
用普通权限目录 + 自增序号，行为对测试透明。
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_counter = itertools.count()


class _Factory:
    def __init__(self, base: Path) -> None:
        self.base = base

    def mktemp(self, name: str, numbered: bool = True) -> Path:  # noqa: ARG002
        d = self.base / f"{name}-{next(_counter)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def getbasetemp(self) -> Path:
        return self.base


@pytest.fixture(scope="session")
def tmp_path_factory() -> _Factory:
    base = REPO / ".pytest_tmp" / f"s{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    return _Factory(base)


@pytest.fixture
def tmp_path(tmp_path_factory: _Factory) -> Path:
    return tmp_path_factory.mktemp("t")
