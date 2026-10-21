"""分类头结构与 checkpoint 加载。

结构必须与 ``starlife/starlife_scripts/train_heads.py:49-59 Head`` 完全一致，
否则 ``torch.load`` 的 state_dict 键名/shape 对不上（键名是 ``net.0.*`` ...）；
权重文件格式为 ``{"state_dict", "n_classes", "d_in"}``（train_heads.py:116-118 /
eval_compare_cloud.py:134-138），其中 ``d_in`` 是输入维度的**唯一权威**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

HEAD_NAMES: tuple[str, ...] = ("change", "type", "relation", "rel_pos", "attrs")
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.2


class Head(nn.Module):
    """ported from train_heads.py:49-59 Head（逐层一致，含默认 hidden=512/dropout=0.2）。"""

    def __init__(self, d_in: int, n_out: int, hidden: int = HEAD_HIDDEN, dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


@dataclass(frozen=True)
class HeadCheckpoint:
    """一个已加载的冻结头 + 其 checkpoint 元信息。"""

    name: str
    path: Path
    model: Head
    d_in: int
    n_classes: int
    arch: str | None = None


def _torch_load(path: Path) -> Any:
    """兼容不同 torch 版本的 weights_only 默认值；只读取安全的基本类型。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 旧版 torch 没有 weights_only 参数
        return torch.load(path, map_location="cpu")


def load_head(
    path: str | Path,
    *,
    name: str | None = None,
    expected_d_in: int | None = None,
    expected_n_classes: int | None = None,
) -> HeadCheckpoint:
    """加载单个 ``{name}.pt``，校验字段与（可选的）期望维度。"""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少头权重：{ckpt_path}")
    payload = _torch_load(ckpt_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{ckpt_path}: checkpoint 期望 dict，实际 {type(payload).__name__}")
    for key in ("state_dict", "n_classes", "d_in"):
        if key not in payload:
            raise ValueError(f"{ckpt_path}: checkpoint 缺少字段 {key!r}（期望 state_dict/n_classes/d_in）")
    d_in = int(payload["d_in"])
    n_classes = int(payload["n_classes"])
    if d_in <= 0 or n_classes <= 0:
        raise ValueError(f"{ckpt_path}: d_in={d_in}, n_classes={n_classes} 必须为正")
    if expected_d_in is not None and int(expected_d_in) != d_in:
        raise ValueError(f"{ckpt_path}: d_in={d_in} 与特征维度 {expected_d_in} 不一致")
    if expected_n_classes is not None and int(expected_n_classes) != n_classes:
        raise ValueError(f"{ckpt_path}: n_classes={n_classes} 与期望 {expected_n_classes} 不一致")

    model = Head(d_in, n_classes)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{ckpt_path}: state_dict 与 Head(d_in={d_in}, n_out={n_classes}) 不匹配: {exc}") from exc
    model.eval()

    return HeadCheckpoint(
        name=name or ckpt_path.stem,
        path=ckpt_path,
        model=model,
        d_in=d_in,
        n_classes=n_classes,
        arch=payload.get("arch"),
    )


def load_heads(ckpt_dir: str | Path) -> dict[str, HeadCheckpoint]:
    """加载目录下全部 5 个头（change/type/relation/rel_pos/attrs），缺一个就报错。"""
    directory = Path(ckpt_dir)
    if not directory.exists():
        raise FileNotFoundError(f"缺少 checkpoint 目录：{directory}")
    out: dict[str, HeadCheckpoint] = {}
    for name in HEAD_NAMES:
        out[name] = load_head(directory / f"{name}.pt", name=name)
    return out


def head_features(model: Head, x: "torch.Tensor | Any") -> torch.Tensor:
    """``model.eval()`` + ``torch.no_grad()`` 前向（推理统一入口）。"""
    model.eval()
    with torch.no_grad():
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return model(x)
