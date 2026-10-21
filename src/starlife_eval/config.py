"""评测配置：--data-root 相对路径 + 模型清单。签名由 Lead 维护。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    id: str
    kind: str  # local_heads | replay_cloud | chat_vlm | unavailable
    mode: str = "LOCAL-HEADS"
    family: str = "StarLife"
    provider: str = "StarLife AI"
    provider_color: str | None = None  # None -> 由 report 按 family 取默认色
    # chat_vlm 用
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    # 通用
    source: str | None = None  # 可选的产物相对路径覆盖
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        spec = cls(**kw)
        spec.extra.update(extra)
        return spec


@dataclass
class EvalConfig:
    data_root: Path
    dataset: str = "dataset_spatial_v2"
    features: str = "features/spatial_v2.npz"
    slot_geometry: str | None = None
    checkpoints: str = "checkpoints/pilot_v2g"
    bench: str = "bench/bench_pipeline.json"
    cloud_cache: str = "cloud/eval_cloud_raw.jsonl"
    split: str = "test"
    n_pairs: int = 60
    sample_seed: int = 20261020
    models: list[ModelSpec] = field(default_factory=list)
    out: Path | None = None
    version: str = "v1.1"
    allow_live: bool = False

    # -- 解析后的路径 -------------------------------------------------------
    @property
    def slot_geometry_path(self) -> Path | None:
        return None if self.slot_geometry is None else self.data_root / self.slot_geometry

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / self.dataset

    @property
    def features_path(self) -> Path:
        return self.data_root / self.features

    @property
    def features_meta_path(self) -> Path:
        p = self.features_path
        return p.with_suffix(".meta.json")

    @property
    def checkpoint_dir(self) -> Path:
        return self.data_root / self.checkpoints

    @property
    def bench_path(self) -> Path:
        return self.data_root / self.bench

    @property
    def cloud_cache_path(self) -> Path:
        return self.data_root / self.cloud_cache

    @classmethod
    def from_json(cls, path: Path, data_root: Path | None = None, out: Path | None = None) -> "EvalConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if data_root is not None:
            raw["data_root"] = str(data_root)
        if out is not None:
            raw["out"] = str(out)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalConfig":
        raw = dict(d)
        if "data_root" not in raw:
            raise ValueError("配置缺少 data_root（或命令行 --data-root）")
        root = raw.pop("data_root")
        models = [ModelSpec.from_dict(m) for m in raw.pop("models", [])]
        out = raw.pop("out", None)
        cfg = cls(data_root=Path(root), models=models, out=Path(out) if out else None, **raw)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = {
            "data_root": str(self.data_root),
            "dataset": self.dataset,
            "features": self.features,
            "slot_geometry": self.slot_geometry,
            "checkpoints": self.checkpoints,
            "bench": self.bench,
            "cloud_cache": self.cloud_cache,
            "split": self.split,
            "n_pairs": self.n_pairs,
            "sample_seed": self.sample_seed,
            "version": self.version,
            "allow_live": self.allow_live,
            "models": [m.__dict__ for m in self.models],
        }
        if self.out is not None:
            d["out"] = str(self.out)
        return d

    def require_data(self) -> None:
        needed = [self.dataset_dir]
        if any(m.kind == "local_heads" for m in self.models):
            needed.append(self.features_path)
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "缺少真实产物（--data-root 配置不对？）:\n  " + "\n  ".join(missing)
            )


def env_api_key(spec: ModelSpec) -> str | None:
    """API key 只从环境变量读，绝不落盘、绝不写进仓库。"""
    if not spec.api_key_env:
        return None
    return os.environ.get(spec.api_key_env) or None
