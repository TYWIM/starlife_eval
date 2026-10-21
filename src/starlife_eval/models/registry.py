"""模型注册表：按 :class:`ModelSpec.kind` 造适配器，跑不了就 ``unavailable``。

职责边界：
- **不计算任何指标**（指标在 ``verifiers/``），只负责“造适配器 + 执行预测 + 汇总 runtime/na”；
- **禁止造数**：无缓存 / 未 ``--live`` / 无 API key / 缺 checkpoint 时，
  返回 ``kind="unavailable"`` + 明确 ``na`` 原因，由 report 原样输出 ``N/A``；
- 执行期真实错误（如回放缺 pair、图片缺失）直接抛出，不静默降级成空预测。

``na`` 的键约定：``local_heads`` / ``cloud_cache`` / ``live`` / ``api_key_env`` / ``kind``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import EvalConfig, ModelSpec
from ..dataset.features import load_features
from ..types import DatasetManifest, FeatureBundle, ModelRun

UNAVAILABLE = "unavailable"


@dataclass
class PreparedModel:
    """一个待评测模型：可用时持 adapter，不可用时持 na 原因。"""

    model_id: str
    kind: str
    mode: str
    family: str
    provider: str
    spec: ModelSpec
    adapter: Any | None = None
    na: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.adapter is not None and self.kind != UNAVAILABLE


def live_cache_path(cfg: EvalConfig, spec: ModelSpec) -> Path:
    """云端 live 缓存默认路径：``<data-root>/cloud/live_<model-id>.jsonl``。"""
    return cfg.data_root / "cloud" / f"live_{spec.id}.jsonl"


def _resolve(cfg: EvalConfig, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else cfg.data_root / p


def _asset_path(cfg: EvalConfig, spec: ModelSpec, default: Path) -> Path:
    return _resolve(cfg, spec.source) if spec.source else default


def _has_live_key(spec: ModelSpec) -> bool:
    return bool(spec.api_key_env and os.environ.get(spec.api_key_env))


def _unavailable(spec: ModelSpec, reason_key: str, reason: str, **extra: str) -> PreparedModel:
    na = {reason_key: reason}
    na.update(extra)
    return PreparedModel(
        model_id=spec.id,
        kind=UNAVAILABLE,
        mode=spec.mode,
        family=spec.family,
        provider=spec.provider,
        spec=spec,
        adapter=None,
        na=na,
    )


def resolve_cache_path(cfg: EvalConfig, spec: ModelSpec) -> Path:
    """live 缓存路径：extra.cache_path > spec.source（历史写法） > 默认 cloud/live_<id>.jsonl。"""
    override = (spec.extra or {}).get("cache_path") or spec.source
    return _resolve(cfg, str(override)) if override else Path(live_cache_path(cfg, spec))


def build_model(
    spec: ModelSpec,
    cfg: EvalConfig,
    manifest: DatasetManifest,
    features: FeatureBundle | None = None,
    *,
    live: bool | None = None,
) -> PreparedModel:
    """按 kind 造单个适配器；返回 :class:`PreparedModel`（不抛“不可用”异常）。"""
    allow_live = cfg.allow_live if live is None else bool(live)
    kind = spec.kind

    if kind == "local_heads":
        try:
            if features is None:
                features = load_features(cfg.features_path)
            from .local_heads import LocalHeadsAdapter

            slot_geometry = None
            geom_path = cfg.slot_geometry_path
            if geom_path is not None:
                if not geom_path.exists():
                    raise FileNotFoundError(f"缺少 slot 几何特征：{geom_path}")
                with np.load(geom_path, allow_pickle=False) as g:
                    if "slot_geom" not in g.files:
                        raise ValueError(f"{geom_path}: 缺少 slot_geom")
                    slot_geometry = g["slot_geom"].astype(np.float32)
            adapter = LocalHeadsAdapter(
                _asset_path(cfg, spec, cfg.checkpoint_dir), features, manifest,
                slot_geometry=slot_geometry,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            return _unavailable(spec, "local_heads", f"{type(exc).__name__}: {exc}")
        return PreparedModel(
            model_id=spec.id,
            kind="local_heads",
            mode=spec.mode,
            family=spec.family,
            provider=spec.provider,
            spec=spec,
            adapter=adapter,
        )

    if kind == "replay_cloud":
        from .replay_cloud import ReplayCloudAdapter

        cache = _asset_path(cfg, spec, cfg.cloud_cache_path)
        if not cache.exists():
            return _unavailable(spec, "cloud_cache", f"缺少离线回放缓存：{cache}")
        try:
            adapter = ReplayCloudAdapter(cache, manifest)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return _unavailable(spec, "cloud_cache", f"{type(exc).__name__}: {exc}")
        return PreparedModel(
            model_id=spec.id,
            kind="replay_cloud",
            mode=spec.mode,
            family=spec.family,
            provider=spec.provider,
            spec=spec,
            adapter=adapter,
            notes=[f"replay cache: {cache}"],
        )

    if kind == "chat_vlm":
        from .openai_compat import ChatVLMAdapter

        if not spec.base_url or not spec.model:
            return _unavailable(spec, "kind", "chat_vlm 缺少 base_url/model 配置")
        # 缓存路径：extra.cache_path > spec.source（历史写法） > 默认 cloud/live_<id>.jsonl
        cache = resolve_cache_path(cfg, spec)
        cache_has_records = cache.exists() and cache.stat().st_size > 0
        if not allow_live and not cache_has_records:
            return _unavailable(
                spec,
                "live",
                f"--live 未开启且无 live 缓存（{cache}）：禁止联网",
            )
        if allow_live and not _has_live_key(spec):
            return _unavailable(
                spec,
                "api_key_env",
                f"环境变量 {spec.api_key_env or '(未配置 api_key_env)'} 未设置：无法联网（key 只从环境变量读取）",
            )
        adapter = ChatVLMAdapter(
            spec.base_url,
            spec.model,
            spec.api_key_env,
            allow_live=allow_live,
            cache_path=cache,
            max_tokens=int((spec.extra or {}).get("max_tokens", 4000)),
            headers=(spec.extra or {}).get("headers"),
            tolerate_call_errors=bool((spec.extra or {}).get("tolerate_call_errors", False)),
            refetch_empty=bool((spec.extra or {}).get("refetch_empty", False)),
            refetch_unparsable=bool((spec.extra or {}).get("refetch_unparsable", False)),
            refetch_invalid=bool((spec.extra or {}).get("refetch_invalid", False)),
            retries=int((spec.extra or {}).get("retries", 2)),
            retry_sleep_s=float((spec.extra or {}).get("retry_sleep_s", 2.0)),
            timeout_s=float((spec.extra or {}).get("timeout_s", 120.0)),
        )
        notes = [f"live cache: {cache}"] if cache_has_records else [f"live cache (new): {cache}"]
        return PreparedModel(
            model_id=spec.id,
            kind="chat_vlm",
            mode=spec.mode,
            family=spec.family,
            provider=spec.provider,
            spec=spec,
            adapter=adapter,
            notes=notes,
        )

    if kind == "unavailable":
        return _unavailable(spec, "unavailable", spec.note or "该模型本次未参与评测")

    return _unavailable(spec, "kind", f"未知模型 kind：{kind!r}")


def build_models(
    cfg: EvalConfig,
    manifest: DatasetManifest,
    features: FeatureBundle | None = None,
    *,
    live: bool | None = None,
) -> list[PreparedModel]:
    """按 ``cfg.models`` 顺序构造全部模型；特征缓存只加载一次。"""
    needs_features = any(m.kind == "local_heads" for m in cfg.models)
    if needs_features and features is None:
        try:
            features = load_features(cfg.features_path)
        except (FileNotFoundError, ValueError, OSError):
            features = None  # 各 local_heads spec 会各自返回 unavailable，不中断整场评测
    return [build_model(spec, cfg, manifest, features, live=live) for spec in cfg.models]


def execute(
    prepared: PreparedModel,
    manifest: DatasetManifest,
    sample_indices: Iterable[int] | None = None,
    *,
    dataset_dir: str | Path | None = None,
    live: bool = False,
    cache_path: str | Path | None = None,
) -> ModelRun:
    """执行一次预测，返回 :class:`ModelRun`（runtime/na 已填，指标由 verifiers 算）。"""
    run = ModelRun(
        model_id=prepared.model_id,
        kind=prepared.kind,
        mode=prepared.mode,
        family=prepared.family,
        provider=prepared.provider,
        na=dict(prepared.na),
        notes=list(prepared.notes),
    )
    adapter = prepared.adapter
    if prepared.kind == UNAVAILABLE or adapter is None:
        run.kind = UNAVAILABLE
        return run

    indices = None if sample_indices is None else [int(i) for i in sample_indices]
    if cache_path is not None and hasattr(adapter, "set_cache_path"):
        adapter.set_cache_path(cache_path)

    if prepared.kind == "local_heads":
        run.head = adapter.predict()
        run.runtime = adapter.runtime_stats()
        if run.head.latency_s is None:
            run.na["latency"] = "延迟测量被跳过（无可用 pair）"
        return run

    if prepared.kind == "replay_cloud":
        run.pairs = adapter.predict(indices)
        run.runtime = adapter.runtime_stats(indices)
        failures = getattr(adapter, "parse_failures", [])
        if failures:
            run.notes.append(
                f"raw 解析失败 {len(failures)} 对（按原脚本口径计为空预测，不造数）：{failures[:10]}"
                f"{' ...' if len(failures) > 10 else ''}"
            )
        return run

    if prepared.kind == "chat_vlm":
        root = Path(dataset_dir) if dataset_dir is not None else manifest.root
        run.pairs = adapter.predict(manifest, indices, dataset_dir=root, live=bool(live))
        run.runtime = adapter.runtime_stats(indices)
        empty = list(getattr(adapter, "live_empty", []))
        if empty:
            run.notes.append(
                f"live 空响应 {len(empty)} 对（重试后仍为空，按 cloud_pred 口径计空预测并计入分母）："
                f"{empty[:10]}{' ...' if len(empty) > 10 else ''}"
            )
        failures = getattr(adapter, "parse_failures", [])
        if failures:
            run.notes.append(
                f"content 解析失败 {len(failures)} 对（计为空预测，不造数）：{failures[:10]}"
            )
        if not live:
            run.na.setdefault("live", "本次为离线缓存回放（--live 未开启），未发生网络调用")
        return run

    run.kind = UNAVAILABLE
    run.na.setdefault("kind", f"未知模型 kind：{prepared.kind!r}")
    return run


def execute_all(
    prepared: Sequence[PreparedModel],
    manifest: DatasetManifest,
    sample_indices: Iterable[int] | None = None,
    *,
    dataset_dir: str | Path | None = None,
    live: bool = False,
    cache_paths: dict[str, str | Path] | None = None,
) -> list[ModelRun]:
    """批量执行；``cache_paths`` 可按 model_id 覆盖 live 缓存路径。"""
    cache_paths = cache_paths or {}
    return [
        execute(
            p,
            manifest,
            sample_indices,
            dataset_dir=dataset_dir,
            live=live,
            cache_path=cache_paths.get(p.model_id),
        )
        for p in prepared
    ]
