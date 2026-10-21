"""report.json 组装：把各模型的真实测量结果拼成唯一真报告。

只做序列化与口径声明，不做任何"补数"。缺失维度一律保留 None + na 原因。
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import EvalConfig
from ..types import ModelRun
from .composite import add_composite_score

REPORT_SCHEMA_VERSION = 2

# 指标定义出处（写死在报告里，保证"测试与文档同源"）
METRIC_DEFINITIONS: dict[str, Any] = {
    "head.change": {
        "doc": "change 4 类逐 slot top-1 / macro-F1",
        "macro_f1_variant": "macro_f1_train（跳过 tp+fp+fn==0 的类）",
        "source": "starlife_scripts/train_heads.py:35,114-115",
        "split": "test",
    },
    "head.type": {
        "doc": "type 闭集 22 类 top-1 / macro-F1（内部诊断项，不进榜单）",
        "source": "starlife_scripts/train_heads.py:252-257",
    },
    "head.relation": {
        "doc": "relation 12 类（11 关系 + none，含负样本）top-1 / macro-F1",
        "negative_sampling": "train_heads.build_relation_data，rng seed 20261020",
        "source": "starlife_scripts/train_heads.py:169-205,272-276",
    },
    "head.rel_pos": {
        "doc": "rel_pos 10 类，仅非 none 真关系；输入以 GT 关系类型 one-hot 为条件",
        "source": "starlife_scripts/train_heads.py:268-285",
    },
    "head.attrs": {
        "doc": "contact / support 二分类 macro-F1 与均值",
        "source": "starlife_scripts/train_heads.py:304-335",
    },
    "pair_protocol": {
        "doc": "60 对云端同批协议：change top1/compare-macroF1、relation_type_acc、rel_pos_acc、attrs_acc",
        "macro_f1_variant": "macro_f1_compare（跳过 tp==0 的类，云端 0.7781 依赖此口径）",
        "sample": "sorted(random.Random(20261020).sample(test_idx, 60))",
        "source": "starlife_scripts/eval_compare_cloud.py:213,244-281",
    },
    "event": {
        "doc": "事件级 P/R/F1：StateChanged / EntityCreated / RelationAdded 逐类 + macro",
        "relation_removed": "v2 无 before 关系标注 → N/A",
        "source": "starlife_doc/决策判别模型/测试套件设计.md §3 L3",
    },
    "state_replay": {
        "doc": "事件回放后世界状态与 GT after 场景逐维一致率（objects/types/changes/relations/attrs）",
        "score": "state_acc = 可用维度等权平均",
        "source": "starlife_doc/决策判别模型/测试套件设计.md §3 L4",
    },
    "runtime": {
        "doc": "运行记录（内部信息，不计分）：延迟实测",
        "local_cost": "自有硬件，cost_usd=0（cost_basis=on-prem）",
        "cloud_cost": "无公开计价口径 → null",
    },
    "composite": {
        "doc": "综合能力分 = 7 个「所有模型都可测」维度（各 0-100）的等权平均；缺项从分母剔除、不当 0 分",
        "dims": [
            {"key": "change_top1", "plain": "两帧之间每个物体发生了什么（出现/消失/被挪动/没变），认对的比例",
             "source": "pair_protocol.change_top1"},
            {"key": "change_balance", "plain": "稀有变化（如「消失」）也判得准吗；防止「全猜没变」刷分",
             "source": "pair_protocol.change_macro_f1"},
            {"key": "relation_type", "plain": "物体之间 11 类空间关系判对的比例（前提是找对两个物体）",
             "source": "pair_protocol.relation_type_acc"},
            {"key": "rel_pos", "plain": "关系判对后，前/后/左/右/上/下调对的比例",
             "source": "pair_protocol.rel_pos_acc"},
            {"key": "attrs", "plain": "两物体是否接触、是否互相支撑（交互/affordance 推断的地基）",
             "source": "pair_protocol.attrs_acc"},
            {"key": "event_f1", "plain": "写成事件流后 StateChanged + RelationAdded 的 P/R/F1",
             "source": "event_metrics.macro_f1_common"},
            {"key": "state_consistency", "plain": "事件回放重建的 after 状态与真实状态的一致率（objects/changes/relations/attrs）",
             "source": "state_replay.state_acc_common"},
        ],
        "weights": "equal（1/7 每维，公开 weights 数组；总分依赖权重，故同时输出逐维分数）",
        "state_types_excluded": "types 仅本地模型可测，混入会让云端被扣一个它没有的维度",
    },
    "world_model_stage": {
        "doc": "本评测在世界模型六层中的位置：② 状态估计 / ③ 变化识别 / ④ 关系与拓扑 / ⑤ 事件化 / ⑥ 世界状态一致性",
        "source": "starlife_doc/决策判别模型/测试套件设计.md §3（L1-L5 分层）",
    },
}


PROVIDER_COLORS = {
    "StarLife": "var(--color-starlife)",
    "DeepSeek": "var(--color-deepseek)",
    "Google": "var(--color-gemini)",
    "OpenAI": "var(--color-openai)",
    "Qwen": "var(--color-qwen)",
    "Anthropic": "var(--color-anthropic, #c96442)",
    "Mistral": "var(--color-mistral, #ff7000)",
    "Meta": "var(--color-meta, #0866ff)",
    "Zhipu": "var(--color-zhipu, #3b82f6)",
    "ByteDance": "var(--color-bytedance, #325ab4)",
    "MiniMax": "var(--color-minimax, #f97316)",
    "Xiaomi": "var(--color-xiaomi, #ff6900)",
    "Tencent": "var(--color-tencent, #12b7f5)",
}


def _default_color(family: str | None) -> str:
    return PROVIDER_COLORS.get(family or "", "var(--color-muted, #8b93a7)")


def sha256_prefix(path: Path, n: int = 16) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _jsonable(obj: Any) -> Any:
    """递归把 numpy 标量/数组转成原生类型；None 保持 None。"""
    import numpy as np

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # NaN/Inf 不是合法 JSON：一律 None，并由 na 说明"指标不可定义"
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class ModelEvaluation:
    """一个模型在一次评测中的完整结果（真实测量，缺失为 None）。"""

    model_id: str
    kind: str
    mode: str
    family: str
    provider: str
    run: ModelRun
    name: str | None = None
    provider_color: str | None = None
    head_metrics: dict[str, Any] | None = None
    pair_protocol: dict[str, Any] | None = None
    event_metrics: dict[str, Any] | None = None
    state_replay: dict[str, Any] | None = None
    na: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def runtime_dict(self) -> dict[str, Any]:
        r = self.run.runtime
        return {
            "latency_ms_mean": r.latency_ms_mean,
            "latency_ms_p50": r.latency_ms_p50,
            "latency_ms_max": r.latency_ms_max,
            "tokens_prompt": r.tokens_prompt,
            "tokens_completion": r.tokens_completion,
            "cost_usd": r.cost_usd,
            "cost_basis": r.cost_basis,
            "n_calls": r.n_calls,
            "source": r.source,
        }


# score 只使用"所有模型都能产出的事件类"：StateChanged / RelationAdded。
# EntityCreated 需要实体类型标注（云端零样本不返回类型 → N/A），若并入 macro 会让
# 本地模型因多测一个更难的维度而被系统性压低，横向不可比。故单独展示、不计入 score。
SCORE_KINDS = ("StateChanged", "RelationAdded")


def add_common_score(report: dict[str, Any]) -> None:
    """给每个模型的 event_metrics 追加 macro_f1_common（固定类集合，横向可比）。"""
    for m in (report.get("models") or {}).values():
        ev = m.get("event_metrics") or {}
        per_kind = ev.get("per_kind") or {}
        f1s = []
        kinds = []
        for k in SCORE_KINDS:
            entry = per_kind.get(k) or {}
            if entry.get("available") is False or entry.get("f1") is None:
                continue
            f1s.append(float(entry["f1"]))
            kinds.append(k)
        ev["macro_f1_common"] = (sum(f1s) / len(f1s)) if f1s else None
        ev["common_kinds"] = kinds
        m["event_metrics"] = ev


def add_pipeline_latency(report: dict[str, Any]) -> None:
    """本地头的横向延迟必须含特征抽取（bench_pipeline 实测），否则 1.6ms 不可比。"""
    bench = ((report.get("meta") or {}).get("artifacts") or {}).get("bench_pipeline") or {}
    event_s = bench.get("event_latency_s")
    for m in (report.get("models") or {}).values():
        rt = m.get("runtime") or {}
        if m.get("kind") != "local_heads":
            rt["pipeline_latency_ms_mean"] = rt.get("latency_ms_mean")
            m["runtime"] = rt
            continue
        if event_s is None or rt.get("latency_ms_mean") is None:
            rt["pipeline_latency_ms_mean"] = rt.get("latency_ms_mean")
        else:
            rt["pipeline_latency_ms_mean"] = round(1000.0 * float(event_s) + float(rt["latency_ms_mean"]), 1)
            rt["latency_breakdown"] = {
                "feat_extract_ms": round(1000.0 * float(event_s), 1),
                "head_forward_ms": round(float(rt["latency_ms_mean"]), 2),
                "source": "bench_pipeline.json:event_latency_s（2 帧 + 12 crop，1024×768）",
            }
        m["runtime"] = rt


def _scrub_paths(obj: Any, cfg: EvalConfig) -> Any:
    """把 report 里的绝对路径换成占位符（报告要能对外发布，不泄露本机目录）。"""
    from pathlib import Path as _Path

    root = str(cfg.data_root)
    out = str(cfg.out) if cfg.out else None

    def one(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: one(v) for k, v in value.items()}
        if isinstance(value, list):
            return [one(v) for v in value]
        if isinstance(value, str):
            text = value
            if out:
                text = text.replace(out, "<OUT>")
            if root:
                text = text.replace(root, "<DATA_ROOT>")
            return text
        return value

    return one(obj)


def build_report(cfg: EvalConfig, evaluations: list[ModelEvaluation], *, run_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    bench = None
    if cfg.bench_path.exists():
        try:
            bench = json.loads(cfg.bench_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bench = None

    meta = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator": "starlife_eval",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": "空间理解与关系推理基准（Spatial & Relational Understanding Benchmark）",
        "open_to": "任何能输出逐 slot 变化与槽位对关系的视觉语言模型（零样本或专用头均可）。",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "config": cfg.to_dict(),
        "artifacts": {
            "dataset_dir": str(cfg.dataset_dir),
            "manifest_sha256_16": sha256_prefix(cfg.dataset_dir / "manifest.jsonl"),
            "features": str(cfg.features_path),
            "features_sha256_16": sha256_prefix(cfg.features_path),
            "checkpoints": str(cfg.checkpoint_dir),
            "checkpoint_files": sorted(p.name for p in cfg.checkpoint_dir.glob("*.pt"))
            if cfg.checkpoint_dir.exists()
            else [],
            "cloud_cache": str(cfg.cloud_cache_path),
            "cloud_cache_sha256_16": sha256_prefix(cfg.cloud_cache_path),
            "bench_pipeline": bench,
        },
        "split": cfg.split,
        "n_pairs": cfg.n_pairs,
        "sample_seed": cfg.sample_seed,
    }
    if run_meta:
        meta.update(run_meta)

    models: dict[str, Any] = {}
    for ev in evaluations:
        models[ev.model_id] = {
            "name": ev.name or ev.model_id,
            "kind": ev.kind,
            "mode": ev.mode,
            "family": ev.family,
            "provider": ev.provider,
            "providerColor": ev.provider_color or _default_color(ev.family),
            "head_metrics": ev.head_metrics,
            "pair_protocol": ev.pair_protocol,
            "event_metrics": ev.event_metrics,
            "state_replay": ev.state_replay,
            "runtime": ev.runtime_dict(),
            "na": ev.na,
            "notes": ev.notes,
        }

    report = {
        "meta": meta,
        "metric_definitions": METRIC_DEFINITIONS,
        "models": models,
    }
    add_common_score(report)
    add_pipeline_latency(report)
    add_composite_score(report)
    report["leaderboard"] = build_leaderboard_rows(report)
    return _scrub_paths(_jsonable(report), cfg)


# --------------------------------------------------------------------- 榜单映射


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100.0 * float(x), 1)


def build_leaderboard_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """§5.6 映射：score=event.macro_f1, topologyF1=relation_type_acc,
    affordanceAcc=attrs_acc, stateTransitionAcc=change_top1，全部真实值或 None。"""
    rows: list[dict[str, Any]] = []
    for model_id, m in report["models"].items():
        pair = m.get("pair_protocol") or {}
        ev = m.get("event_metrics") or {}
        rt = m.get("runtime") or {}
        na = dict(m.get("na") or {})
        score = _pct(ev.get("macro_f1_common")) if ev.get("macro_f1_common") is not None else None
        if score is None:
            na.setdefault("score", "事件级指标不可用（无事件级预测/GT）")
        latency = rt.get("pipeline_latency_ms_mean", rt.get("latency_ms_mean"))
        row = {
            "id": model_id,
            "name": m.get("name") or model_id,
            "family": m.get("family") or "Community",
            "provider": m.get("provider") or "Community",
            "providerColor": m.get("providerColor") or _default_color(m.get("family")),
            "mode": m.get("mode") or "—",
            "compositeScore": m.get("composite_score"),
            "composite": m.get("composite"),
            "score": score,
            "topologyF1": _pct(pair.get("relation_type_acc")),
            "affordanceAcc": _pct(pair.get("attrs_acc")),
            "stateTransitionAcc": _pct(pair.get("change_top1")),
            "latency": (round(float(latency), 1) if latency is not None else None),
            "confidence": None,
            "version": "v1.3",
            "metrics": {
                "head": _head_summary(m.get("head_metrics")),
                "pair": {k: pair.get(k) for k in (
                    "change_top1", "change_macro_f1", "relation_type_acc", "rel_pos_acc", "attrs_acc")} if pair else None,
                "event": {k: ev.get(k) for k in (
                    "macro_f1", "macro_f1_common", "common_kinds", "matched_total", "available_kinds")} if ev else None,
                "state": (m.get("state_replay") or {}).get("state_acc") if m.get("state_replay") else None,
            },
            "na": na,
        }
        rows.append(row)

    # isBest：全榜综合分最高者只有一个（综合分缺失则回落到旧 score）；family 内排序交给前端
    def _rank_key(r: dict[str, Any]) -> tuple[float, float, float]:
        comp = r.get("compositeScore")
        return (comp if comp is not None else -1.0, r.get("score") or 0.0, r.get("topologyF1") or 0.0)

    ranked = [r for r in rows if _rank_key(r)[0] >= 0]
    for r in rows:
        r["isBest"] = False
    if ranked:
        best = max(ranked, key=_rank_key)
        best["isBest"] = True

    rows.sort(key=lambda r: (r["compositeScore"] is None, -(r["compositeScore"] or 0.0),
                             -(r["score"] or 0.0)))
    return rows


def _head_summary(head: dict[str, Any] | None) -> dict[str, Any] | None:
    if not head:
        return None
    out: dict[str, Any] = {}
    for key in ("change", "relation", "rel_pos"):
        d = head.get(key) or {}
        out[f"{key}_top1"] = d.get("top1")
        out[f"{key}_macro_f1"] = d.get("macro_f1")
    # `type`（22 类实体类型/词表分类）是本地头的内部诊断项，不属于本基准的评测维度，
    # 不进排行榜契约（云端模型没有类型输出，横向不可比）。
    a = head.get("attrs") or {}
    out["attrs_contact_f1"] = a.get("contact_f1")
    out["attrs_support_f1"] = a.get("support_f1")
    out["attrs"] = a.get("attrs")
    return out


def _fmt_tokens(p: int | None, c: int | None) -> str | None:
    if p is None and c is None:
        return None
    total = (p or 0) + (c or 0)
    return f"{total / 1000:.1f}k" if total >= 1000 else str(total)


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
