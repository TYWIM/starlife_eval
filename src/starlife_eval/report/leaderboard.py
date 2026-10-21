"""生成前端契约文件 leaderboard.generated.json（third_party/openwsv 消费）。

契约见 ARCHITECTURE.md §8 与 frontend 确认的结构：
  {schemaVersion, meta:{benchmark,split,nPairs,nTestPairs,dataset,checkpoints,
   sampleSeed,generatedAt,sourceRun,telemetry}, models:[ModelResult...]}
所有数值可为 null（前端渲染 N/A），na 给字段级 N/A 原因。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4


def build_leaderboard_payload(report: dict[str, Any], *, n_test_pairs: int | None = None,
                              source_run: str | None = None) -> dict[str, Any]:
    meta = report.get("meta") or {}
    rows = report.get("leaderboard") or []
    models = []
    for r in rows:
        models.append(
            {
                "id": r["id"],
                "name": r.get("name") or r["id"],
                "family": r.get("family") or "Community",
                "provider": r.get("provider") or "Community",
                "providerColor": r.get("providerColor") or "var(--color-muted, #8b93a7)",
                "mode": r.get("mode") or "—",
                "compositeScore": r.get("compositeScore"),
                "composite": r.get("composite"),
                "score": r.get("score"),
                "topologyF1": r.get("topologyF1"),
                "affordanceAcc": r.get("affordanceAcc"),
                "stateTransitionAcc": r.get("stateTransitionAcc"),
                "latency": r.get("latency"),
                "confidence": None,
                "version": r.get("version") or "v1.3",
                "isBest": bool(r.get("isBest")),
                "metrics": r.get("metrics"),
                "na": r.get("na") or {},
            }
        )
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "meta": {
            "benchmark": meta.get("benchmark") or "空间理解与关系推理基准",
            "benchmarkShort": "Spatial & Relational Understanding Benchmark",
            "split": meta.get("split"),
            "nPairs": meta.get("n_pairs"),
            "nTestPairs": n_test_pairs,
            "dataset": (meta.get("config") or {}).get("dataset"),
            "checkpoints": (meta.get("config") or {}).get("checkpoints"),
            "sampleSeed": meta.get("sample_seed"),
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sourceRun": source_run,
            "artifacts": {
                "features_sha256_16": (meta.get("artifacts") or {}).get("features_sha256_16"),
                "manifest_sha256_16": (meta.get("artifacts") or {}).get("manifest_sha256_16"),
                "cloud_cache_sha256_16": (meta.get("artifacts") or {}).get("cloud_cache_sha256_16"),
            },
            "telemetry": _telemetry(report, n_test_pairs),
            "composite": meta.get("composite"),
        },
        "models": models,
    }
    return payload


def _telemetry(report: dict[str, Any], n_test_pairs: int | None) -> dict[str, Any]:
    """真实计数，取不到就是 None（前端隐藏而不写死）。

    `pairs` = 每模型实际评测的对数（60 对协议样本，榜单可比范围）；
    `testPairs` = test split 总对数（200，数据集规模）。两者语义不同，不要混用。
    """
    meta = report.get("meta") or {}
    counts = meta.get("counts") or {}
    return {
        "tasks": counts.get("heads", 5),
        "scenes": counts.get("scene_groups"),
        "pairs": meta.get("n_pairs"),
        "testPairs": n_test_pairs,
        "slots": counts.get("slots"),
        "relations": counts.get("relations"),
        "models": len(report.get("models") or {}),
    }


def write_leaderboard(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
