"""云端离线回放适配器（deepseek-flash raw jsonl → PairPrediction）。

解析逻辑逐行移植 ``starlife/starlife_scripts/eval_compare_cloud.py``：

- ``parse_json``  (90-102 行)：兼容 ``` 代码块包裹与前后解释文字；
- ``norm_pred``   (105-126 行)：changes 兼容 list/dict，relations 兼容 dict/list；
- ``cloud_pred``  (283-287 行)：解析失败 → 空预测（评测口径如此，**不静默编造**，
  失败 pair 记录到 ``parse_failures`` 供报告注明）；
- raw 记录格式 (236-240 行)：``{"pair_index","pair_id","raw","latency_s","usage"}``。

回放是纯离线操作，不联网。缺请求的 pair 时 **明确报错**（``MissingCloudPairs``），
而不是像原脚本那样默认空预测——那样会把缓存缺失伪装成模型答错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..types import DatasetManifest, PairPrediction, RelationPrediction, RuntimeStats


class MissingCloudPairs(LookupError):
    """回放缓存缺少被请求的 pair。"""


def parse_json(text: str) -> dict[str, Any]:
    """ported from eval_compare_cloud.py:90-102 parse_json（逐行一致）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def norm_pred(p: Any) -> dict[str, Any]:
    """ported from eval_compare_cloud.py:105-126 norm_pred（逐行一致）。"""
    if not isinstance(p, dict):
        return {"changes": {}, "relations": []}
    ch = p.get("changes", {})
    if isinstance(ch, list):
        nch = {}
        for x in ch:
            if isinstance(x, dict) and "slot" in x:
                try:
                    nch[int(x["slot"])] = str(x.get("type", "none"))
                except Exception:  # noqa: BLE001
                    pass
        ch = nch
    elif isinstance(ch, dict):
        ch = {int(k): v for k, v in ch.items()}
    else:
        ch = {}
    rel = p.get("relations", [])
    if isinstance(rel, dict):
        rel = list(rel.values())
    rel = [x for x in rel if isinstance(x, dict)]
    return {"changes": ch, "relations": rel}


def _slot_number(value: Any) -> Any:
    """关系槽位编号：数字（含 float）转 int，其他原样保留（与原脚本比较语义一致）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value)
    return value


def _relation_type(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "none"


def to_relation_prediction(raw: Mapping[str, Any]) -> RelationPrediction:
    """单条云端关系 dict → :class:`RelationPrediction`。

    contact/support 用 ``bool(...)`` 语义（与 eval_compare_cloud.py:270-271
    的 ``int(bool(cand.get("contact")) == bool(g["contact"]))`` 完全一致：
    缺省/None/非空对象都按 Python 真值处理）。
    """
    return RelationPrediction(
        from_slot=_slot_number(raw.get("from_slot")),
        to_slot=_slot_number(raw.get("to_slot")),
        type=_relation_type(raw.get("type")),
        rel_pos=raw.get("rel_pos") if isinstance(raw.get("rel_pos"), str) else None,
        contact=bool(raw.get("contact")),
        support=bool(raw.get("support")),
    )


def to_pair_prediction(pair_index: int, parsed: Mapping[str, Any]) -> PairPrediction:
    """``norm_pred`` 的输出 → :class:`PairPrediction`（云端不输出 types）。"""
    norm = norm_pred(parsed)
    return PairPrediction(
        pair_index=int(pair_index),
        changes={int(k): str(v) for k, v in norm["changes"].items()},
        types={},
        relations=[to_relation_prediction(x) for x in norm["relations"]],
    )


def load_raw_records(path: str | Path) -> dict[int, dict[str, Any]]:
    """读 raw jsonl；坏行静默跳过、同 pair 后写覆盖（与 eval_compare_cloud.py:217-224 一致）。

    额外把 ``pair_index`` 统一成 int（缓存是我们自己写的格式）。
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"缺少云端回放缓存：{raw_path}")
    done: dict[int, dict[str, Any]] = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            done[int(rec["pair_index"])] = rec
        except Exception:  # noqa: BLE001
            continue
    return done


class ReplayCloudAdapter:
    """离线回放云端缓存（不联网、无 key 需求）。"""

    def __init__(self, raw_jsonl_path: str | Path, manifest: DatasetManifest | None = None) -> None:
        self.path = Path(raw_jsonl_path)
        self.manifest = manifest
        self.records: dict[int, dict[str, Any]] = load_raw_records(self.path)
        self.parse_failures: list[int] = []

    # ------------------------------------------------------------------ data
    def available_pair_indices(self) -> list[int]:
        return sorted(self.records)

    def _missing(self, indices: Sequence[int]) -> list[int]:
        return [int(i) for i in indices if int(i) not in self.records]

    def predict(self, sample_indices: Iterable[int] | None = None) -> dict[int, PairPrediction]:
        """返回 ``{pair_index: PairPrediction}``；缺 pair 直接抛 :class:`MissingCloudPairs`。"""
        if sample_indices is None:
            indices = self.available_pair_indices()
        else:
            indices = [int(i) for i in sample_indices]
        missing = self._missing(indices)
        if missing:
            raise MissingCloudPairs(
                f"{self.path}: 缺少 {len(missing)} 个请求 pair 的回放记录：{missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}（不允许默认空预测）"
            )

        out: dict[int, PairPrediction] = {}
        for i in indices:
            rec = self.records[i]
            raw = rec.get("raw")
            # ported from eval_compare_cloud.py:283-287 cloud_pred:
            # 解析失败 → 空预测（该 pair 计为全错），并记录失败清单供报告注明
            try:
                if not isinstance(raw, str):
                    raise TypeError(f"raw 不是字符串: {type(raw).__name__}")
                pred = to_pair_prediction(i, parse_json(raw))
            except Exception:  # noqa: BLE001
                self.parse_failures.append(i)
                pred = PairPrediction(pair_index=i)
            out[i] = pred
        return out

    # --------------------------------------------------------------- runtime
    def latency_values(self, sample_indices: Iterable[int] | None = None) -> list[float]:
        indices = self.available_pair_indices() if sample_indices is None else [int(i) for i in sample_indices]
        vals: list[float] = []
        for i in indices:
            rec = self.records.get(i)
            if rec is None:
                continue
            lat = rec.get("latency_s")
            if isinstance(lat, (int, float)) and not isinstance(lat, bool):
                vals.append(float(lat))
        return vals

    def usage_values(self, sample_indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        indices = self.available_pair_indices() if sample_indices is None else [int(i) for i in sample_indices]
        out: list[dict[str, Any]] = []
        for i in indices:
            rec = self.records.get(i)
            if rec is None:
                continue
            usage = rec.get("usage")
            out.append(usage if isinstance(usage, dict) else {})
        return out

    def runtime_stats(self, sample_indices: Iterable[int] | None = None) -> RuntimeStats:
        """缓存里的真实 ``latency_s`` / ``usage``；成本无公开计价口径 → None。"""
        lat = self.latency_values(sample_indices)
        usages = self.usage_values(sample_indices)
        tokens_prompt = int(sum(int(u.get("prompt_tokens", 0) or 0) for u in usages))
        tokens_completion = int(sum(int(u.get("completion_tokens", 0) or 0) for u in usages))
        stats = RuntimeStats(
            tokens_prompt=tokens_prompt if usages else None,
            tokens_completion=tokens_completion if usages else None,
            cost_usd=None,
            cost_basis="cloud API pricing not publicly defined; cost left null (tokens_* are measured)",
            n_calls=len(lat) if lat else (len(usages) if usages else None),
            source="measured (cache latency_s/usage)",
        )
        if lat:
            arr = np.asarray(lat, dtype=np.float64) * 1000.0
            stats.latency_ms_mean = float(arr.mean())
            stats.latency_ms_p50 = float(np.percentile(arr, 50))
            stats.latency_ms_max = float(arr.max())
        return stats


def write_raw_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """append 一行 raw 记录（与 eval_compare_cloud.py:236-240 写缓存格式一致）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
