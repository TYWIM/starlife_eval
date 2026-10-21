"""真实产物对拍（硬断言）：逐位复现服务器上**原始脚本**跑出的 reference oracle。

fixtures 来源与含义（全部由 Lead 从服务器真实产物 dump，未改原脚本）：
- ``reference_oracle.json``：``ref_oracle.py`` 用 ``train_heads.py`` / ``eval_compare_cloud.py``
  跑出的基准（头级全 test split + 60 对协议）；
- ``ref_gt.npz``：GT 数组 ``slot_change/slot_type/slot_pair/pair_split``（全 1000 对）；
- ``ref_relation_data.npz``：原始 ``train_heads.build_relation_data`` 完整输出（含负样本）；
- ``ref_pred_pilot_v2{,g}.npz``：原始头的 test 全 split 预测数组（change/type/relation/rel_pos/attrs）；
- ``ref_manifest_sample60.json``：60 个抽样 test pair 的原始 manifest 行 + ``sample_indices``；
- ``eval_cloud_raw.jsonl``：``eval_compare_cloud.py`` 的云端回放缓存（raw/latency_s/usage）。

容差 1e-12：两边是**同一套 numpy 浮点运算**（同输入、同顺序），应当逐位一致；
留 1e-12 只是吸收 JSON 十进制↔二进制往返误差。若对不上说明口径实现有偏差，
**不要改期望值**，应改实现或报 Lead。

注意：``eval_cloud_raw.jsonl`` 里有 19/60 条 raw 为空或截断；原脚本
``eval_compare_cloud.cloud_pred`` 对这些记录的兜底是 ``{"changes": {},"relations": []}``
（即整对按全错计），本测试完全镜像该兜底，否则云端 0.7781 对不上。
生产实现里对应 ``models/replay_cloud.py`` 的解析逻辑（归 backend-data）。
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from starlife_eval.types import (
    CHANGE_INDEX,
    DatasetManifest,
    Entity,
    HeadPrediction,
    PairPrediction,
    PairSample,
    Relation,
    RelationPrediction,
)
from starlife_eval.verifiers.head_metrics import evaluate_head_metrics
from starlife_eval.verifiers.pair_protocol import (
    evaluate_pair_protocol,
    local_pair_predictions,
)
from starlife_eval.verifiers.runtime import from_latencies_s, summarize_tokens

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOL = 1e-12
MODELS = ("pilot_v2", "pilot_v2g")

_REQUIRED_FIXTURES = (
    "reference_oracle.json",
    "ref_gt.npz",
    "ref_relation_data.npz",
    "ref_pred_pilot_v2.npz",
    "ref_pred_pilot_v2g.npz",
    "ref_manifest_sample60.json",
    "eval_cloud_raw.jsonl",
    "spatial_v2.meta.json",
)

pytestmark = pytest.mark.skipif(
    not all((FIXTURES / name).exists() for name in _REQUIRED_FIXTURES),
    reason=f"缺少真实产物 fixtures（应位于 {FIXTURES}，从服务器 tests/fixtures 同步）",
)


# --------------------------------------------------------------------- loaders


@lru_cache(maxsize=1)
def _oracle() -> dict:
    return json.loads((FIXTURES / "reference_oracle.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _meta() -> dict:
    return json.loads((FIXTURES / "spatial_v2.meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _gt_z() -> dict:
    d = np.load(FIXTURES / "ref_gt.npz")
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=1)
def _relation() -> dict:
    d = np.load(FIXTURES / "ref_relation_data.npz")
    rel = {k: d[k] for k in d.files}
    rel["is_none"] = rel["is_none"].astype(bool)
    return rel


@lru_cache(maxsize=1)
def _manifest60_json() -> dict:
    return json.loads((FIXTURES / "ref_manifest_sample60.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _cloud_records() -> dict[int, dict]:
    path = FIXTURES / "eval_cloud_raw.jsonl"
    return {
        int(rec["pair_index"]): rec
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for rec in [json.loads(line)]
    }


@lru_cache(maxsize=1)
def _pair_row_by_index() -> dict[int, dict]:
    man = _manifest60_json()
    return {int(i): row for i, row in zip(man["sample_indices"], man["rows"], strict=True)}


@lru_cache(maxsize=1)
def _slot_offsets() -> np.ndarray:
    """每个 manifest 行在 npz 全局 slot 空间的起始下标。

    ``slot_pair`` 是按 pair 行号非降序排列的（特征抽取顺序），
    所以 ``offsets[i] = searchsorted(slot_pair, i, "left")``。
    """
    slot_pair = _gt_z()["slot_pair"]
    assert np.all(np.diff(slot_pair) >= 0), "slot_pair 不是按 pair 行号有序，无法推 offsets"
    return np.searchsorted(slot_pair, np.arange(len(_gt_z()["pair_split"])))


@lru_cache(maxsize=1)
def _sample_indices() -> tuple[int, ...]:
    return tuple(int(i) for i in _manifest60_json()["sample_indices"])


def _head_prediction(model: str) -> HeadPrediction:
    pred = np.load(FIXTURES / f"ref_pred_{model}.npz")
    rel = _relation()
    return HeadPrediction(
        change=pred["change"].astype(np.int64),
        type=pred["type"].astype(np.int64),
        relation=pred["relation"].astype(np.int64),
        rel_pos=pred["rel_pos"].astype(np.int64),
        attrs=pred["attrs"].astype(np.int64),
        n_relation_positives=int(len(rel["frm"]) - int(pred["n_none"])),
    )


# -------------------------------------------------------------- assertions util


def assert_metrics_equal(actual: Any, expected: Any, path: str = "", tol: float = TOL) -> None:
    """递归比较指标 dict/list；浮点用 ``abs diff <= tol``（同源运算应为 0）。"""
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path}: 期望映射，实际 {type(actual)}"
        assert set(actual) == set(expected), (
            f"{path}: 键不一致\n  actual={sorted(actual)}\n  expected={sorted(expected)}"
        )
        for key, value in expected.items():
            assert_metrics_equal(actual[key], value, f"{path}/{key}", tol)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{path}: 期望序列，实际 {type(actual)}"
        assert len(actual) == len(expected), f"{path}: 长度 {len(actual)} != {len(expected)}"
        for idx, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_metrics_equal(a, e, f"{path}[{idx}]", tol)
    elif isinstance(expected, bool):
        assert bool(actual) == expected, f"{path}: {actual} != {expected}"
    elif isinstance(expected, (int, float)):
        actual_f = float(actual)
        assert math.isfinite(actual_f), f"{path}: 实际值非有限 {actual}"
        diff = abs(actual_f - float(expected))
        assert diff <= tol, f"{path}: {actual} != {expected} (abs diff {diff:.3e} > {tol:.1e})"
    elif expected is None:
        assert actual is None, f"{path}: 期望 None，实际 {actual}"
    else:  # pragma: no cover
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# ----------------------------------------------------------------- head metrics


@pytest.mark.parametrize("model", MODELS)
def test_head_metrics_parity_with_original_scripts(model: str) -> None:
    """evaluate_head_metrics 必须逐位复现 train_heads.main 在 test split 的口径。"""
    oracle_head = _oracle()[model]["head_metrics"]
    # reference_oracle 里的 "_dims" 是原始脚本记录的特征维度，不属于指标
    expected = {k: v for k, v in oracle_head.items() if k != "_dims"}

    got = evaluate_head_metrics(_gt_z(), _meta(), _head_prediction(model), _relation(), split="test")
    assert_metrics_equal(got, expected, path=f"{model}/head_metrics")

    # 类别数锁定：4 / 22 / 12 / 10（不因 test 缺类而收缩）
    assert len(got["change"]["per_class_f1"]) == 4
    assert len(got["type"]["per_class_f1"]) == len(_meta()["type_keys"]) == 22
    assert len(got["relation"]["per_class_f1"]) == int(_relation()["n_types"]) == 12
    assert len(got["rel_pos"]["per_class_f1"]) == len(_meta()["rel_pos_keys"]) == 10
    # n 与原始脚本一致（test split 全量）
    assert got["change"]["n"] == 1165
    assert got["relation"]["n"] == 1370
    assert got["rel_pos"]["n"] == 551


# --------------------------------------------------------- 60-pair protocol: local


def _build_manifest60() -> DatasetManifest:
    gt = _gt_z()
    man = _manifest60_json()
    rows_by_index = _pair_row_by_index()
    offsets = _slot_offsets()
    pairs: list[PairSample] = []
    for i in man["sample_indices"]:
        row = rows_by_index[i]
        base = int(offsets[i])
        entities = []
        for pos, entity in enumerate(row["entities"]):
            global_slot = base + pos
            assert int(gt["slot_pair"][global_slot]) == i, (
                f"slot_pair 与 manifest 不一致: pair={i} pos={pos}"
            )
            entities.append(
                Entity(
                    slot=int(entity["slot"]),
                    bbox=tuple(float(v) for v in entity["bbox"]),
                    type_key=str(entity.get("type_key", "")),
                    present_after=bool(entity.get("present_after", True)),
                )
            )
        relations = tuple(
            Relation(
                from_slot=int(rel["from_slot"]),
                to_slot=int(rel["to_slot"]),
                type=str(rel["type"]),
                rel_pos=str(rel["rel_pos"]) if "rel_pos" in rel else None,
                contact=bool(rel.get("contact", False)),
                support=bool(rel.get("support", False)),
                dist_m=float(rel["dist_m"]) if "dist_m" in rel else None,
            )
            for rel in row.get("relations", [])
        )
        pairs.append(
            PairSample(
                index=int(i),
                pair_id=str(row["pair_id"]),
                split=str(row["split"]),
                before=row.get("before", ""),
                after=row.get("after", ""),
                entities=tuple(entities),
                changes={int(c["slot"]): str(c["type"]) for c in row.get("changes", [])},
                relations=relations,
                raw=row,
            )
        )
    # 用真实**全局** slot 偏移：local_pair_predictions 靠 slot_offsets 做
    # 全局 slot 下标 -> pair 行号映射。pairs 只放了 60 对（对拍不需要其余行的实体内容），
    # 因此这里 pairs 长度与 slot_offsets 长度不满足"行数+1"，仅测试构造如此。
    full_offsets = np.append(_slot_offsets(), len(gt["slot_pair"]))
    return DatasetManifest(
        root=FIXTURES,
        pairs=tuple(pairs),
        meta=_meta(),
        slot_offsets=tuple(int(x) for x in full_offsets),
    )


def _local_pair_preds(model: str) -> dict[int, PairPrediction]:
    """本地头预测 -> 逐 pair PairPrediction（调用 pair_protocol 的生产 helper）。"""
    return local_pair_predictions(
        _build_manifest60(),
        _head_prediction(model),
        _relation(),
        _sample_indices(),
        _meta(),
    )


@pytest.mark.parametrize("model", MODELS)
def test_pair_protocol_local_parity_with_original_script(model: str) -> None:
    """60 对协议（本地头）必须复现 eval_compare_cloud.evaluate 的 local 段。"""
    indices = list(_sample_indices())
    result = evaluate_pair_protocol(_build_manifest60(), indices, _local_pair_preds(model))

    expected = _oracle()[model]["pair_protocol"]["local"]
    assert_metrics_equal(result, expected, path=f"{model}/pair_protocol/local")

    # 自检（Lead 指定）：change 统计 347 个 slot、176 个 GT 关系对
    assert result["change_slots"] == 347
    assert result["relation_pairs"] == 176


def test_sample_indices_fixture_selfcheck() -> None:
    man = _manifest60_json()
    indices = man["sample_indices"]
    assert len(indices) == man["n"] == 60
    assert len(set(indices)) == 60
    assert indices == sorted(indices)
    assert man["seed"] == 20261020
    rows_by_index = _pair_row_by_index()
    assert all(rows_by_index[i]["split"] == "test" for i in indices)
    # 60 对 manifest 的 slot 总数（与协议 change_slots 自检一致）
    total_slots = sum(len(rows_by_index[i]["entities"]) for i in indices)
    assert total_slots == 347


def test_pair_protocol_change_uses_compare_semantics() -> None:
    """本地头 60 对 change macro-F1 必须用 macro_f1_compare 口径（0.6409）。"""
    from starlife_eval.verifiers.metrics import macro_f1_compare, macro_f1_train

    man = _manifest60_json()
    indices = [int(i) for i in man["sample_indices"]]
    preds = _local_pair_preds("pilot_v2g")
    rows_by_index = _pair_row_by_index()
    manifest = _build_manifest60()
    pairs_by_index = {p.index: p for p in manifest.pairs}

    true_labels: list[str] = []
    pred_labels: list[str] = []
    for i in indices:
        pair = pairs_by_index[i]
        changes = preds[i].changes
        for entity in pair.entities:
            true_labels.append(pair.changes.get(entity.slot, "none"))
            pred_labels.append(changes.get(entity.slot, "none"))

    true_idx = [CHANGE_INDEX[x] for x in true_labels]
    pred_idx = [CHANGE_INDEX[x] for x in pred_labels]
    compare = macro_f1_compare(true_idx, pred_idx, 4)
    train, _ = macro_f1_train(true_idx, pred_idx, 4)
    assert round(compare, 4) == 0.6409
    # 这组真实数据上两种口径恰好同值；单测里另有一组能区分的构造例子
    assert round(train, 4) == round(compare, 4)

    result = evaluate_pair_protocol(manifest, indices, preds)
    assert result["change_macro_f1"] == round(compare, 4)


# --------------------------------------------------------- 60-pair protocol: cloud


def _parse_json_like_eval_compare(text: str) -> dict:
    """测试内镜像 eval_compare_cloud.parse_json（第 90-102 行）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - 原脚本同样兜底
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i : j + 1])
        raise


def _norm_pred_like_eval_compare(p: Any) -> dict:
    """测试内镜像 eval_compare_cloud.norm_pred（第 105-126 行）。"""
    if not isinstance(p, dict):
        return {"changes": {}, "relations": []}
    ch = p.get("changes", {})
    if isinstance(ch, list):
        nch: dict[int, str] = {}
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


def _build_cloud_preds() -> tuple[dict[int, PairPrediction], int]:
    """把 eval_cloud_raw.jsonl 解析成 PairPrediction；返回 (preds, 兜底条数)。

    原脚本 ``cloud_pred`` 对解析失败（含空 raw）的 pair 返回空预测——完全镜像。
    """
    records = _cloud_records()
    indices = [int(i) for i in _manifest60_json()["sample_indices"]]
    preds: dict[int, PairPrediction] = {}
    n_fallback = 0
    for pair_index in indices:
        try:
            parsed = _norm_pred_like_eval_compare(
                _parse_json_like_eval_compare(records[pair_index]["raw"])
            )
        except Exception:  # noqa: BLE001 - 与 eval_compare_cloud.cloud_pred 一致
            parsed = {"changes": {}, "relations": []}
            n_fallback += 1
        preds[pair_index] = PairPrediction(
            pair_index=pair_index,
            changes=parsed["changes"],
            relations=[
                RelationPrediction(
                    from_slot=rel.get("from_slot"),
                    to_slot=rel.get("to_slot"),
                    type=rel.get("type"),
                    rel_pos=rel.get("rel_pos"),
                    contact=rel.get("contact"),
                    support=rel.get("support"),
                )
                for rel in parsed["relations"]
            ],
        )
    return preds, n_fallback


@pytest.mark.parametrize("model", MODELS)
def test_pair_protocol_cloud_parity_with_reference(model: str) -> None:
    """60 对云端协议：deepseek-flash 回放缓存 -> 0.9251/0.7781/0.2784/0.4898/0.9286。"""
    man = _manifest60_json()
    indices = [int(i) for i in man["sample_indices"]]
    preds, n_fallback = _build_cloud_preds()
    assert n_fallback >= 1, "fixtures 里应有解析失败/空 raw 的记录（原脚本按空预测兜底）"

    result = evaluate_pair_protocol(_build_manifest60(), indices, preds)
    expected = _oracle()[model]["pair_protocol"]["cloud_deepseek_flash"]
    assert_metrics_equal(result, expected, path=f"{model}/pair_protocol/cloud")

    # 关键校准点（ARCHITECTURE §10）：云端 change macro-F1 0.7781 依赖 compare 口径
    assert result["change_slots"] == 347
    assert result["relation_pairs"] == 176
    assert result["change_macro_f1"] == 0.7781


# ------------------------------------------------------------- runtime (cloud)


def test_cloud_latency_and_tokens_parity() -> None:
    """云端延迟 mean/median/max 与 token 汇总必须复现原脚本（真实测量值）。"""
    records = _cloud_records()
    indices = [int(i) for i in _manifest60_json()["sample_indices"]]
    latencies_s = [float(records[i]["latency_s"]) for i in indices]
    usages = [records[i].get("usage", {}) for i in indices]

    mean_ms, p50_ms, max_ms = from_latencies_s(latencies_s)
    ref_lat = _oracle()["pilot_v2"]["pair_protocol"]["cloud_latency_s"]
    # reference 把秒值 round 到 3 位后写出
    assert round(mean_ms / 1000.0, 3) == ref_lat["mean"] == 6.407
    assert round(p50_ms / 1000.0, 3) == ref_lat["median"] == 6.01
    assert round(max_ms / 1000.0, 3) == ref_lat["max"] == 11.376

    prompt, completion = summarize_tokens(usages)
    ref_tokens = _oracle()["pilot_v2"]["pair_protocol"]["cloud_tokens"]
    assert prompt == ref_tokens["prompt"] == 82356
    assert completion == ref_tokens["completion"] == 81524


def test_reference_oracle_matches_standalone_pair_protocol_files() -> None:
    """reference_oracle.json 内嵌的协议段应与独立 dump 的 ref_pair_protocol_*.json 一致。"""
    for model in MODELS:
        standalone = json.loads(
            (FIXTURES / f"ref_pair_protocol_{model}.json").read_text(encoding="utf-8")
        )
        embedded = _oracle()[model]["pair_protocol"]
        assert_metrics_equal(embedded, standalone, path=f"{model}/standalone")
