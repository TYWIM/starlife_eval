"""models 层云端适配器单测：raw 解析 / 离线回放 / OpenAI 兼容 live（不真联网）。

覆盖：
- ``parse_json`` / ``norm_pred`` 逐行对照 eval_compare_cloud.py 的语义；
- ``ReplayCloudAdapter`` 的缺 pair 报错、latency/usage 汇总、解析失败口径；
- ``ChatVLMAdapter`` 的请求体/提示词/max_tokens(默认 4000)/缓存断点续跑/
  response_format 降级/空 content 重试后降级空预测/key redact；
- 真实 deepseek 缓存 fixture 的端到端回放，并复算 60 对协议里的云端数字
  （change_top1 0.9251 / change_macro_f1 0.7781 …）与 reference_oracle.json 对齐。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from starlife_eval.config import EvalConfig, ModelSpec
from starlife_eval.models import registry as registry_mod
from starlife_eval.models.openai_compat import (
    ChatVLMAdapter,
    HTTPPostError,
    build_prompt,
    endpoint_url,
    image_data_url,
)
from starlife_eval.models.replay_cloud import (
    MissingCloudPairs,
    ReplayCloudAdapter,
    norm_pred,
    parse_json,
)
from starlife_eval.types import DatasetManifest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------- parse / norm_pred


def test_parse_json_variants() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"changes": [{"slot": 0, "type": "none"}]}\n```') == {
        "changes": [{"slot": 0, "type": "none"}]
    }
    assert parse_json('```\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('前缀 {"x": 2} 后缀') == {"x": 2}
    with pytest.raises(Exception):
        parse_json("完全不是 JSON")


def test_norm_pred_list_and_dict() -> None:
    p = {
        "changes": [{"slot": "2", "type": "appeared"}, {"slot": 1}, {"oops": 1}],
        "relations": [{"from_slot": 0, "to_slot": 1, "type": "near"}, "bad", {"from_slot": 1}],
    }
    out = norm_pred(p)
    assert out["changes"] == {2: "appeared", 1: "none"}
    assert len(out["relations"]) == 2

    out2 = norm_pred({"changes": {"3": "displaced"}, "relations": {"k": {"from_slot": 9, "to_slot": 8}}})
    assert out2["changes"] == {3: "displaced"}
    assert out2["relations"] == [{"from_slot": 9, "to_slot": 8}]

    assert norm_pred(None) == {"changes": {}, "relations": []}
    assert norm_pred({"changes": "bad", "relations": []}) == {"changes": {}, "relations": []}
    # 与原脚本一致：relations 既不是 list 也不是 dict 时直接抛错（cloud_pred 会兜底成空预测）
    with pytest.raises(TypeError):
        norm_pred({"changes": {}, "relations": 5})


# --------------------------------------------------------------- replay adapter


def _write_jsonl(path: Path, records: list[Any]) -> None:
    lines = []
    for rec in records:
        lines.append(rec if isinstance(rec, str) else json.dumps(rec, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_cache(path: Path) -> Path:
    _write_jsonl(path, [
        "not-a-json-line",
        {
            "pair_index": 0,
            "pair_id": "000000",
            "raw": '```json\n{"changes":[{"slot":0,"type":"appeared"},{"slot":1,"type":"none"}],'
                   '"relations":[{"from_slot":0,"to_slot":1,"type":"near","rel_pos":"front",'
                   '"contact":true,"support":false}]}\n```',
            "latency_s": 1.5,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
        {
            "pair_index": 1,
            "pair_id": "000001",
            "raw": '{"changes":{"0":"disappeared"},"relations":[{"from_slot":2,"to_slot":0,'
                   '"type":"on","rel_pos":"back","contact":false,"support":true}]}',
            "latency_s": 2.5,
            "usage": {"prompt_tokens": 300, "completion_tokens": 40},
        },
        {"pair_index": 2, "pair_id": "000002", "raw": "", "latency_s": None, "usage": {}},
        {"raw": "missing pair_index"},
    ])
    return path


def test_replay_cloud_predict_and_errors(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache.jsonl")
    adapter = ReplayCloudAdapter(cache)

    preds = adapter.predict([0, 1, 2])
    assert sorted(preds) == [0, 1, 2]

    p0 = preds[0]
    assert p0.pair_index == 0
    assert p0.changes == {0: "appeared", 1: "none"}
    assert p0.types == {}
    assert len(p0.relations) == 1
    r = p0.relations[0]
    assert (r.from_slot, r.to_slot, r.type, r.rel_pos, r.contact, r.support) == (0, 1, "near", "front", True, False)

    p1 = preds[1]
    assert p1.changes == {0: "disappeared"}
    assert p1.relations[0].from_slot == 2 and p1.relations[0].support is True

    # 空 raw → 按 cloud_pred 口径为空预测，并记录解析失败
    assert preds[2].changes == {} and preds[2].relations == []
    assert adapter.parse_failures == [2]

    # 缺 pair 明确报错（不默认空预测）
    with pytest.raises(MissingCloudPairs, match="99"):
        adapter.predict([0, 99])
    with pytest.raises(FileNotFoundError):
        ReplayCloudAdapter(tmp_path / "nope.jsonl")


def test_replay_cloud_runtime_stats(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache.jsonl")
    adapter = ReplayCloudAdapter(cache)
    stats = adapter.runtime_stats([0, 1, 2])
    assert stats.n_calls == 2  # 只有 pair 0/1 有 latency/usage
    assert stats.latency_ms_mean == pytest.approx(2000.0)
    assert stats.latency_ms_p50 == pytest.approx(2000.0)
    assert stats.latency_ms_max == pytest.approx(2500.0)
    assert stats.tokens_prompt == 400 and stats.tokens_completion == 60
    assert stats.cost_usd is None and stats.cost_basis


# ------------------------------------------------------------- prompt / endpoint


def test_build_prompt_matches_reference_template() -> None:
    entities = [
        {"slot": 0, "bbox": [0.0, 342.4, 1023.0, 767.0], "type_key": "table"},
        {"slot": 1, "bbox": [136.16, 420.04, 236.0, 508.96], "type_key": "mug"},
    ]
    # 参考 eval_compare_cloud.py:44-56 的模板（独立抄写，用于逐字符比对）
    slots = [{"slot": e["slot"], "bbox": [round(v, 1) for v in e["bbox"]]} for e in entities]
    expected = (
        "你是桌面场景的结构化标注器。给你同一场景的 before 和 after 两张图，以及图中实体的槽位与像素框。\n"
        f"实体：{json.dumps(slots, ensure_ascii=False)}\n"
        "只输出一个 JSON 对象，不要解释、不要代码块，格式：\n"
        '{"changes":[{"slot":0,"type":"none|appeared|displaced|disappeared"}],'
        '"relations":[{"from_slot":0,"to_slot":1,"type":"on|in|contains|held_by|part_of|belongs_to|near|next_to|facing|above|below",'
        '"rel_pos":"top|inside|front|back|left|right|front_left|front_right|back_left|back_right",'
        '"contact":true,"support":false}]}\n'
        "规则：changes 必须覆盖所有槽位（没变化写 none）；relations 只描述 after 帧里同时存在的实体，"
        "每个事实只写一次，from_slot 是主体、to_slot 是宿主。"
    )
    prompt = build_prompt(entities)
    assert prompt == expected
    assert '"slot": 0, "bbox": [0.0, 342.4, 1023.0, 767.0]' in prompt
    assert '"bbox": [136.2, 420.0, 236.0, 509.0]' in prompt  # round 到 1 位
    for rel in ("on|in|contains|held_by|part_of|belongs_to|near|next_to|facing|above|below").split("|"):
        assert rel in prompt
    for pos in ("top|inside|front|back|left|right|front_left|front_right|back_left|back_right").split("|"):
        assert pos in prompt


def test_endpoint_and_data_url() -> None:
    assert endpoint_url("https://api.example.com/v1") == "https://api.example.com/v1/chat/completions"
    assert endpoint_url("https://api.example.com/v1/") == "https://api.example.com/v1/chat/completions"
    assert endpoint_url("https://api.example.com/v1/chat/completions") == "https://api.example.com/v1/chat/completions"
    url = image_data_url(b"abc")
    assert url.startswith("data:image/jpeg;base64,")
    assert url.endswith("YWJj")  # base64("abc")


# --------------------------------------------------------------- live adapter


def _make_live_dataset(root: Path, n_pairs: int = 2) -> DatasetManifest:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_pairs):
        before = f"frames/{i:06d}_before.jpg"
        after = f"frames/{i:06d}_after.jpg"
        (root / "frames").mkdir(exist_ok=True)
        (root / before).write_bytes(b"before-bytes")
        (root / after).write_bytes(b"after-bytes")
        rows.append({
            "pair_id": f"{i:06d}", "split": "test", "vocab_version": "1.0.0",
            "before": before, "after": after,
            "entities": [{"slot": 0, "bbox": [1.2, 2.3, 3.4, 4.5], "type_key": "mug", "present_after": True}],
            "changes": [], "relations": [], "negative": False, "meta": {},
        })
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    from starlife_eval.dataset import load_manifest

    return load_manifest(root)


def _ok_payload(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }


def _valid_content() -> str:
    return '{"changes":[{"slot":0,"type":"displaced"}],"relations":[]}'


def test_chat_vlm_refuses_without_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    called: list[int] = []

    def transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called.append(1)
        raise AssertionError("allow_live=False 时不得联网")

    monkeypatch.setenv("DUMMY_KEY", "sk-should-not-be-used")
    adapter = ChatVLMAdapter("https://api.example.com/v1", "m", "DUMMY_KEY", allow_live=False, transport=transport)
    with pytest.raises(RuntimeError, match="无缓存"):
        adapter.predict(manifest, [0])
    assert called == []


def test_chat_vlm_requires_env_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.delenv("MISSING_KEY_FOR_TEST", raising=False)
    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "MISSING_KEY_FOR_TEST", allow_live=True,
        transport=lambda *a, **k: _ok_payload(_valid_content()),
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY_FOR_TEST"):
        adapter.predict(manifest, [0])


def test_chat_vlm_live_request_body_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-SECRET-123")
    cache = tmp_path / "cloud" / "live_test.jsonl"
    bodies: list[dict[str, Any]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        bodies.append(body)
        assert url == "https://api.example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-SECRET-123"
        assert headers["Content-Type"] == "application/json"
        return _ok_payload('```json\n' + _valid_content() + "\n```")

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "test-model", "TEST_API_KEY",
        allow_live=True, cache_path=cache, transport=transport,
    )
    preds = adapter.predict(manifest, [0])
    assert preds[0].changes == {0: "displaced"}
    assert len(bodies) == 1
    body = bodies[0]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 4000
    assert body["response_format"] == {"type": "json_object"}
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text" and "结构化标注器" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["image_url"]["url"] != content[2]["image_url"]["url"]  # before != after

    # 写缓存且格式与 eval_compare_cloud raw jsonl 兼容
    lines = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["pair_index"] == 0 and lines[0]["model"] == "test-model"
    assert lines[0]["raw"] == '```json\n' + _valid_content() + "\n```"
    assert isinstance(lines[0]["latency_s"], float)

    stats = adapter.runtime_stats([0])
    assert stats.tokens_prompt == 123 and stats.tokens_completion == 45
    assert stats.n_calls == 1 and stats.cost_usd is None
    assert stats.latency_ms_mean is not None and stats.latency_ms_mean >= 0.0

    # 第二次：allow_live=False + 命中缓存 → 完全离线；不调用 transport
    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("命中缓存时不应联网")

    replay = ChatVLMAdapter(
        "https://api.example.com/v1", "test-model", "TEST_API_KEY",
        allow_live=False, cache_path=cache, transport=forbidden,
    )
    preds2 = replay.predict(manifest, [0])
    assert preds2[0].changes == {0: "displaced"}
    assert replay.calls[0]["cached"] is True
    stats2 = replay.runtime_stats([0])
    assert stats2.tokens_prompt == 123

    # 未缓存的 pair + allow_live=False → 明确报错
    with pytest.raises(RuntimeError, match="无缓存"):
        replay.predict(manifest, [1])


def test_chat_vlm_response_format_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    bodies: list[dict[str, Any]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        bodies.append(body)
        if len(bodies) == 1:
            raise HTTPPostError(400, '{"error": "response_format is not supported"}')
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, retry_sleep_s=0.0,
    )
    preds = adapter.predict(manifest, [0])
    assert preds[0].changes == {0: "displaced"}
    assert len(bodies) == 2
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
    assert bodies[1]["temperature"] == 0 and bodies[1]["max_tokens"] == 4000


def test_chat_vlm_empty_content_retries_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            return _ok_payload("")  # reasoning tokens 吃光 max_tokens
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "deepseek-flash", "TEST_API_KEY",
        allow_live=True, transport=transport, retry_sleep_s=0.0,
    )
    preds = adapter.predict(manifest, [0])
    assert preds[0].changes == {0: "displaced"}
    assert len(calls) == 2


def test_chat_vlm_empty_after_retry_degrades_to_empty_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gpt-6.1-sol 真实故障复现：两次空 content 也不许 raise，按空预测兜底并写缓存。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    cache = tmp_path / "cloud" / "live_gpt.jsonl"
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        return {
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 757},
        }

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "gpt-6.1-sol", "TEST_API_KEY",
        allow_live=True, cache_path=cache, transport=transport, retry_sleep_s=0.0,
    )
    preds = adapter.predict(manifest, [0])
    assert preds[0].changes == {} and preds[0].relations == []  # 整对按全错计（计入分母）
    assert len(calls) == 2  # 空 content 重试一次
    assert adapter.live_empty == [0] and adapter.parse_failures == [0]

    # 原始空响应 + usage 也要写缓存（离线回放同一兜底路径）
    lines = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["pair_index"] == 0 and lines[0]["raw"] == ""
    assert lines[0]["usage"]["completion_tokens"] == 757
    stats = adapter.runtime_stats([0])
    assert stats.n_calls == 1 and stats.tokens_completion == 757

    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("离线回放空缓存记录不应联网")

    replay = ChatVLMAdapter(
        "https://api.example.com/v1", "gpt-6.1-sol", "TEST_API_KEY",
        allow_live=False, cache_path=cache, transport=forbidden,
    )
    preds2 = replay.predict(manifest, [0])
    assert preds2[0].changes == {} and replay.parse_failures == [0]


def test_chat_vlm_max_tokens_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    bodies: list[dict[str, Any]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        bodies.append(body)
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, retry_sleep_s=0.0, max_tokens=6000,
    )
    adapter.predict(manifest, [0])
    assert bodies[0]["max_tokens"] == 6000

    # registry 从 ModelSpec.extra["max_tokens"] 透传，缺省 4000
    spec = ModelSpec.from_dict({
        "id": "gpt-6.1-sol", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
        "model": "gpt-6.1-sol", "api_key_env": "TEST_API_KEY", "max_tokens": 5000,
    })
    cfg = _cfg(tmp_path, [spec], allow_live=True)
    prepared = registry_mod.build_model(spec, cfg, manifest, live=True)
    assert prepared.available and prepared.adapter.max_tokens == 5000
    default_spec = ModelSpec.from_dict({
        "id": "gemini-3.8-flash", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
        "model": "gemini-3.8-flash", "api_key_env": "TEST_API_KEY",
    })
    prepared2 = registry_mod.build_model(default_spec, cfg, manifest, live=True)
    assert prepared2.available and prepared2.adapter.max_tokens == 4000


def test_registry_note_for_live_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """live 空响应要在 run.notes 里累计计数（Lead 要求），且不中断整场。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("REG_LIVE_KEY", "sk-key")
    spec = ModelSpec.from_dict({
        "id": "gpt-6.1-sol", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
        "model": "gpt-6.1-sol", "api_key_env": "REG_LIVE_KEY",
    })
    cfg = _cfg(tmp_path, [spec], allow_live=True)
    prepared = registry_mod.build_model(spec, cfg, manifest, live=True)
    prepared.adapter.transport = lambda *a, **k: {
        "choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }
    run = registry_mod.execute(prepared, manifest, [0], dataset_dir=tmp_path / "ds", live=True)
    assert run.pairs[0].changes == {}
    assert any("live 空响应 1 对" in note for note in run.notes)



def test_chat_vlm_redacts_api_key_in_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    secret = "sk-SUPER-SECRET-987"
    monkeypatch.setenv("TEST_API_KEY", secret)

    def transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ConnectionError(f"auth failed for {secret}")

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, retries=1, retry_sleep_s=0.0,
    )
    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict(manifest, [0])
    message = str(excinfo.value)
    assert secret not in message
    assert "***" in message


def test_chat_vlm_missing_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    (tmp_path / "ds" / "frames" / "000000_before.jpg").unlink()
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=lambda *a, **k: _ok_payload(_valid_content()),
    )
    with pytest.raises(FileNotFoundError, match="000000_before"):
        adapter.predict(manifest, [0])


# ------------------------------------- 真实 deepseek 缓存回放 + 60 对协议数字复算


def _macro_f1_compare(y_true: list[int], y_pred: list[int], n: int) -> float:
    """eval_compare_cloud.py:31-41（跳过 tp==0 的类）——本测试用于独立复算云端数字。"""
    f1s = []
    for c in range(n):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        if tp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def test_replay_real_deepseek_cache_matches_oracle() -> None:
    cache = FIXTURES / "eval_cloud_raw.jsonl"
    sample_path = FIXTURES / "ref_manifest_sample60.json"
    oracle_path = FIXTURES / "reference_oracle.json"
    if not (cache.exists() and sample_path.exists() and oracle_path.exists()):
        pytest.skip("缺少 eval_cloud_raw.jsonl / ref_manifest_sample60.json / reference_oracle.json")

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    indices = sample["sample_indices"]
    rows = sample["rows"]
    assert len(indices) == 60 and len(rows) == 60

    adapter = ReplayCloudAdapter(cache)
    preds = adapter.predict(indices)
    assert sorted(preds) == sorted(indices)
    # 16 对 content 为空（reasoning tokens 吃光 max_tokens）+ 3 对 JSON 被截断
    # 均按原脚本 cloud_pred 口径计为空预测（评测口径如此，不是我们造数）
    assert len(adapter.parse_failures) == 19

    # latency / usage 与参考一致（reference_oracle 里是 round(...,3) 的值）
    stats = adapter.runtime_stats(indices)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))["pilot_v2g"]["pair_protocol"]
    assert stats.latency_ms_mean == pytest.approx(oracle["cloud_latency_s"]["mean"] * 1000, abs=1.0)
    assert stats.latency_ms_p50 == pytest.approx(oracle["cloud_latency_s"]["median"] * 1000, abs=1.0)
    assert stats.latency_ms_max == pytest.approx(oracle["cloud_latency_s"]["max"] * 1000, abs=1.0)
    assert stats.tokens_prompt == oracle["cloud_tokens"]["prompt"]
    assert stats.tokens_completion == oracle["cloud_tokens"]["completion"]

    # 逐行复刻 eval_compare_cloud.main 的 evaluate()，复算 60 对云端指标
    change_labels = ["none", "appeared", "displaced", "disappeared"]
    ch_true: list[str] = []
    ch_pred: list[str] = []
    rel_total = rel_hit = 0
    pos_total = pos_hit = 0
    attr_total = attr_hit = 0
    for i, row in zip(indices, rows):
        pred = preds[i]
        pch = {int(k): str(v) for k, v in pred.changes.items()}
        gt_change = {c["slot"]: c["type"] for c in row["changes"]}
        for s in (e["slot"] for e in row["entities"]):
            ch_true.append(gt_change.get(s, "none"))
            ch_pred.append(pch.get(s, "none"))
        p_rel = {(x.from_slot, x.to_slot): x for x in pred.relations}
        for g in row["relations"]:
            key = (g["from_slot"], g["to_slot"])
            rel_total += 1
            cand = p_rel.get(key)
            if cand is not None and cand.type == g["type"]:
                rel_hit += 1
                if "rel_pos" in g:
                    pos_total += 1
                    if cand.rel_pos == g["rel_pos"]:
                        pos_hit += 1
                attr_total += 2
                attr_hit += int(bool(cand.contact) == bool(g["contact"]))
                attr_hit += int(bool(cand.support) == bool(g["support"]))

    got = {
        "change_slots": len(ch_true),
        "change_top1": round(float(np.mean([a == b for a, b in zip(ch_true, ch_pred)])), 4),
        "change_macro_f1": round(
            _macro_f1_compare([change_labels.index(x) for x in ch_true],
                              [change_labels.index(x) for x in ch_pred], 4), 4),
        "relation_pairs": rel_total,
        "relation_type_acc": round(rel_hit / rel_total, 4),
        "rel_pos_acc": round(pos_hit / pos_total, 4),
        "attrs_acc": round(attr_hit / attr_total, 4),
    }
    cloud = oracle["cloud_deepseek_flash"]
    assert got["change_slots"] == cloud["change_slots"] == 347
    assert got["change_top1"] == cloud["change_top1"] == 0.9251
    assert got["change_macro_f1"] == cloud["change_macro_f1"] == 0.7781
    assert got["relation_pairs"] == cloud["relation_pairs"] == 176
    assert got["relation_type_acc"] == cloud["relation_type_acc"] == 0.2784
    assert got["rel_pos_acc"] == cloud["rel_pos_acc"] == 0.4898
    assert got["attrs_acc"] == cloud["attrs_acc"] == 0.9286


# ------------------------------------------------------------------ registry


def _cfg(tmp_path: Path, models: list[ModelSpec], *, allow_live: bool = False) -> EvalConfig:
    return EvalConfig(data_root=tmp_path, models=models, allow_live=allow_live)


def test_registry_unavailable_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.delenv("REG_TEST_KEY", raising=False)
    cfg = _cfg(tmp_path, [])

    # 缺特征/缺 checkpoint → local_heads unavailable
    spec = ModelSpec.from_dict({"id": "local", "kind": "local_heads"})
    prepared = registry_mod.build_model(spec, cfg, manifest)
    assert prepared.kind == "unavailable" and not prepared.available
    assert "local_heads" in prepared.na and "spatial_v2.npz" in prepared.na["local_heads"]

    # 缺回放缓存 → unavailable
    spec = ModelSpec.from_dict({"id": "cloud", "kind": "replay_cloud"})
    prepared = registry_mod.build_model(spec, cfg, manifest)
    assert prepared.kind == "unavailable" and "cloud_cache" in prepared.na

    # chat_vlm：未 --live 且无缓存 → unavailable（禁止联网）
    spec = ModelSpec.from_dict({
        "id": "gemini", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
        "model": "gemini-3.8-flash", "api_key_env": "REG_TEST_KEY",
    })
    prepared = registry_mod.build_model(spec, cfg, manifest, live=False)
    assert prepared.kind == "unavailable" and "live" in prepared.na

    # chat_vlm：--live 但无 key → unavailable，且消息里不得出现任何 key 值
    prepared = registry_mod.build_model(spec, cfg, manifest, live=True)
    assert prepared.kind == "unavailable" and "api_key_env" in prepared.na
    assert "REG_TEST_KEY" in prepared.na["api_key_env"]

    # chat_vlm：缓存文件存在但为空（live 跑了一半/刚创建）→ 未 --live 视为 unavailable
    empty_cache = tmp_path / "cloud" / "live_empty.jsonl"
    empty_cache.parent.mkdir(parents=True, exist_ok=True)
    empty_cache.write_text("", encoding="utf-8")
    empty_spec = ModelSpec.from_dict({
        "id": "empty-live", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
        "model": "m", "api_key_env": "REG_TEST_KEY", "source": "cloud/live_empty.jsonl",
    })
    prepared = registry_mod.build_model(empty_spec, cfg, manifest, live=False)
    assert prepared.kind == "unavailable" and "live" in prepared.na

    # 未知 kind
    spec = ModelSpec.from_dict({"id": "weird", "kind": "sorcery"})
    prepared = registry_mod.build_model(spec, cfg, manifest)
    assert prepared.kind == "unavailable" and "kind" in prepared.na

    # unavailable 的 execute 不抛错，原样带 na 返回
    run = registry_mod.execute(prepared, manifest, [0])
    assert run.kind == "unavailable" and run.head is None and run.pairs is None
    assert "kind" in run.na


def test_registry_live_cache_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [])
    spec = ModelSpec.from_dict({"id": "gemini-3.8-flash", "kind": "chat_vlm"})
    assert registry_mod.live_cache_path(cfg, spec) == tmp_path / "cloud" / "live_gemini-3.8-flash.jsonl"


def test_registry_execute_replay_cloud(tmp_path: Path) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    cache = tmp_path / "cloud" / "eval_cloud_raw.jsonl"
    _write_jsonl(cache, [
        {"pair_index": 0, "raw": _valid_content(), "latency_s": 1.0, "usage": {"prompt_tokens": 10}},
        {"pair_index": 1, "raw": '{"changes":{"0":"none"},"relations":[]}', "latency_s": 3.0,
         "usage": {"prompt_tokens": 20}},
    ])
    cfg = _cfg(tmp_path, [ModelSpec.from_dict({"id": "deepseek-flash", "kind": "replay_cloud"})])
    prepared = registry_mod.build_models(cfg, manifest)
    assert len(prepared) == 1 and prepared[0].available
    run = registry_mod.execute(prepared[0], manifest, [0, 1])
    assert run.kind == "replay_cloud"
    assert run.pairs is not None and sorted(run.pairs) == [0, 1]
    assert run.pairs[0].changes == {0: "displaced"}
    assert run.runtime.latency_ms_mean == pytest.approx(2000.0)
    assert run.runtime.tokens_prompt == 30
    assert run.head is None


def test_registry_execute_chat_vlm_offline_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("REG_LIVE_KEY", "sk-live-secret")
    cache = tmp_path / "cloud" / "live_gemini.jsonl"
    _write_jsonl(cache, [
        {"pair_index": 0, "pair_id": "000000", "model": "gemini-3.8-flash", "raw": _valid_content(),
         "latency_s": 2.0, "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    ])
    spec = ModelSpec.from_dict({
        "id": "gemini-3.8-flash",
        "kind": "chat_vlm",
        "base_url": "https://api.example.com/v1",
        "model": "gemini-3.8-flash",
        "api_key_env": "REG_LIVE_KEY",
        "source": "cloud/live_gemini.jsonl",
    })
    cfg = _cfg(tmp_path, [spec], allow_live=False)

    prepared = registry_mod.build_model(spec, cfg, manifest, live=False)
    assert prepared.available and prepared.kind == "chat_vlm"
    # live=False 且只有 pair 0 缓存 → 只回放缓存、绝不联网
    run = registry_mod.execute(prepared, manifest, [0], live=False)
    assert run.kind == "chat_vlm" and run.pairs is not None
    assert run.pairs[0].changes == {0: "displaced"}
    assert "live" in run.na  # 明确标注本次是离线回放
    assert run.runtime.tokens_prompt == 7
    # 请求 pair 1：无缓存 + live=False → 明确报错（不静默空预测）
    with pytest.raises(RuntimeError, match="无缓存"):
        registry_mod.execute(prepared, manifest, [1], live=False)


def test_registry_chat_vlm_live_pass_then_offline_total_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """复刻 Lead 的用法：先用 live 跑 60 对写缓存，再离线用同一缓存总跑。"""
    manifest = _make_live_dataset(tmp_path / "ds", n_pairs=2)
    monkeypatch.setenv("REG_LIVE_KEY", "sk-live-secret")
    spec = ModelSpec.from_dict({
        "id": "gemini-3.8-flash", "kind": "chat_vlm", "mode": "ZERO-SHOT",
        "family": "Google", "provider": "Google",
        "base_url": "https://api.example.com/v1", "model": "gemini-3.8-flash",
        "api_key_env": "REG_LIVE_KEY",
    })
    cache_file = tmp_path / "cloud" / "live_gemini-3.8-flash.jsonl"
    bodies: list[dict[str, Any]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        bodies.append(body)
        return _ok_payload(_valid_content())

    # --- live pass ---
    cfg_live = _cfg(tmp_path, [spec], allow_live=True)
    prepared = registry_mod.build_model(spec, cfg_live, manifest, live=True)
    assert prepared.available and len(bodies) == 0
    prepared.adapter.transport = transport  # 注入假 transport（公开注入点），绝不真联网
    run_live = registry_mod.execute(
        prepared, manifest, [0, 1], dataset_dir=tmp_path / "ds", live=True, cache_path=cache_file
    )
    assert len(bodies) == 2 and sorted(run_live.pairs) == [0, 1]
    assert cache_file.exists()
    lines = [json.loads(x) for x in cache_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [line["pair_index"] for line in lines] == [0, 1]
    assert all(line["model"] == "gemini-3.8-flash" for line in lines)
    assert all(set(line) >= {"pair_index", "model", "raw", "latency_s", "usage"} for line in lines)

    # --- offline total run（live=False，同缓存路径）---
    cfg_offline = _cfg(tmp_path, [spec], allow_live=False)
    prepared2 = registry_mod.build_models(cfg_offline, manifest)[0]
    assert prepared2.available
    prepared2.adapter.transport = lambda *a, **k: (_ for _ in ()).throw(AssertionError("离线总跑不应联网"))
    run = registry_mod.execute(
        prepared2, manifest, [0, 1], dataset_dir=tmp_path / "ds", live=False, cache_path=cache_file
    )
    assert sorted(run.pairs) == [0, 1]
    assert run.pairs[0].changes == {0: "displaced"} and run.pairs[1].changes == {0: "displaced"}
    assert "live" in run.na  # 标注本次为离线回放
    assert run.runtime.n_calls == 2 and run.runtime.tokens_prompt == 2 * 123


def test_registry_execute_all(tmp_path: Path) -> None:
    manifest = _make_live_dataset(tmp_path / "ds")
    _write_jsonl(tmp_path / "cloud" / "eval_cloud_raw.jsonl", [
        {"pair_index": 0, "raw": _valid_content(), "latency_s": 1.0, "usage": {}},
        {"pair_index": 1, "raw": _valid_content(), "latency_s": 1.0, "usage": {}},
    ])
    cfg = _cfg(tmp_path, [
        ModelSpec.from_dict({"id": "deepseek-flash", "kind": "replay_cloud"}),
        ModelSpec.from_dict({"id": "local", "kind": "local_heads"}),
    ])
    prepared = registry_mod.build_models(cfg, manifest)
    runs = registry_mod.execute_all(prepared, manifest, [0, 1])
    assert [r.model_id for r in runs] == ["deepseek-flash", "local"]
    assert runs[0].pairs is not None
    assert runs[1].kind == "unavailable"



def test_chat_vlm_headers_include_browser_ua_and_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """网关前置 Cloudflare（1010）只认浏览器 UA；extra.headers 可再覆盖。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    seen: list[dict[str, str]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        seen.append(headers)
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, headers={"X-Trace": "t1"},
    )
    adapter.predict(manifest, [0])
    assert seen[0]["User-Agent"].startswith("Mozilla/5.0")
    assert seen[0]["X-Trace"] == "t1"
    assert seen[0]["Authorization"] == "Bearer sk-key"


def test_chat_vlm_tolerate_call_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """网关抖动（404/5xx）不应毁掉整场：tolerate 模式下该 pair 记空预测并继续。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        raise HTTPPostError(404, '{"error": {"message": "server_error"}}')

    strict = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, retries=0, retry_sleep_s=0.0,
    )
    with pytest.raises(RuntimeError, match="云端调用失败"):
        strict.predict(manifest, [0])

    lenient = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        transport=transport, retries=0, retry_sleep_s=0.0, tolerate_call_errors=True,
    )
    preds = lenient.predict(manifest, [0, 1])
    assert len(preds) == 2
    assert all(p.changes == {} and p.relations == [] for p in preds.values())
    assert lenient.call_failures == [0, 1]
    assert lenient.calls[0]["error"].startswith("m 云端调用失败")


def test_unavailable_kind_uses_note(tmp_path: Path) -> None:
    """kind=unavailable 的模型原样输出 N/A 原因（如网关模型不支持图像输入）。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    cfg = _cfg(tmp_path, [
        ModelSpec.from_dict({
            "id": "glm-5.3", "kind": "unavailable",
            "note": "该网关模型只支持文本输入",
        }),
    ])
    prepared = registry_mod.build_models(cfg, manifest)
    assert prepared[0].kind == "unavailable"
    assert "只支持文本" in next(iter(prepared[0].na.values()))


def test_chat_vlm_refetch_empty_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """旧缓存 raw 为空（reasoning 吃光 max_tokens）时，refetch_empty=True 应重新调用。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    cache = tmp_path / "cloud" / "live_x.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "pair_index": 0, "pair_id": "000000", "model": "m",
        "raw": "", "latency_s": 1.0, "usage": {},
    }) + "\n", encoding="utf-8")
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        cache_path=cache, transport=transport, refetch_empty=True,
    )
    preds = adapter.predict(manifest, [0])
    assert calls == [1]
    assert preds[0].changes == {0: "displaced"}

    # 未开 refetch_empty 时复用空缓存、不联网 → 空预测
    cache2 = tmp_path / "cloud" / "live_y.jsonl"
    cache2.write_text(json.dumps({
        "pair_index": 0, "pair_id": "000000", "model": "m",
        "raw": "", "latency_s": 1.0, "usage": {},
    }) + "\n", encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("不应联网")

    off = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=False,
        cache_path=cache2, transport=forbidden,
    )
    preds2 = off.predict(manifest, [0])
    assert preds2[0].changes == {}


def test_chat_vlm_refetch_unparsable_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """旧缓存解析不出 JSON（推理把正文吃掉）时，refetch_unparsable=True 应重新调用。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    cache = tmp_path / "cloud" / "live_z.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "pair_index": 0, "pair_id": "000000", "model": "m",
        "raw": "<think>long reasoning only</think>", "latency_s": 2.0, "usage": {},
    }) + "\n", encoding="utf-8")
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        cache_path=cache, transport=transport, refetch_unparsable=True,
    )
    preds = adapter.predict(manifest, [0])
    assert calls == [1]
    assert preds[0].changes == {0: "displaced"}


def test_registry_chat_vlm_cache_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """extra.cache_path 可把同一模型的 live 缓存指到别的文件。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    cfg = _cfg(tmp_path, [
        ModelSpec.from_dict({
            "id": "m", "kind": "chat_vlm", "base_url": "https://api.example.com/v1",
            "model": "m", "api_key_env": "TEST_API_KEY",
            "cache_path": "cloud/live_custom.jsonl",
        }),
    ])
    prepared = registry_mod.build_models(cfg, manifest, live=True)
    assert prepared[0].kind == "chat_vlm"
    assert prepared[0].adapter is not None
    cache_path = Path(str(prepared[0].adapter.cache_path))
    assert cache_path.parent.name == "cloud" and cache_path.name == "live_custom.jsonl"
    assert cache_path.is_absolute()


def test_chat_vlm_refetch_invalid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """缓存 JSON 合法但标签是 prompt 回显（none|appeared|...）→ refetch_invalid 重新调用。"""
    manifest = _make_live_dataset(tmp_path / "ds")
    monkeypatch.setenv("TEST_API_KEY", "sk-key")
    cache = tmp_path / "cloud" / "live_i.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    echo = '{"changes":[{"slot":0,"type":"none|appeared|displaced|disappeared"}],"relations":[]}'
    cache.write_text(json.dumps({
        "pair_index": 0, "pair_id": "000000", "model": "m",
        "raw": echo, "latency_s": 1.0, "usage": {},
    }) + "\n", encoding="utf-8")
    calls: list[int] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        calls.append(1)
        return _ok_payload(_valid_content())

    adapter = ChatVLMAdapter(
        "https://api.example.com/v1", "m", "TEST_API_KEY", allow_live=True,
        cache_path=cache, transport=transport, refetch_invalid=True,
    )
    preds = adapter.predict(manifest, [0])
    assert calls == [1]
    assert preds[0].changes == {0: "displaced"}
