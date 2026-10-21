"""OpenAI 兼容 Chat Completions 视觉适配器（两图 + 槽位表 → 结构化 JSON）。

请求体与提示词逐行移植 ``starlife/starlife_scripts/eval_compare_cloud.py``：

- ``build_prompt`` (44-56 行)：11 关系 / 10 方位 schema；实体 bbox round 到 1 位小数；
- ``call_cloud``  (59-87 行)：temperature=0、max_tokens（默认 4000，见下）、
  response_format json_object、before/after 两张 data URL（顺序 fixed：text, before, after）、
  失败重试。

在参考实现之上的**生产化**（不改变请求语义）：
- ``response_format`` 服务端不支持时自动降级重试（去掉该字段）；
- content 为空（reasoning tokens 吃光预算）时先重试一次；仍空则**按
  eval_compare_cloud.cloud_pred 的口径降级为空预测**（整对按全错计、不中断整场），
  原始（空）响应与 usage 照常写缓存，便于离线复现；
- max_tokens 默认 4000（参考脚本是 2000，但 gpt-6.1-sol 单次 reasoning 757 tokens、
  gemini 337 tokens，2000 会截断；可由 ``ModelSpec.extra["max_tokens"]`` 覆盖）；
- 成功一对立即 append 到 live 缓存（格式与 eval_compare_cloud raw jsonl 兼容），
  同 pair 命中缓存则跳过（断点续跑 + 可离线复现）；
- API key 只从 ``os.environ[api_key_env]`` 读；任何异常/日志都做 redact，绝不含 key；
- ``allow_live=False`` 时绝不发起网络调用（只允许读缓存）。

测试注入约定：``transport`` 是**唯一**的 HTTP 注入点（公开属性）。注入后任何路径都
只走注入实现，不会 fallback 到 :func:`_http_post_json`（生产默认值）。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..types import DatasetManifest, PairPrediction, RuntimeStats
from .replay_cloud import (
    load_raw_records,
    parse_json,
    to_pair_prediction,
    write_raw_record,
)

# 与 eval_compare_cloud.py:50-53 build_prompt 中的 schema 完全一致
RELATION_SCHEMA = "on|in|contains|held_by|part_of|belongs_to|near|next_to|facing|above|below"
REL_POS_SCHEMA = "top|inside|front|back|left|right|front_left|front_right|back_left|back_right"

Transport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], dict[str, Any]]


class HTTPPostError(RuntimeError):
    """HTTP 非 2xx；``detail`` 为响应体片段（不含请求头/key）。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail[:400]}")
        self.status = int(status)
        self.detail = detail


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def image_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def endpoint_url(base_url: str) -> str:
    """OpenAI 兼容端点：``<base_url>/chat/completions``（base_url 已含则原样）。"""
    url = str(base_url).rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def build_prompt(entities: list[dict[str, Any]]) -> str:
    """ported from eval_compare_cloud.py:44-56 build_prompt（逐字符一致）。"""
    slots = [{"slot": e["slot"], "bbox": [round(v, 1) for v in e["bbox"]]} for e in entities]
    return (
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


def _http_post_json(url: str, body: Mapping[str, Any], headers: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    """stdlib HTTP 实现（不引入额外依赖）；HTTPError 的响应体只做诊断用。"""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 4xx/5xx：读响应体给调用方判断
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception:  # noqa: BLE001
            pass
        raise HTTPPostError(int(exc.code), detail) from None
    except urllib.error.URLError as exc:
        raise ConnectionError(f"无法连接 {url}: {exc.reason}") from None


def _looks_like_response_format_error(detail: str) -> bool:
    low = detail.lower()
    return any(
        token in low
        for token in ("response_format", "json_object", "not supported", "unsupported", "invalid parameter")
    )


def _strip_think(text: str) -> str:
    """去掉推理型模型输出的 <think>…</think> 段落，只留最终正文（无闭合标签时按思考截断处理）。"""
    raw = str(text or "")
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[-1].strip()
    if raw.lstrip().startswith("<think>"):
        return ""
    return raw


def _json_pred_ok(payload: Mapping[str, Any]) -> bool:
    """缓存里的 JSON 是否给出了合法的 change/relation 标签（防止把 prompt 回显当答案）。"""
    allowed_change = {"none", "appeared", "displaced", "disappeared"}
    allowed_rel = set(RELATION_SCHEMA.split("|"))
    allowed_pos = set(REL_POS_SCHEMA.split("|"))
    changes = payload.get("changes")
    if isinstance(changes, list):
        for item in changes:
            if isinstance(item, Mapping) and str(item.get("type")) not in allowed_change:
                return False
    relations = payload.get("relations")
    if isinstance(relations, list):
        for item in relations:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("type")) not in allowed_rel:
                return False
            if item.get("rel_pos") is not None and str(item.get("rel_pos")) not in allowed_pos:
                return False
    return True


def _extract_content(payload: Mapping[str, Any]) -> str | None:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


class ChatVLMAdapter:
    """OpenAI 兼容视觉模型适配器。

    Parameters
    ----------
    base_url:
        API base（自动补 ``/chat/completions``）。
    model:
        请求体里的 model id。
    api_key_env:
        读取 key 的环境变量名；``None`` 时只允许离线缓存回放。
    allow_live:
        ``False`` 时禁止任何网络调用（只读缓存/报错），CLI 的 ``--live`` 映射到这里。
    cache_path:
        live 结果 append 的 raw jsonl；同 pair 命中则跳过（断点续跑）。
    max_tokens:
        默认 4000（reasoning 模型需要更大预算；参考脚本用 2000）。
    transport:
        唯一的 HTTP 注入点，默认 :func:`_http_post_json`；注入后绝不 fallback 到真网络。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        *,
        allow_live: bool = False,
        cache_path: str | Path | None = None,
        timeout_s: float = 120.0,
        retries: int = 2,
        retry_sleep_s: float = 2.0,
        max_tokens: int = 4000,
        headers: Mapping[str, str] | None = None,
        tolerate_call_errors: bool = False,
        refetch_empty: bool = False,
        refetch_unparsable: bool = False,
        refetch_invalid: bool = False,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = str(base_url)
        self.model = str(model)
        self.api_key_env = api_key_env
        self.allow_live = bool(allow_live)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.retry_sleep_s = float(retry_sleep_s)
        self.max_tokens = int(max_tokens)
        self.extra_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self.tolerate_call_errors = bool(tolerate_call_errors)
        self.refetch_empty = bool(refetch_empty)
        self.refetch_unparsable = bool(refetch_unparsable)
        self.refetch_invalid = bool(refetch_invalid)
        self.transport: Transport = transport or _http_post_json
        self.endpoint = endpoint_url(self.base_url)

        self.cached: dict[int, dict[str, Any]] = {}
        if self.cache_path is not None:
            self._reload_cache()
        self.calls: dict[int, dict[str, Any]] = {}
        self.parse_failures: list[int] = []
        self.call_failures: list[int] = []  # tolerate 模式下 HTTP/网络彻底失败、按空预测兜底的 pair
        self.live_empty: list[int] = []  # live 返回空 content（重试后仍空）的 pair 行号

    # ------------------------------------------------------------------ cache
    def _reload_cache(self) -> None:
        if self.cache_path is not None and self.cache_path.exists():
            self.cached = load_raw_records(self.cache_path)
        else:
            self.cached = {}

    def set_cache_path(self, path: str | Path | None) -> None:
        self.cache_path = Path(path) if path is not None else None
        self._reload_cache()

    # -------------------------------------------------------------- security
    def _api_key(self) -> str:
        if not self.api_key_env:
            raise RuntimeError("未配置 api_key_env：API key 只允许从环境变量读取，拒绝联网")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"环境变量 {self.api_key_env} 未设置：拒绝发起真实云端调用")
        return key

    def _redact(self, text: Any) -> str:
        msg = str(text)
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                msg = msg.replace(key, "***")
        return msg

    # --------------------------------------------------------------- request
    def build_body(self, entities: list[dict[str, Any]], before: bytes, after: bytes) -> dict[str, Any]:
        """请求体（字段/顺序对齐 eval_compare_cloud.py:59-73 call_cloud）。"""
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(entities)},
                    {"type": "image_url", "image_url": {"url": image_data_url(before)}},
                    {"type": "image_url", "image_url": {"url": image_data_url(after)}},
                ],
            }],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # 部分聚合网关前置 Cloudflare（error 1010），默认 UA 会被拦，带浏览器 UA。
            "User-Agent": DEFAULT_USER_AGENT,
        }
        headers.update(self.extra_headers)
        return headers

    def _call_live(
        self,
        entities: list[dict[str, Any]],
        before: bytes,
        after: bytes,
        pair_index: int,
        *,
        allow_live: bool,
    ) -> dict[str, Any]:
        """单次真实调用（重试/降级/空 content 兜底），返回 payload（content 可能为空串）。

        空 content 重试一次后仍空：**返回 payload 让上层按空预测兜底**（不抛错、不中断整场，
        与 eval_compare_cloud.cloud_pred 口径一致）。只有网络/HTTP 级错误重试耗尽才抛错。
        """
        if not allow_live:
            raise RuntimeError("allow_live=False：禁止真实云端调用（只允许离线缓存回放）")
        body = self.build_body(entities, before, after)
        headers = self._headers()
        failures = 0
        downgraded = False
        empty_retried = False
        while True:
            try:
                payload = self.transport(self.endpoint, body, headers, self.timeout_s)
            except HTTPPostError as exc:
                if not downgraded and exc.status in (400, 404, 422) and _looks_like_response_format_error(exc.detail):
                    # 服务端不支持 response_format → 去掉该字段立即重试
                    body = {k: v for k, v in body.items() if k != "response_format"}
                    downgraded = True
                    continue
                failures += 1
                if failures > self.retries:
                    raise RuntimeError(
                        self._redact(f"{self.model} 云端调用失败（pair {pair_index}，{failures} 次）：{exc}")
                    ) from None
                time.sleep(self.retry_sleep_s)
                continue
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if failures > self.retries:
                    raise RuntimeError(
                        self._redact(f"{self.model} 云端调用失败（pair {pair_index}，{failures} 次）：{exc}")
                    ) from None
                time.sleep(self.retry_sleep_s)
                continue

            content = _extract_content(payload)
            if not content or not content.strip():
                # deepseek/gpt-6.1-sol 等 reasoning 模型的 thinking tokens 可能吃光 max_tokens。
                # 先重试一次；仍空则按 eval_compare_cloud.cloud_pred 口径降级为空预测
                # （整对按全错计、不中断整场），返回 payload 让调用方缓存原始空响应与 usage。
                if not empty_retried:
                    empty_retried = True
                    time.sleep(self.retry_sleep_s)
                    continue
                return payload
            return payload

    def _entities_of(self, pair: Any) -> list[dict[str, Any]]:
        raw = getattr(pair, "raw", None) or {}
        entities = raw.get("entities") if isinstance(raw, Mapping) else None
        if isinstance(entities, list) and entities:
            return entities
        return [
            {"slot": e.slot, "bbox": list(e.bbox), "type_key": e.type_key, "present_after": e.present_after}
            for e in pair.entities
        ]

    # --------------------------------------------------------------- predict
    def predict(
        self,
        manifest: DatasetManifest,
        sample_indices: Iterable[int] | None = None,
        *,
        dataset_dir: str | Path | None = None,
        live: bool | None = None,
    ) -> dict[int, PairPrediction]:
        """逐 pair 预测；命中缓存复用，未命中且允许 live 才真实调用。

        ``live=None`` 用构造时的 ``allow_live``；``live=False`` 强制只读缓存
        （CLI 未加 ``--live`` 时由 registry/runner 传入，绝不联网）。
        """
        effective_live = self.allow_live if live is None else bool(live)
        if manifest is None:
            raise ValueError("ChatVLMAdapter.predict 需要 DatasetManifest")
        root = Path(dataset_dir) if dataset_dir is not None else manifest.root
        pair_map = {p.index: p for p in manifest.pairs}
        if sample_indices is None:
            indices = sorted(pair_map)
        else:
            indices = sorted({int(i) for i in sample_indices})
        unknown = [i for i in indices if i not in pair_map]
        if unknown:
            raise KeyError(f"manifest 不含 pair 行号 {unknown[:10]}{' ...' if len(unknown) > 10 else ''}")

        out: dict[int, PairPrediction] = {}
        pending_live: list[int] = []
        for i in indices:
            rec = self.cached.get(i)
            if rec is not None:
                if self.refetch_empty and not str(rec.get("raw") or "").strip():
                    # 旧缓存是空响应（reasoning 吃光 max_tokens）→ 视为未命中，重新调用
                    pending_live.append(i)
                    continue
                # 命中缓存：即使 content 为空/解析失败也按原脚本 cloud_pred 口径计为空预测
                # （不联网、不重试；记入 parse_failures 供报告注明）
                try:
                    parsed = parse_json(_strip_think(rec.get("raw") or ""))
                    if self.refetch_invalid and not _json_pred_ok(parsed):
                        # 缓存里是 prompt 回显/非法标签（思考型模型常见）→ 重新调用
                        pending_live.append(i)
                        continue
                    out[i] = to_pair_prediction(i, parsed)
                except Exception:  # noqa: BLE001
                    if self.refetch_unparsable:
                        # 旧缓存解析不出结构化结果（如推理把预算吃光）→ 重新调用
                        pending_live.append(i)
                        continue
                    self.parse_failures.append(i)
                    out[i] = PairPrediction(pair_index=i)
                self.calls[i] = {
                    "latency_s": rec.get("latency_s"),
                    "usage": rec.get("usage") if isinstance(rec.get("usage"), dict) else {},
                    "cached": True,
                }
                continue
            pending_live.append(i)

        if pending_live and not effective_live:
            raise RuntimeError(
                f"{self.model}: {len(pending_live)} 个 pair 无缓存且 allow_live=False"
                f"（--live 未开启），拒绝联网：{pending_live[:10]}"
            )

        for i in pending_live:
            pair = pair_map[i]
            before_path = root / pair.before
            after_path = root / pair.after
            for p in (before_path, after_path):
                if not p.exists():
                    raise FileNotFoundError(f"缺少图片：{p}（pair {pair.pair_id}）")
            entities = self._entities_of(pair)
            before_bytes = before_path.read_bytes()
            after_bytes = after_path.read_bytes()
            t0 = time.perf_counter()
            try:
                payload = self._call_live(entities, before_bytes, after_bytes, i,
                                          allow_live=effective_live)
            except RuntimeError as exc:
                # 网关抖动（404/5xx/超时）不应毁掉整场：tolerate 模式下按空预测兜底并记录
                if not self.tolerate_call_errors:
                    raise
                self.call_failures.append(int(i))
                out[i] = PairPrediction(pair_index=i)
                self.calls[i] = {"latency_s": None, "usage": {}, "cached": False,
                                 "error": self._redact(str(exc))[:200]}
                continue
            latency_s = time.perf_counter() - t0
            content = _extract_content(payload) or ""
            if not content.strip():
                # 与 cloud_pred 一致的兜底：空响应 → 空预测（下面 parse 会失败 → parse_failures）
                self.live_empty.append(int(i))
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            record = {
                "pair_index": int(i),
                "pair_id": pair.pair_id,
                "model": self.model,
                "raw": content,
                "latency_s": latency_s,
                "usage": usage,
            }
            if self.cache_path is not None:
                write_raw_record(self.cache_path, record)
                self.cached[int(i)] = record
            self.calls[i] = {"latency_s": latency_s, "usage": usage, "cached": False}
            try:
                out[i] = to_pair_prediction(i, parse_json(_strip_think(content)))
            except Exception:  # noqa: BLE001
                self.parse_failures.append(i)
                out[i] = PairPrediction(pair_index=i)
        return out

    # --------------------------------------------------------------- runtime
    def runtime_stats(self, sample_indices: Iterable[int] | None = None) -> RuntimeStats:
        indices = sorted(self.calls) if sample_indices is None else [int(i) for i in sample_indices]
        lat: list[float] = []
        usages: list[dict[str, Any]] = []
        for i in indices:
            call = self.calls.get(i)
            if call is None:
                continue
            value = call.get("latency_s")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lat.append(float(value))
            usage = call.get("usage")
            usages.append(usage if isinstance(usage, dict) else {})
        stats = RuntimeStats(
            tokens_prompt=int(sum(int(u.get("prompt_tokens", 0) or 0) for u in usages)) if usages else None,
            tokens_completion=int(sum(int(u.get("completion_tokens", 0) or 0) for u in usages)) if usages else None,
            cost_usd=None,
            cost_basis="cloud API pricing not publicly defined; cost left null (tokens_* are measured)",
            n_calls=len(lat) if lat else (len(usages) if usages else None),
            source="measured (live call wall clock + usage)",
        )
        if lat:
            arr = np.asarray(lat, dtype=np.float64) * 1000.0
            stats.latency_ms_mean = float(arr.mean())
            stats.latency_ms_p50 = float(np.percentile(arr, 50))
            stats.latency_ms_max = float(arr.max())
        return stats
