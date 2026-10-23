"""OpenWSV 模型适配器：文本 VLM（OpenAI 兼容）与 StarLife-1-WSV 结构化服务。

两类适配器共同约定：**适配器内部捕获所有异常，不向调用方抛出**，统一返回

    {"text": str | None, "latency_s": float | None, "error": str | None}

（结构化适配器返回 ``{"response": dict | None, "latency_s", "error"}``）。
错误不是"错题"：runner 会把它计入未覆盖（coverage），不按 0 分处理。

* :class:`HttpVlmAdapter` — 任意 OpenAI 兼容 ``POST {base_url}/chat/completions``
  多模态端点；图像以 base64 ``data:`` URL 内联；``temperature=0`` 默认。
* :class:`WsvApiAdapter` — StarLife-1-WSV-Lite ``POST {api_url}/predict``，
  body ``{before, after, detections}``，返回结构化响应（slots/pairs/events）。

密钥只存在于内存：构造参数或环境变量读取，绝不出现在返回值/日志/结果 JSON 中。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["HttpVlmAdapter", "WsvApiAdapter", "api_key_from_env"]

_RETRYABLE_HTTP = (408, 409, 425, 429, 500, 502, 503, 504)

#: 本机服务（例如 SSH 隧道）不应走系统代理
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _image_url(path_or_url: str | Path) -> str:
    """本地文件 -> base64 data URL；http(s) URL 原样返回。"""
    value = str(path_or_url)
    if value.startswith(("http://", "https://")):
        return value
    suffix = Path(value).suffix.lower().lstrip(".") or "jpeg"
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    return f"data:image/{mime};base64,{_read_b64(value)}"


def _truncate(text: str, limit: int = 300) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


class HttpVlmAdapter:
    """OpenAI 兼容多模态 chat adapter。

    Parameters
    ----------
    base_url:
        例如 ``https://api.example.com/v1``；会请求 ``{base_url}/chat/completions``。
    api_key:
        明文密钥（仅内存）。为空时不发 ``Authorization`` 头（本地 vLLM 等）。
    model:
        服务端模型名。
    temperature / max_tokens:
        默认 ``0.0`` / ``1024``，与 framework 冻结口径一致。
    """

    kind = "vlm"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        *,
        timeout_s: float = 180.0,
        retries: int = 2,
        user_agent: str = "openwsv/1",
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self._api_key = api_key or ""
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.retries = max(1, int(retries))
        self.user_agent = user_agent

    @property
    def adapter_id(self) -> str:
        return f"openai-chat:{self.model}"

    # -- internal -----------------------------------------------------------
    def _body(self, prompt: str, images: Sequence[str | Path]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": _image_url(image)}})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _extract_text(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices") or []
        message = (choices[0] or {}).get("message") or {} if choices else {}
        text = message.get("content", "") or ""
        if isinstance(text, list):  # 少数网关返回 content parts
            text = "".join(
                str(part.get("text", "")) for part in text if isinstance(part, Mapping)
            )
        return str(text)

    # -- public -------------------------------------------------------------
    def complete(self, prompt: str, images: Sequence[str | Path] = (), **_: Any) -> dict[str, Any]:
        """发一次 chat 请求，返回 ``{"text", "latency_s", "error"}``；异常不抛。"""
        started = time.perf_counter()
        last_error = "request failed"
        for attempt in range(self.retries):
            try:
                payload = self._post(self._body(prompt, images))
                text = self._extract_text(payload)
                latency = time.perf_counter() - started
                if not text.strip():
                    # 空回复按未覆盖处理（reasoning 模型可能耗尽 max_tokens）
                    return {"text": None, "latency_s": latency,
                            "error": "empty response (max_tokens/reasoning truncation)"}
                return {"text": text, "latency_s": latency, "error": None}
            except urllib.error.HTTPError as exc:
                detail = _truncate(exc.read().decode("utf-8", errors="replace"))
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in _RETRYABLE_HTTP or attempt + 1 >= self.retries:
                    break
                delay = min(30.0, 2.0 ** attempt)
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 —— 超时/连接/解码失败都不抛
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= self.retries:
                    break
                time.sleep(min(30.0, 2.0 ** attempt))
        return {"text": None, "latency_s": time.perf_counter() - started, "error": last_error}


class WsvApiAdapter:
    """StarLife-1-WSV-Lite 结构化服务 adapter（``POST /predict``）。

    Parameters
    ----------
    api_url:
        服务根 URL，例如 ``http://127.0.0.1:8890``（SSH 隧道）或内网地址。
    api_key:
        ``X-API-Key`` 头；为空时不发。
    """

    kind = "structured"

    def __init__(
        self,
        api_url: str,
        api_key: str | None,
        *,
        timeout_s: float = 180.0,
        retries: int = 1,
    ) -> None:
        self.api_url = str(api_url).rstrip("/")
        self._api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.retries = max(1, int(retries))

    @property
    def adapter_id(self) -> str:
        return f"wsv-api:{self.api_url}"

    @staticmethod
    def _payload_value(value: str | bytes | Path) -> str:
        """路径/bytes -> base64；已是 base64 的字符串原样使用。"""
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, Path):
            return _read_b64(value)
        text = str(value)
        try:
            path = Path(text)
            if path.exists():
                return _read_b64(path)
        except OSError:  # 超长/非法路径：按已编码字符串处理
            pass
        return text

    def predict(
        self,
        before: str | bytes | Path,
        after: str | bytes | Path,
        detections: Mapping[str, Any],
    ) -> dict[str, Any]:
        """发一次结构化推理，返回 ``{"response", "latency_s", "error"}``；异常不抛。"""
        body = {
            "before": self._payload_value(before),
            "after": self._payload_value(after),
            "detections": dict(detections),
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        request = urllib.request.Request(
            self.api_url + "/predict",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        last_error = "request failed"
        for attempt in range(self.retries):
            try:
                with _DIRECT_OPENER.open(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return {"response": payload, "latency_s": time.perf_counter() - started,
                        "error": None}
            except urllib.error.HTTPError as exc:
                detail = _truncate(exc.read().decode("utf-8", errors="replace"))
                last_error = f"HTTP {exc.code}: {detail}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self.retries:
                time.sleep(min(10.0, 2.0 ** attempt))
        return {"response": None, "latency_s": time.perf_counter() - started, "error": last_error}


def api_key_from_env(*names: str) -> str | None:
    """按顺序从环境变量读密钥（只读、不落盘）；都没有返回 ``None``。"""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None