from __future__ import annotations

"""OpenAI-compatible LLM 客户端（真实 LLM API）。

用环境变量配置，支持本机 vLLM Qwen 或 DeepSeek：
    LLM_API_BASE  /  LLM_API_KEY  /  LLM_MODEL
"""

import os
import time
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.api_base = (api_base or os.getenv("LLM_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.api_base or not self.api_key:
            raise LLMError("LLM_API_BASE / LLM_API_KEY 未配置")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:  # 网络/超时统一包装
            raise LLMError(f"llm request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"llm http {response.status_code}: {response.text[:200]}")

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected llm response: {body}") from exc

        return {
            "content": str(content or "").strip(),
            "model": self.model,
            "usage": body.get("usage", {}),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
