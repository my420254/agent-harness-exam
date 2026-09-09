from __future__ import annotations

"""OpenAI-compatible LLM 客户端（真实 LLM API 调用层）。

设计职责：
1. 零第三方 Agent 框架与官方 SDK 依赖，直接通过 HTTP 协议对接标准 API。
2. 环境变量自适应加载：兼顾自定义变量与行业通用标准（OPENAI_API_KEY / OPENAI_BASE_URL）。
3. 支持长连接复用（Connection Pooling），显著降低 ReAct 循环内的多次握手时延。
4. 捕获 Token 消耗、精准耗时（latency_ms）与思考内容（reasoning_content）。
"""

import os
import time
from pathlib import Path
from typing import Any

import httpx


def _load_dotenv() -> None:
    """零依赖加载项目根目录下的 .env 配置文件。
    
    原则：若系统中已显式设置环境变量，则以系统环境优先，避免覆盖。
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 跳过空行和注释
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 模块导入时自动执行环境发现
_load_dotenv()


class LLMError(RuntimeError):
    """LLM 统一异常基类（网络异常、状态码异常、解析异常均收敛于此）"""
    pass


class LLMClient:
    """支持 OpenAI 兼容协议的高性能 LLM 客户端"""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        # 兼容双重命名规则：自定义命名优先，回退到官方标准命名
        raw_base = (
            api_base
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.api_base = raw_base.rstrip("/")
        
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        
        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or os.getenv("MODEL_NAME")
            or "deepseek-v4-flash"
        )
        
        self.timeout = timeout
        self.max_tokens = max_tokens
        # Agent 任务建议锁定 temperature=0.0 以获得最稳定的 JSON/结构化输出
        self.temperature = temperature

        # 使用长连接 Client 复用 TCP 连接，避免 ReAct 循环反复三次握手
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """向 LLM 发送对话上下文并同步等待响应。
        
        Args:
            messages: 符合 OpenAI 格式的上下文列表 [{"role": "user", "content": "..."}]
            
        Returns:
            dict 包含:
                - content: 模型的主输出文本
                - reasoning: 模型的思考过程（支持 DeepSeek 等 reasoning_content）
                - model: 实际调用的模型名称
                - usage: Token 消耗统计字典
                - latency_ms: 毫秒级网络与推理总耗时
        """
        if not self.api_key:
            raise LLMError("LLM API Key 未配置（请检查 LLM_API_KEY 或 OPENAI_API_KEY）")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.api_base}/chat/completions"
        started = time.perf_counter()

        try:
            response = self._client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM 请求超时 ({self.timeout}s): {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络传输失败: {exc}") from exc

        # 校验 HTTP 状态码
        if response.status_code != 200:
            raise LLMError(
                f"LLM 接口返回错误 HTTP {response.status_code}: {response.text[:300]}"
            )

        # 校验 JSON 载荷有效性
        try:
            body = response.json()
        except Exception as exc:
            raise LLMError(f"LLM 返回内容非合法 JSON: {response.text[:200]}") from exc

        # 提取核心数据
        try:
            choice = body["choices"][0]["message"]
            content = choice.get("content") or ""
            # 兼容 DeepSeek-R1 / 特殊推理模型的显式思考字段
            reasoning = choice.get("reasoning_content") or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"LLM 响应载荷格式不符合 OpenAI 规范: {body}") from exc

        # 计算耗时并组装标准回包
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "content": str(content).strip(),
            "reasoning": str(reasoning).strip(),
            "model": self.model,
            "usage": body.get("usage", {}),
            "latency_ms": latency_ms,
        }

    def close(self) -> None:
        """关闭底层的 HTTP 资源连接池"""
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()