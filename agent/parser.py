from __future__ import annotations

"""LLM 输出解析器：从模型回复中提取思考过程、工具调用或最终答案。

核心职责：
1. 思考链捕获：兼容 <think>、<thinking>、<thought> 各种开源与闭源模型标签。
2. 污染隔离：提取思考后立即从文本流中物理剔除，防止思考区内的代码/括号干扰 JSON 定位。
3. 格式自愈：兼容 Markdown 围栏包裹、前后闲聊噪声、以及输出截断时缺失的 '}'。
4. 容错转换：支持 arguments 字段被模型双重序列化为字符串的情形。
"""

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

# 兼容主流思考标签：<think> (DeepSeek/Qwen), <thinking> (Claude/标准), <thought>
_THINKING_RE = re.compile(
    r"<(?:think|thinking|thought)>(.*?)</(?:think|thinking|thought)>",
    re.IGNORECASE | re.DOTALL,
)

# 优先捕获被标注为 json 的代码围栏，若无则捕获通用代码围栏
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    """解析后的结构化动作对象"""
    thinking: str | None
    kind: Literal["tool", "answer"]
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None


class ParseError(ValueError):
    """解析协议不匹配异常（将反馈给 Runtime 触发 Self-Correction）"""
    pass


def parse_llm_output(text: str) -> ParsedAction:
    """解析 LLM 原始文本输出为标准结构化动作。"""
    raw = (text or "").strip()

    # 1. 提取思考链（如果有）
    thinking: str | None = None
    think_match = _THINKING_RE.search(raw)
    if think_match:
        thinking = think_match.group(1).strip()
        # 【关键修复】：从主文本中抹除思考块，防止思考区内的括号干扰后续 JSON 定位
        raw = _THINKING_RE.sub("", raw).strip()

    # 2. 剥离 Markdown 代码围栏
    fence_match = _FENCE_RE.search(raw)
    body = fence_match.group(1).strip() if fence_match else raw

    # 3. 提取有效 JSON 字典
    payload = _extract_json(body)
    if payload is None:
        # 二次兜底：若去掉围栏提取失败，尝试对整段剩余文本全量搜寻
        payload = _extract_json(raw)

    if payload is None:
        raise ParseError(
            "未在输出中找到有效的 JSON 动作对象，请务必输出符合协议规范的单行 JSON。"
        )

    # 4. 判定动作类型（兼容部分模型的近义词别名）
    raw_action = str(payload.get("action", "")).strip().lower()

    # 分支 A：调用工具
    if raw_action in ("tool", "call", "tool_call"):
        tool_name = str(payload.get("tool") or payload.get("tool_name") or "").strip()
        raw_args = payload.get("arguments", {})

        # 【关键修复】：防范模型输出 arguments: "{...}" 字符串
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except Exception:
                arguments = {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {}

        if not tool_name:
            raise ParseError("工具调用指令缺失目标工具名称 (tool 字段为空)")

        return ParsedAction(
            thinking=thinking,
            kind="tool",
            tool_name=tool_name,
            arguments=arguments,
        )

    # 分支 B：最终输出回答
    if raw_action in ("answer", "final_answer", "response"):
        answer = str(payload.get("answer") or payload.get("response") or "").strip()
        return ParsedAction(
            thinking=thinking,
            kind="answer",
            answer=answer,
        )

    raise ParseError(
        f"无法识别的 action 类型: {raw_action!r}，必须为 'tool' 或 'answer'"
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """从包含杂质文本中定位并截取合法的 JSON 对象，具备尾部截断补全自愈能力。"""
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        return None

    end = stripped.rfind("}")
    # 截取首尾大括号之间的候选子串
    base = stripped[start : end + 1] if end > start else stripped[start:]

    # 尝试直接解析；若因截断失败，尝试补充闭合大括号（最多追加 4 层闭合）
    for extra in range(5):
        candidate = base + ("}" * extra)
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            continue

    return None