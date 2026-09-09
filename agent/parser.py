from __future__ import annotations

"""LLM 输出解析：从模型回复里提取思考过程、工具调用或最终答案。

约定模型输出（外层可先给一段 <thinking>）：

    <thinking>用户想知道天气，先查一下。</thinking>
    {"action":"tool","tool":"search","arguments":{"query":"天气"}}

或直接给最终答案：

    {"action":"answer","answer":"北京今天多云转晴，8~15 度。"}

解析器容错处理 markdown 围栏、前后噪声、以及 JSON 被截断的情况。
"""

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    thinking: str | None
    kind: Literal["tool", "answer"]
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None


class ParseError(ValueError):
    pass


def parse_llm_output(text: str) -> ParsedAction:
    raw = text or ""

    thinking: str | None = None
    match = _THINKING_RE.search(raw)
    if match:
        thinking = match.group(1).strip()

    # 去掉 markdown 围栏，方便直接取 JSON 块
    body = raw
    fenced = _FENCE_RE.search(raw)
    if fenced:
        body = fenced.group(1)

    payload = _extract_json(body)
    if payload is None:
        raise ParseError("no valid JSON found in LLM output")

    action = str(payload.get("action", "")).strip().lower()
    if action == "tool":
        tool_name = str(payload.get("tool", "")).strip()
        arguments = payload.get("arguments") or {}
        if not tool_name or not isinstance(arguments, dict):
            raise ParseError("tool action missing tool name or arguments")
        return ParsedAction(thinking=thinking, kind="tool", tool_name=tool_name, arguments=arguments)
    if action == "answer":
        answer = str(payload.get("answer", "")).strip()
        return ParsedAction(thinking=thinking, kind="answer", answer=answer)
    raise ParseError(f"unknown action type: {action!r}")


def _extract_json(text: str) -> dict[str, Any] | None:
    """从文本里抠出第一个可解析的 JSON 对象，支持截断时补 `}` 恢复。"""
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        return None

    end = stripped.rfind("}")
    base = stripped[start : end + 1] if end > start else stripped[start:]

    # 先尝试完整解析；失败则逐个补 `}`（最多 4 个），覆盖"JSON 被截断"的情况
    for extra in range(5):
        piece = base + "}" * extra
        try:
            value = json.loads(piece)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
