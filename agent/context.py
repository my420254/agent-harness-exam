from __future__ import annotations

"""context 有效管理。

关键设计（对应题目「memory 的召回时机与放置方式」）：

1. **结构化状态（todo 等）放在系统 prompt，不放在消息流里**：
   每轮开始前把 session.tool_state 里的待办注入系统提示，模型每轮都能"记住"
   待办，又不会污染对话消息。

2. **对话记忆靠保留最近 N 轮消息**：纯对话追问、带工具的追问，都因为历史
   （含工具结果）留在消息里而自然成立。

3. **过长压缩**：超过 max_messages 时，把最旧的中间轮次折叠成一条摘要占位，
   保留系统提示 + 最近的消息，避免上下文爆掉、又不丢关键状态。
"""

import json
from typing import Any


def build_system_prompt(tool_schema: list[dict[str, Any]], tool_state: dict[str, Any]) -> str:
    lines = [
        "你是一个助手 Agent，只能通过下面的工具获取信息或执行操作。",
        "工具清单（JSON）：",
        json.dumps(tool_schema, ensure_ascii=False),
    ]
    todos = tool_state.get("todos") or []
    if todos:
        lines.append("当前待办（务必记住，用户问起时直接据此回答）：")
        lines.append(json.dumps(todos, ensure_ascii=False))
    lines += [
        "",
        "每轮只输出一个动作，格式如下（可先给 <thinking>...</thinking>，再给 JSON）：",
        '调用工具：{"action":"tool","tool":"<名称>","arguments":{...}}',
        '最终回答：{"action":"answer","answer":"..."}',
        "规则：需要信息就调用工具；信息足够就直接 answer；不要再向用户反问，尽量用工具或已有上下文解决。",
    ]
    return "\n".join(lines)


def build_messages(
    session_messages: list[dict[str, str]],
    *,
    system_prompt: str,
    max_messages: int,
) -> list[dict[str, str]]:
    """组装 [system] + (压缩后的) 历史消息。

    历史过长时，保留最旧的两条（开场上下文）与最近 max_messages 条，
    中间折叠为一条摘要占位，实现「基础压缩」。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    history = list(session_messages)

    if len(history) <= max_messages:
        return messages + history

    kept_head = history[:2]
    kept_tail = history[-(max_messages - 2) :]
    compressed: list[dict[str, str]] = [
        *kept_head,
        {
            "role": "assistant",
            "content": "[较早的对话与工具结果已压缩省略，关键结论保留在后续消息中]",
        },
        *kept_tail,
    ]
    return messages + compressed
