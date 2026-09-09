from __future__ import annotations

"""Context 上下文与记忆有效管理模块（agent/context.py）。

核心设计哲学（对应笔试 README「memory 的召回时机与放置方式」）：
1. 结构化持久记忆（Working Memory）：
   - 待办列表 (todos) 等业务状态独立维护在 Session 的 working_memory 中。
   - 放置方式：每轮 ReAct 调度前，动态编译并挂载到 System Prompt 头部。
   - 召回时机：确定性状态注入，彻底规避因多轮滑动窗口截断导致的“状态遗忘”。

2. 非结构化对话历史（Episodic Memory）：
   - 用户输入、工具执行结果、Agent 思考链以消息流序列持久化。
   - 纯对话追问、带工具的多跳追问完全依赖该序列自然递推。

3. 上下文基础压缩（Head-Tail Truncation with Safe Role Padding）：
   - 超过 max_messages 时，保留 Head（最初的用户目标）与 Tail（最近的多轮上下文）。
   - 中间旧轮次折叠为摘要占位符，规避负数切片下溢与连续角色冲突。
"""

import json
from typing import Any


def build_system_prompt(
    tool_schema: list[dict[str, Any]], 
    working_memory: dict[str, Any]
) -> str:
    """构建严格受约束的系统级提示词。
    
    Args:
        tool_schema: 工具清单 JSON Schema 定义。
        working_memory: 当前会话绑定的工作记忆状态池（如 todos 待办）。
    """
    lines: list[str] = [
        "你是一个具备工具调用能力的自主 Agent 助手，只能通过下方提供的工具获取信息或执行操作。",
        "工具清单（JSON Schema）：",
        json.dumps(tool_schema, ensure_ascii=False, indent=2),
    ]

    # 动态注入全局看板状态（Working Memory）
    # 严格判定：只有当存在待办且列表非空时才追加看板，为空时绝不出现“当前待办”字样
    todos = working_memory.get("todos") or []
    if todos:
        lines.extend([
            "",
            "当前持有的业务状态（Working Memory）：",
            "当前待办列表（务必记住，用户问起时直接据此回答）：",
            json.dumps(todos, ensure_ascii=False, indent=2),
        ])

    lines.extend([
        "",
        "每轮只能输出一个动作，且必须输出合法的单行 JSON，格式如下：",
        '调用工具：{"action":"tool","tool":"<工具名>","arguments":{...}}',
        '最终回答：{"action":"answer","answer":"面向用户的回复内容"}',
        "",
        "规则：缺少信息或需要记录时果断调用工具；信息充足时直接给出 answer；不要反问用户。",
    ])
    return "\n".join(lines)


def build_messages(
    session_messages: list[dict[str, str]],
    *,
    system_prompt: str,
    max_messages: int = 20,
) -> list[dict[str, str]]:
    """装配带有系统指令与安全压缩的历史消息上下文。

    历史过长时，保留最旧的两条（开场目标）与最近的消息，
    中间折叠为一条摘要占位，实现基础压缩。

    Args:
        session_messages: 当前会话积累的原始消息流。
        system_prompt: 编译完成的系统提示词。
        max_messages: 上下文滑动窗口容量限制（下限防御 >= 4）。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    history = list(session_messages)

    # 防御性规整：保证窗口容量下限 >= 4，避免切片计算出现负数或 0
    effective_limit = max(max_messages, 4)

    if len(history) <= effective_limit:
        return messages + history

    # 保留最初 2 条交互（锁定初始意图）
    kept_head = history[:2]
    # 为 head(2条) 和 占位折叠消息(1条) 预留空间
    tail_count = effective_limit - 3
    kept_tail = history[-tail_count:] if tail_count > 0 else []

    # 包含明确的“已压缩”关键词，且使用 user 角色防止出现连续两条 assistant 引发 API 报错
    fold_message: dict[str, str] = {
        "role": "user",
        "content": "[系统提示：较早的对话与工具执行细节已压缩，关键业务状态已固化在系统提示词中]",
    }

    return messages + [*kept_head, fold_message, *kept_tail]