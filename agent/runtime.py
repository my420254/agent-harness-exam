from __future__ import annotations

"""Agent 运行时调度中枢（agent/runtime.py）。

设计职责：
1. ReAct 状态机编排：Think -> Act -> Observe -> Finish 循环。
2. 命名契约统一：全流程严格使用 session.working_memory，彻底摒弃旧版 tool_state。
3. 可观测性追踪：细粒度记录 llm_call、tool_call、tool_result 与错误流水（Trace）。
4. 自我修复与防御：
   - 捕获模型输出格式畸变（ParseError），反哺 Observation 引导模型自纠错。
   - 捕获任意工具执行崩坏异常，以纯文本观测方式回传，阻断程序闪退。
"""

import json
import time
from typing import Any

from agent.context import build_messages, build_system_prompt
from agent.llm import LLMClient, LLMError
from agent.parser import ParseError, parse_llm_output
from agent.registry import ToolNotFoundError, ToolRegistry
from agent.session import SessionStore


def _event(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """生成标准化结构的可观测性 Trace 事件条目"""
    return {
        "ts": round(time.time(), 3),
        "event": event_type,
        **(payload or {}),
    }


class AgentRuntime:
    """Agent 运行时中枢：管理 Session、调度 Registry 与执行 ReAct 推理循环"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        store: SessionStore | None = None,
        max_turns: int = 8,
        max_messages: int = 20,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store or SessionStore()
        self.max_turns = max_turns
        self.max_messages = max_messages

    def run(self, session_id: str, user_input: str) -> dict[str, Any]:
        """执行单次用户请求的完整 Agent 循环。
        
        Args:
            session_id: 隔离会话唯一标识符
            user_input: 用户本轮输入的自然语言指令
            
        Returns:
            执行结果字典，包含最终答案、轮次统计及详细 Trace 日志
        """
        # 1. 取出或新建隔离会话，追加用户输入
        session = self.store.get_or_create(session_id)
        session.append("user", user_input)

        trace: list[dict[str, Any]] = []
        answer = ""

        # 2. 启动 ReAct 单次任务多轮循环
        for turn_idx in range(1, self.max_turns + 1):
            # 【修复点 1】：全面使用 session.working_memory 渲染系统提示词
            system_prompt = build_system_prompt(
                self.registry.schema(), 
                session.working_memory
            )
            
            messages = build_messages(
                session.messages,
                system_prompt=system_prompt,
                max_messages=self.max_messages,
            )

            # -------------------------------------------------------------
            # 步骤一：调用大模型推理 (Think & Decide)
            # -------------------------------------------------------------
            try:
                reply = self.llm.chat(messages)
            except LLMError as exc:
                trace.append(_event("llm_error", {"error": str(exc), "turn": turn_idx}))
                answer = f"模型调用失败：{exc}"
                session.append("assistant", answer)
                break

            trace.append(
                _event(
                    "llm_call",
                    {
                        "turn": turn_idx,
                        "model": reply.get("model", "unknown"),
                        "latency_ms": reply.get("latency_ms", 0),
                        "usage": reply.get("usage", {}),
                    },
                )
            )

            raw_content = reply.get("content", "")

            # -------------------------------------------------------------
            # 步骤二：解析大模型输出 (Parse)
            # -------------------------------------------------------------
            try:
                parsed = parse_llm_output(raw_content)
            except ParseError as exc:
                # 记录格式解析失败并反哺提示，引导模型自纠错
                session.append("assistant", raw_content)
                error_feedback = (
                    f"[System Observation] 输出格式无法解析：{exc}。请严格按 JSON 规范格式输出。"
                )
                session.append("user", error_feedback)
                trace.append(
                    _event(
                        "parse_error",
                        {"turn": turn_idx, "error": str(exc), "raw_content": raw_content},
                    )
                )
                continue

            # 提取思维链思考内容并记入日志
            if getattr(parsed, "thinking", None):
                trace.append(
                    _event("thinking", {"turn": turn_idx, "text": parsed.thinking[:300]})
                )

            # -------------------------------------------------------------
            # 步骤三：命中终结条件 (Finish Answer)
            # -------------------------------------------------------------
            if parsed.kind == "answer":
                answer = parsed.answer or "（空回答）"
                session.append("assistant", answer)
                trace.append(_event("answer", {"turn": turn_idx, "text": answer}))
                break

            # -------------------------------------------------------------
            # 步骤四：执行工具调用分支 (Act & Observe)
            # -------------------------------------------------------------
            tool_name = parsed.tool_name or ""
            arguments = parsed.arguments or {}
            
            # 将模型的决策意图作为 assistant 发言写入历史
            session.append("assistant", raw_content)
            trace.append(
                _event("tool_call", {"turn": turn_idx, "tool": tool_name, "arguments": arguments})
            )

            # 调度工具：捕获所有异常，确保环境观察能够安全回填
            try:
                # 【修复点 2】：工具调度网关注入 session.working_memory
                result = self.registry.call(
                    name=tool_name, 
                    arguments=arguments, 
                    working_memory=session.working_memory
                )
                result_text = json.dumps(result, ensure_ascii=False, default=str)
                trace.append(
                    _event(
                        "tool_result",
                        {"turn": turn_idx, "tool": tool_name, "ok": True, "result": result},
                    )
                )
            except Exception as exc:
                result_text = f"工具执行失败：{type(exc).__name__} - {exc}"
                trace.append(
                    _event(
                        "tool_error",
                        {"turn": turn_idx, "tool": tool_name, "ok": False, "error": str(exc)},
                    )
                )

            # 将工具结果作为环境观测（Observation）回填给模型
            session.append("user", f"[工具 {tool_name} 执行结果观测]:\n{result_text}")

        else:
            # 循环用尽兜底逻辑
            answer = "已达到单次任务最大思考轮次上限，未能输出最终结论。"
            session.append("assistant", answer)
            trace.append(_event("max_turns", {"turns": self.max_turns}))

        return {
            "session_id": session_id,
            "answer": answer,
            "turn_count": session.turn_count,
            "trace": trace,
        }