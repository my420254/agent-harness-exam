from __future__ import annotations

"""核心 Agent Runtime：自实现的主循环（不依赖任何 agent 框架）。

循环逻辑：
    接收用户输入
      -> 组装上下文（系统 prompt + 工具清单 + 待办 + 历史）
      -> LLM 决策：回复 or 调工具
      -> 若调工具：执行 -> 把结果塞回上下文 -> 回到 LLM 决策
      -> 若回复：返回给用户，结束

异常与 trace 都在这层统一处理。
"""

import json
import time
from typing import Any

from agent.context import build_messages, build_system_prompt
from agent.llm import LLMClient, LLMError
from agent.parser import ParseError, parse_llm_output
from agent.registry import ToolRegistry, ToolNotFoundError
from agent.session import SessionStore


class AgentRuntime:
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
        session = self.store.get_or_create(session_id)
        session.append("user", user_input)
        trace: list[dict[str, Any]] = []
        answer = ""

        for _ in range(self.max_turns):
            system_prompt = build_system_prompt(self.registry.schema(), session.tool_state)
            messages = build_messages(
                session.messages,
                system_prompt=system_prompt,
                max_messages=self.max_messages,
            )

            try:
                reply = self.llm.chat(messages)
            except LLMError as exc:
                trace.append(_event("llm_error", {"error": str(exc)}))
                answer = f"模型调用失败：{exc}"
                break

            trace.append(
                _event(
                    "llm_call",
                    {
                        "model": reply.get("model"),
                        "latency_ms": reply.get("latency_ms"),
                        "usage": reply.get("usage", {}),
                    },
                )
            )

            try:
                parsed = parse_llm_output(reply["content"])
            except ParseError as exc:
                # 解析失败：把提示塞回上下文，让模型重试一次
                session.append("assistant", reply["content"])
                session.append("user", f"输出格式错误：{exc}。请重新按约定 JSON 输出。")
                trace.append(_event("parse_error", {"error": str(exc)}))
                continue

            if parsed.thinking:
                trace.append(_event("thinking", {"text": parsed.thinking[:200]}))

            if parsed.kind == "answer":
                answer = parsed.answer or "（空回答）"
                session.append("assistant", answer)
                trace.append(_event("answer", {"text": answer}))
                break

            # 工具调用
            tool_name = parsed.tool_name or ""
            arguments = parsed.arguments or {}
            session.append("assistant", reply["content"])
            try:
                result = self.registry.call(tool_name, arguments, session.tool_state)
                result_text = json.dumps(result, ensure_ascii=False, default=str)
                trace.append(_event("tool_call", {"tool": tool_name, "arguments": arguments}))
                trace.append(_event("tool_result", {"tool": tool_name, "ok": True, "result": result}))
            except (ToolNotFoundError, ValueError, ZeroDivisionError) as exc:
                result_text = f"工具调用失败：{exc}"
                trace.append(_event("tool_error", {"tool": tool_name, "error": str(exc)}))
            session.append("user", f"工具 {tool_name} 的执行结果：\n{result_text}")
        else:
            # 达到最大轮次仍未 answer：给兜底
            answer = "已达最大轮次仍未得到最终答案，请把问题拆小或补充信息。"
            session.append("assistant", answer)
            trace.append(_event("max_turns", {"turns": self.max_turns}))

        return {
            "session_id": session_id,
            "answer": answer,
            "turn_count": session.turn_count,
            "trace": trace,
        }


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"ts": round(time.time(), 3), "event": event_type, **payload}
