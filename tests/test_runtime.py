from __future__ import annotations

"""ReAct 运行时中枢与集成链路测试（tests/test_runtime.py）。

测试覆盖场景：
1. 完整闭环链路（Happy Path）：Tool Call -> Tool Result 回填 -> 最终 Answer。
2. 异常降级恢复（Fault Tolerance）：调用不存在的工具抛错，错误被安全包装为 Observation，模型继续完成应答。
3. 格式自愈重试（Self-Correction）：模型首次输出非结构化脏数据触发 ParseError，系统反哺提示词引导重试。
4. 会话物理隔离（Session Isolation）：验证不同 session_id 的情景对话与 working_memory 看板互不干扰、强隔离。
"""

from typing import Any

from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_tools


class FakeLLM:
    """按预设脚本依次返回模型输出的 Mock 对象，用于实现确定性的 ReAct 链路断言。"""

    def __init__(self, scripted: list[str]) -> None:
        self.scripted = list(scripted)
        # 记录模型每次收到的上下文历史镜像，供断言检查
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """模拟单次对话调用，依次弹出预设指令，弹空则返回 fallback 兜底。"""
        self.calls.append([dict(m) for m in messages])
        content = (
            self.scripted.pop(0)
            if self.scripted
            else '{"action":"answer","answer":"fallback"}'
        )
        return {"content": content, "model": "fake", "usage": {}, "latency_ms": 0}


def _runtime(llm: FakeLLM, store: SessionStore | None = None) -> AgentRuntime:
    """辅助工厂：快速组装搭载内置工具池的 AgentRuntime 实例。"""
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    return AgentRuntime(llm=llm, registry=registry, store=store, max_turns=8, max_messages=20)


def test_full_loop_tool_then_answer() -> None:
    """验证完整的 ReAct 调用闭环：模型发起工具调用 -> 执行工具 -> 观测回填 -> 产出最终答案。"""
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}',
            '{"action":"answer","answer":"结果是 3"}',
        ]
    )
    result = _runtime(llm).run("s1", "1+2 等于几？")

    # 1. 最终回答断言
    assert result["answer"] == "结果是 3"

    # 2. Trace 追踪流水完整性断言
    events = [item["event"] for item in result["trace"]]
    assert "tool_call" in events
    assert "tool_result" in events
    assert "answer" in events

    # 3. 观测闭环断言：第二轮模型必须能看到第一轮工具计算出的结果 3
    assert any("3" in msg["content"] for msg in llm.calls[-1])


def test_tool_error_is_handled_and_continues() -> None:
    """验证当模型调用非法/不存在的工具时，系统不会闪退崩溃，而是转为错误观测后继续执行。"""
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"not_exist","arguments":{}}',
            '{"action":"answer","answer":"工具不存在"}',
        ]
    )
    result = _runtime(llm).run("s1", "随便")

    assert result["answer"] == "工具不存在"
    assert any(item["event"] == "tool_error" for item in result["trace"])


def test_parse_error_triggers_retry() -> None:
    """验证模型输出脏格式时，系统记录 parse_error 并引导模型进入下一轮自纠错。"""
    llm = FakeLLM(
        [
            "这是一段没有 JSON 的纯文本输出",
            '{"action":"answer","answer":"好了"}',
        ]
    )
    result = _runtime(llm).run("s1", "hi")

    assert result["answer"] == "好了"
    assert any(item["event"] == "parse_error" for item in result["trace"])


def test_sessions_isolated_via_runtime() -> None:
    """验证多窗口状态强隔离：window-1 写入的状态看板绝不会泄露到 window-2 中。"""
    store = SessionStore()
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"todo","arguments":{"action":"add","item":"带伞"}}',
            '{"action":"answer","answer":"已记下"}',
            '{"action":"answer","answer":"窗口2好"}',
        ]
    )
    runtime = _runtime(llm, store=store)

    # 在窗口 1 中记录待办
    runtime.run("user-a-1", "把带伞记到待办")
    # 在窗口 2 中单纯对话
    runtime.run("user-a-2", "你好")

    session_1 = store.get("user-a-1")
    session_2 = store.get("user-a-2")

    assert session_1 is not None
    assert session_2 is not None

    # 【关键修正】：统一断言 working_memory，杜绝 tool_state
    assert session_1.working_memory["todos"] == ["带伞"]
    assert session_2.working_memory == {}