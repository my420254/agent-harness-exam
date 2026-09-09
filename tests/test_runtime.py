from __future__ import annotations

from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_tools


class FakeLLM:
    """按脚本依次返回预置输出，便于确定性测试完整 loop。"""

    def __init__(self, scripted: list[str]) -> None:
        self.scripted = list(scripted)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> dict:
        self.calls.append([dict(m) for m in messages])
        content = self.scripted.pop(0) if self.scripted else '{"action":"answer","answer":"fallback"}'
        return {"content": content, "model": "fake", "usage": {}, "latency_ms": 0}


def _runtime(llm: FakeLLM, store: SessionStore | None = None) -> AgentRuntime:
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    return AgentRuntime(llm=llm, registry=registry, store=store, max_turns=8, max_messages=20)


def test_full_loop_tool_then_answer():
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}',
            '{"action":"answer","answer":"结果是 3"}',
        ]
    )
    result = _runtime(llm).run("s1", "1+2 等于几？")
    assert result["answer"] == "结果是 3"
    events = [item["event"] for item in result["trace"]]
    assert "tool_call" in events and "tool_result" in events and "answer" in events
    # 第二轮 LLM 应能看到工具结果
    assert any("3" in msg["content"] for msg in llm.calls[-1])


def test_tool_error_is_handled_and_continues():
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"not_exist","arguments":{}}',
            '{"action":"answer","answer":"工具不存在"}',
        ]
    )
    result = _runtime(llm).run("s1", "随便")
    assert result["answer"] == "工具不存在"
    assert any(item["event"] == "tool_error" for item in result["trace"])


def test_parse_error_triggers_retry():
    llm = FakeLLM(
        [
            "这是一段没有 JSON 的输出",
            '{"action":"answer","answer":"好了"}',
        ]
    )
    result = _runtime(llm).run("s1", "hi")
    assert result["answer"] == "好了"
    assert any(item["event"] == "parse_error" for item in result["trace"])


def test_sessions_isolated_via_runtime():
    store = SessionStore()
    llm = FakeLLM(
        [
            '{"action":"tool","tool":"todo","arguments":{"action":"add","item":"带伞"}}',
            '{"action":"answer","answer":"已记下"}',
            '{"action":"answer","answer":"窗口2好"}',
        ]
    )
    runtime = _runtime(llm, store=store)
    runtime.run("user-a-1", "把带伞记到待办")
    runtime.run("user-a-2", "你好")
    assert store.get("user-a-1").tool_state["todos"] == ["带伞"]
    assert store.get("user-a-2").tool_state == {}
