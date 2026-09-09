from __future__ import annotations

from agent.context import build_messages, build_system_prompt


def test_todo_state_injected_into_system_prompt():
    prompt = build_system_prompt([{"name": "todo"}], {"todos": ["带伞", "买牛奶"]})
    assert "带伞" in prompt and "买牛奶" in prompt


def test_no_todo_section_when_empty():
    prompt = build_system_prompt([{"name": "todo"}], {})
    assert "当前待办" not in prompt


def test_context_within_limit_keeps_all():
    messages = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}]
    result = build_messages(messages, system_prompt="SYS", max_messages=10)
    assert result[0] == {"role": "system", "content": "SYS"}
    assert len(result) == 3


def test_context_compresses_long_history():
    messages = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    result = build_messages(messages, system_prompt="SYS", max_messages=10)
    # 1 system + 2 head + 1 压缩占位 + 8 tail = 12
    assert result[0]["role"] == "system"
    assert any("已压缩" in m["content"] for m in result)
    assert result[-1]["content"] == "m29"
