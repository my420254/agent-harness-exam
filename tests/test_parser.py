from __future__ import annotations

import pytest

from agent.parser import ParseError, parse_llm_output


def test_parse_tool_call_with_thinking():
    text = '<thinking>先算一下。</thinking>\n{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}'
    parsed = parse_llm_output(text)
    assert parsed.thinking == "先算一下。"
    assert parsed.kind == "tool"
    assert parsed.tool_name == "calculator"
    assert parsed.arguments == {"expression": "1+2"}


def test_parse_answer():
    parsed = parse_llm_output('{"action":"answer","answer":"北京今天多云转晴。"}')
    assert parsed.kind == "answer"
    assert parsed.answer == "北京今天多云转晴。"


def test_parse_with_markdown_fence_and_noise():
    text = '这是回复：\n```json\n{"action":"answer","answer":"好的"}\n```\n结束'
    parsed = parse_llm_output(text)
    assert parsed.kind == "answer"
    assert parsed.answer == "好的"


def test_parse_malformed_raises():
    with pytest.raises(ParseError):
        parse_llm_output("完全没有 JSON 的文本")


def test_parse_truncated_json_recovers():
    # JSON 被截断时，应能回退到最后一个完整的 } 前解析成功
    text = '{"action":"tool","tool":"todo","arguments":{"action":"add","item":"买牛奶"}'
    parsed = parse_llm_output(text)
    assert parsed.kind == "tool"
    assert parsed.tool_name == "todo"
    assert parsed.arguments["item"] == "买牛奶"
