from __future__ import annotations

"""LLM 结构化输出解析模块单元测试（tests/test_parser.py）。

测试覆盖场景：
1. 思维链隔离提取：验证 <thinking> 标签剥离与正文 JSON 正确解析。
2. 纯文本回答分支：验证 kind == 'answer' 决策路径。
3. Markdown 围栏清洗：验证包含 ```json 代码块及前后噪音文本时的健壮提取。
4. 格式畸变防护：验证无合法 JSON 时安全抛出预期的 ParseError。
5. 截断容错修复：验证模型输出因 Token 上限末尾缺失括号时的自动修复能力。
"""

import pytest

from agent.parser import ParseError, parse_llm_output


def test_parse_tool_call_with_thinking() -> None:
    """测试带有思考过程标签的工具调用。
    
    验证点：
    - <thinking> 内部内容被独立提取至 parsed.thinking。
    - 外部的 JSON 正确解析为 tool 类型，并包含对应名称与参数。
    """
    text = (
        "<thinking>先算一下。</thinking>\n"
        '{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}'
    )
    parsed = parse_llm_output(text)
    
    assert parsed.thinking == "先算一下。"
    assert parsed.kind == "tool"
    assert parsed.tool_name == "calculator"
    assert parsed.arguments == {"expression": "1+2"}


def test_parse_answer() -> None:
    """测试模型直接输出最终答复的场景。
    
    验证点：
    - action 为 answer 时，正确映射至 kind == 'answer'。
    - answer 内容被完整还原，无需发起后续工具调用。
    """
    text = '{"action":"answer","answer":"北京今天多云转晴。"}'
    parsed = parse_llm_output(text)
    
    assert parsed.kind == "answer"
    assert parsed.answer == "北京今天多云转晴。"


def test_parse_with_markdown_fence_and_noise() -> None:
    """测试包含 Markdown 代码块标记及前后干扰文字的清洗能力。
    
    验证点：
    - 正则/提取器能忽略前缀引导词（如'这是回复：'）与后缀噪音。
    - 能自动剥离 ```json 和 ``` 包裹标记。
    """
    text = '这是回复：\n```json\n{"action":"answer","answer":"好的"}\n```\n结束'
    parsed = parse_llm_output(text)
    
    assert parsed.kind == "answer"
    assert parsed.answer == "好的"


def test_parse_malformed_raises() -> None:
    """测试遇到非结构化非法文本时的异常抛出。
    
    验证点：
    - 当模型输出完全不包含任何合法 JSON 结构时，必须显式抛出 ParseError，
      由运行时捕获并触发自修复反哺机制。
    """
    with pytest.raises(ParseError):
        parse_llm_output("完全没有 JSON 的文本")


def test_parse_truncated_json_recovers() -> None:
    """测试对 Token 截断导致的残缺 JSON 的容错恢复能力。
    
    说明：
    - 原始文本结尾缺少最外层闭合花括号 '}'。
    - 验证 parser 是否具备括号自动补齐或截断修复能力，确保关键动作与参数不丢失。
    """
    # 结尾缺少最外层的闭合花括号 '}'
    text = '{"action":"tool","tool":"todo","arguments":{"action":"add","item":"买牛奶"}'
    parsed = parse_llm_output(text)
    
    assert parsed.kind == "tool"
    assert parsed.tool_name == "todo"
    # 全量参数字典断言，避免字段遗漏
    assert parsed.arguments == {"action": "add", "item": "买牛奶"}