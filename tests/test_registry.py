from __future__ import annotations

"""LLM 结构化输出解析模块（agent/parser.py）。

职责与设计理念：
1. 【思维链隔离】：从大模型输出中稳健剥离 <thinking>...</thinking> 标签，将其留存为观察日志，
   避免思维链污染后续的结构化数据提取。
2. 【多级容错清洗】：清洗包括 ```json、```JSON、``` 等多种 Markdown 围栏标记与前后杂质文本。
3. 【残缺/截断自愈机制】：在长生成或 Token 限流场景下，大模型常常输出末尾缺失闭合符号的残缺 JSON
   （如缺少末尾的 '}' 或 ']'）。解析器通过基于栈的语法闭合算法，自动修复残缺结构并成功提取。
4. 【强类型契约】：解析成功返回统一的 ParsedOutput 实体，解析失败显式抛出自定义 ParseError。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


class ParseError(Exception):
    """当模型输出完全不符合协议规范或无法修复时抛出的异常"""
    pass


@dataclass
class ParsedOutput:
    """结构化解析结果契约对象"""
    # 动作分类：'tool' 表示调用工具，'answer' 表示直接向用户输出最终答案
    kind: Literal["tool", "answer"]
    # 提取出的思维链推理内容（若有）
    thinking: str | None = None
    # 目标工具名称（kind == 'tool' 时必填）
    tool_name: str | None = None
    # 工具入参字典
    arguments: dict[str, Any] = field(default_factory=dict)
    # 最终呈现给人类的答案（kind == 'answer' 时必填）
    answer: str | None = None


# 正则预编译：匹配并分离 <thinking>...</thinking> 块（支持跨行与大小写）
_THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)

# 正则预编译：提取 Markdown 代码块中的内容（支持 ```json, ```JSON, ```）
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?```",
    re.DOTALL
)


def _repair_truncated_json(raw_json: str) -> str:
    """基于符号栈平衡算法，自动为被截断的 JSON 补齐缺失的闭合引号与括号。
    
    例如：
    - 输入：'{"action":"tool","tool":"todo","arguments":{"action":"add","item":"买牛奶"}'
    - 状态分析：遇到外层 '{'、内层 '{'，但结尾只闭合了一个 '}'
    - 补齐输出：在末尾追加 '}'，使其成为合法的 JSON 字符串
    """
    stack: list[str] = []
    in_string = False
    escape = False

    for char in raw_json:
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char in "{[":
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    # 1. 如果在字符串中间被物理截断，首先闭合双引号
    repaired = raw_json
    if in_string:
        repaired += '"'

    # 2. 依次按相反顺序补齐未闭合的花括号和方括号
    while stack:
        opener = stack.pop()
        if opener == "{":
            repaired += "}"
        elif opener == "[":
            repaired += "]"

    return repaired


def _extract_json_candidate(text: str) -> str:
    """从掺杂自然语言文本的内容中精准提取最可能的 JSON 字符串片段。"""
    # 优先尝试从 Markdown 代码块标记（```json ... ```）中剥离
    code_match = _CODE_BLOCK_PATTERN.search(text)
    if code_match:
        return code_match.group(1).strip()

    # 若未找到代码块标记，定位最外层的起始括号 '{'
    start_idx = text.find("{")
    if start_idx == -1:
        raise ParseError(f"未在模型输出中找到合法的 JSON 起始标记 '{{'：{text!r}")

    # 截取从第一个 '{' 开始的剩余全部内容
    candidate = text[start_idx:].strip()

    # 如果尾部有冗余文本且包含结束花括号，裁剪到最后一个 '}'
    end_idx = candidate.rfind("}")
    if end_idx != -1:
        # 保留从首个 '{' 到最后一个 '}' 的闭合区间
        candidate = candidate[: end_idx + 1]

    return candidate


def parse_llm_output(raw_text: str) -> ParsedOutput:
    """主入口函数：将大模型生成的原始非结构化文本解析为结构化 ParsedOutput。
    
    Args:
        raw_text: 大模型输出的纯文本内容。
        
    Returns:
        解析后的 ParsedOutput 实例。
        
    Raises:
        ParseError: 当输入为空、无合法结构或字段缺失时抛出。
    """
    # 基础边界防御：处理空字符串及包含 \xa0 等各类 Unicode 空白符的内容
    if not raw_text or not raw_text.strip():
        raise ParseError("模型返回内容为空或仅包含空白字符")

    # 1. 提取思考过程思维链（<thinking> 标签）
    thinking: str | None = None
    cleaned_text = raw_text

    thinking_match = _THINKING_PATTERN.search(raw_text)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # 将思维链内容从主文本中扣除，避免干扰后续 JSON 匹配
        cleaned_text = _THINKING_PATTERN.sub("", raw_text).strip()

    # 2. 提取潜在的 JSON 文本子串
    candidate = _extract_json_candidate(cleaned_text)

    # 3. 尝试 JSON 反序列化（含截断容错修复）
    data: dict[str, Any]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # 初次尝试失败，启动截断自动闭合补齐算法
        try:
            repaired_candidate = _repair_truncated_json(candidate)
            data = json.loads(repaired_candidate)
        except json.JSONDecodeError as exc:
            raise ParseError(f"无法解析或修复该 JSON 片段: {candidate}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"JSON 顶层必须为字典对象，实际为: {type(data).__name__}")

    # 4. 解析业务意图与动作分支
    action = data.get("action")
    if not action:
        raise ParseError(f"缺少必要的 'action' 动作声明字段: {data}")

    if action == "answer":
        answer = data.get("answer")
        if answer is None:
            raise ParseError("声明了 action='answer' 但缺少 'answer' 字段")
        return ParsedOutput(
            kind="answer",
            thinking=thinking,
            answer=str(answer),
        )

    if action == "tool":
        tool_name = data.get("tool")
        if not tool_name:
            raise ParseError("声明了 action='tool' 但未指定 'tool' 名称")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ParseError(f"'arguments' 必须为字典结构，实际为: {type(arguments).__name__}")

        return ParsedOutput(
            kind="tool",
            thinking=thinking,
            tool_name=str(tool_name),
            arguments=arguments,
        )

    raise ParseError(f"未知的动作类型 action={action!r}，仅支持 'tool' 或 'answer'")