from __future__ import annotations

"""三个内置工具：calculator（真实计算）、search（mock 检索）、todo（session 内待办）。

每个工具的 `fn(state, arguments)` 都接收 session 状态字典 `state`，便于 todo
这类需要跨轮持久化的工具把数据写进 session，而不是依赖全局变量。
"""

import ast
import operator
from typing import Any

from agent.registry import Tool

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    """只允许数字、四则运算、括号和幂，杜绝 eval 任意代码。"""
    tree = ast.parse(expression, mode="eval")
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant)
    if not isinstance(tree, allowed_nodes):
        raise ValueError(f"unsupported expression: {expression}")

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _walk(node.left), _walk(node.right)
            if type(node.op) is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_walk(node.operand))
        raise ValueError(f"unsupported expression: {expression}")

    result = _walk(tree.body)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def calculator_fn(state: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("expression is required")
    return {"expression": expression, "result": _safe_eval(expression)}


_SEARCH_DB: dict[str, str] = {
    "天气": "北京今天多云转晴，气温 8~15 度，空气质量良。",
    "周报": "本周完成了 Agent runtime 的物理沙盒审计与公平评测，并接入了 Milvus 向量库。",
    "default": "这是一条 mock 搜索结果，演示 search 工具如何返回文本。",
}


def search_fn(state: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    for keyword, text in _SEARCH_DB.items():
        if keyword in query:
            return {"query": query, "results": [text]}
    return {"query": query, "results": [_SEARCH_DB["default"]]}


def todo_fn(state: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    action = str(arguments.get("action", "")).strip().lower()
    todos: list[str] = state.setdefault("todos", [])
    if action == "add":
        item = str(arguments.get("item", "")).strip()
        if not item:
            raise ValueError("item is required for todo add")
        todos.append(item)
        return {"action": "add", "item": item, "todos": list(todos)}
    if action == "list":
        return {"action": "list", "todos": list(todos)}
    if action == "clear":
        todos.clear()
        return {"action": "clear", "todos": list(todos)}
    raise ValueError(f"unsupported todo action: {action}")


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="calculator",
            description="计算数学表达式，支持四则运算、括号和幂。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 (1+2)*3"},
                },
                "required": ["expression"],
            },
            fn=calculator_fn,
        ),
        Tool(
            name="search",
            description="检索信息（当前为 mock 实现，返回与关键词相关的文本）。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的关键词或问题"},
                },
                "required": ["query"],
            },
            fn=search_fn,
        ),
        Tool(
            name="todo",
            description="管理用户待办：add 添加、list 列出、clear 清空。",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "clear"]},
                    "item": {"type": "string", "description": "待办内容（add 时必填）"},
                },
                "required": ["action"],
            },
            fn=todo_fn,
        ),
    ]
