from __future__ import annotations

"""三个内置工具模块（agent/tools.py）：calculator、search、todo。

职责划分：
1. 依赖倒置：从 agent.registry 导入 Tool 契约类。
2. 入参标准统一：所有工具均实现为 fn(arguments, working_memory)。
3. 安全沙箱化：calculator 基于 AST 白名单求值，彻底避免 eval 安全漏洞。
4. 状态看板绑定：todo 工具操作 working_memory 字典，跨轮对话自动保留状态。
"""

import ast
import operator
from typing import Any

from agent.registry import Tool


# =====================================================================
# 1. Calculator 计算器：基于 AST 白名单的安全四则运算
# =====================================================================

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


def _safe_eval(expression: str) -> float | int:
    """基于抽象语法树 (AST) 进行纯算术求值，隔绝任意代码注入。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"数学表达式语法错误: {expression}") from exc

    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant)
    if not isinstance(tree, allowed_nodes):
        raise ValueError(f"不支持的非法表达式: {expression}")

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left_val = _walk(node.left)
            right_val = _walk(node.right)
            if type(node.op) is ast.Div and right_val == 0:
                raise ZeroDivisionError("除数不能为零")
            return _BIN_OPS[type(node.op)](left_val, right_val)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            operand_val = _walk(node.operand)
            return _UNARY_OPS[type(node.op)](operand_val)

        raise ValueError(f"表达式包含不支持的运算或危险语法: {type(node).__name__}")

    result = _walk(tree.body)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def calculator_fn(
    arguments: dict[str, Any],
    working_memory: dict[str, Any],
) -> dict[str, Any]:
    """计算数学表达式。属于无状态工具，无需读写 working_memory。"""
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("缺少必要的 expression 参数")
    return {
        "expression": expression,
        "result": _safe_eval(expression),
    }


# =====================================================================
# 2. Search Mock 检索：模拟获取外部事实型观测数据
# =====================================================================

_SEARCH_DB: dict[str, str] = {
    "天气": "北京今天多云转晴，气温 8~15 度，空气质量良。",
    "周报": "本周完成了 Agent runtime 的物理沙盒审计与公平评测，并接入了 Milvus 向量库。",
    "default": "这是一条通用的 mock 检索结果，演示 search 工具如何返回文本。",
}


def search_fn(
    arguments: dict[str, Any],
    working_memory: dict[str, Any],
) -> dict[str, Any]:
    """Mock 信息检索。属于无状态工具，无需读写 working_memory。"""
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("缺少必要的 query 参数")

    for keyword, text in _SEARCH_DB.items():
        if keyword in query:
            return {"query": query, "results": [text]}
    return {"query": query, "results": [_SEARCH_DB["default"]]}


# =====================================================================
# 3. Todo 待办：实现多轮持久化工作记忆
# =====================================================================

def todo_fn(
    arguments: dict[str, Any],
    working_memory: dict[str, Any],
) -> dict[str, Any]:
    """管理待办事项列表。直接更新 working_memory 看板中的 todos。"""
    action = str(arguments.get("action", "")).strip().lower()

    todos_container = working_memory.setdefault("todos", [])
    if not isinstance(todos_container, list):
        todos_container = []
        working_memory["todos"] = todos_container

    todos: list[str] = todos_container

    if action == "add":
        item = str(arguments.get("item", "")).strip()
        if not item:
            raise ValueError("todo add 操作缺少 item 参数")
        todos.append(item)
        return {"action": "add", "item": item, "current_todos": list(todos)}

    if action == "list":
        return {"action": "list", "current_todos": list(todos)}

    if action == "clear":
        todos.clear()
        return {"action": "clear", "current_todos": list(todos)}

    raise ValueError(f"不支持的 todo 动作: {action!r}")


# =====================================================================
# 4. 工厂函数：打包导出 Tool 实例
# =====================================================================

def build_tools() -> list[Tool]:
    """构建内置工具实例列表，供 ToolRegistry 注册使用。"""
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