from __future__ import annotations

"""工具注册与执行调度模块（agent/registry.py）。

设计职责：
1. 统一工具契约：封装 Tool 不可变实体（名称、描述、JSON Schema、执行函数句柄）。
2. 参数规范：所有工具函数统一约定签名：fn(arguments, working_memory)。
3. 调度网关：负责工具检索、参数注入以及将未捕获异常统一转换为标准错误。
"""

from dataclasses import dataclass
from typing import Any, Callable

# 统一工具函数签名：接收 (arguments, working_memory)，返回任意可序列化结果
ToolFn = Callable[[dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    """工具元数据与调用句柄不可变实体"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 格式
    fn: ToolFn


class ToolNotFoundError(KeyError):
    """请求了未注册的工具异常"""
    pass


class ToolExecutionError(RuntimeError):
    """工具内部执行崩溃异常（供 Runtime 捕获并转为 Observation 反馈给模型）"""
    pass


class ToolRegistry:
    """Agent 工具注册表与调度网关"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个新工具，重名时抛出异常"""
        if tool.name in self._tools:
            raise ValueError(f"工具已存在，禁止重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """根据名称获取工具实体"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"未找到对应工具: {name!r}，可用工具列表: {self.names()}")
        return tool

    def names(self) -> list[str]:
        """返回所有已注册工具名称列表"""
        return list(self._tools.keys())

    def schema(self) -> list[dict[str, Any]]:
        """导出符合 OpenAI 规范的工具清单元数据"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        working_memory: dict[str, Any],
    ) -> Any:
        """统一调用网关：执行工具并注入 working_memory。
        
        约定：arguments 在前，working_memory 在后。
        """
        tool = self.get(name)
        try:
            return tool.fn(arguments, working_memory)
        except Exception as exc:
            raise ToolExecutionError(f"工具 '{name}' 执行发生错误: {exc}") from exc