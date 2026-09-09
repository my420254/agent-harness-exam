from __future__ import annotations

"""工具注册机制。

每个工具由 `name`（唯一标识）、`description`（LLM 判断用途）、
`parameters`（JSON Schema，LLM 生成参数）、`fn`（真实执行函数）构成。
Registry 负责登记、生成给 LLM 的工具清单、以及统一调用（带异常捕获）。
"""

from dataclasses import dataclass
from typing import Any, Callable

ToolFn = Callable[[dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema（object 根）
    fn: ToolFn


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schema(self) -> list[dict[str, Any]]:
        """生成给 LLM 的工具清单（含名称、描述、参数 Schema）。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any], state: dict[str, Any]) -> Any:
        """统一调用：把 session 状态与参数传给工具，异常向上抛给 runtime 处理。"""
        tool = self.get(name)
        return tool.fn(state, arguments)
