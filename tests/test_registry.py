from __future__ import annotations

import pytest

from agent.registry import ToolNotFoundError, ToolRegistry
from agent.tools import build_tools


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    return registry


def test_registry_schema_has_name_description_parameters():
    registry = _registry()
    schema = registry.schema()
    names = {tool["name"] for tool in schema}
    assert names == {"calculator", "search", "todo"}
    for tool in schema:
        assert tool["description"]
        assert tool["parameters"]["type"] == "object"


def test_registry_call():
    registry = _registry()
    result = registry.call("calculator", {"expression": "3*4"}, state={})
    assert result["result"] == 12


def test_registry_unknown_tool():
    registry = _registry()
    with pytest.raises(ToolNotFoundError):
        registry.call("not_exist", {}, state={})


def test_registry_duplicate_rejected():
    registry = _registry()
    with pytest.raises(ValueError):
        registry.register(build_tools()[0])
