from __future__ import annotations

import pytest

from agent.tools import calculator_fn, search_fn, todo_fn


def test_calculator_basic():
    assert calculator_fn({}, {"expression": "(1+2)*3"})["result"] == 9
    assert calculator_fn({}, {"expression": "2**10"})["result"] == 1024
    assert calculator_fn({}, {"expression": "7/2"})["result"] == 3.5


def test_calculator_rejects_arbitrary_code():
    with pytest.raises(ValueError):
        calculator_fn({}, {"expression": "__import__('os').system('id')"})


def test_calculator_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator_fn({}, {"expression": "1/0"})


def test_search_mock_matches_keyword():
    result = search_fn({}, {"query": "北京天气"})
    assert "多云" in result["results"][0]


def test_todo_add_list_clear():
    state: dict = {}
    todo_fn(state, {"action": "add", "item": "带伞"})
    todo_fn(state, {"action": "add", "item": "买牛奶"})
    listed = todo_fn(state, {"action": "list"})
    assert listed["todos"] == ["带伞", "买牛奶"]
    todo_fn(state, {"action": "clear"})
    assert todo_fn(state, {"action": "list"})["todos"] == []
