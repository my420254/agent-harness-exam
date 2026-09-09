# tests/test_tools.py
from __future__ import annotations

"""工具沙箱与状态看板单元测试套件（tests/test_tools.py）。

测试覆盖维度：
1. Calculator 安全沙箱求值：
   - 基础四则运算、多层嵌套括号与幂运算正确性。
   - 抽象语法树（AST）白名单校验：对任意系统调用、代码注入行为执行前置拦截。
   - 数学异常捕获：除以零边界场景的 ZeroDivisionError 校验。

2. Search Mock 检索能力：
   - 模拟外部事实型检索库命中，校验返回结构契约一致性。

3. Todo 状态看板持久化（有状态工具）：
   - 基于会话级独立 working_memory 字典的读写一致性。
   - 严格遵循底层契约（action、item、current_todos）执行连续追加、列表读取与重置清空校验。
"""

from typing import Any
import pytest

from agent.tools import calculator_fn, search_fn, todo_fn


# =====================================================================
# 1. Calculator 计算器工具单元测试（基于 AST 安全沙箱）
# =====================================================================

def test_calculator_basic() -> None:
    """验证安全计算器的基础代数计算能力。
    
    测试覆盖：
    - 加法运算求值。
    - 组合四则运算与嵌套括号优先级计算。
    - 幂运算求值及整型/浮点型规整结果。
    """
    # 验证加法求值
    assert calculator_fn({"expression": "1 + 2"}, {})["result"] == 3
    # 验证复合优先级运算：(3 * 7) - 1 = 20
    assert calculator_fn({"expression": "(3 * 7) - 1"}, {})["result"] == 20
    # 验证幂运算：2 ** 4 = 16
    assert calculator_fn({"expression": "2 ** 4"}, {})["result"] == 16


def test_calculator_rejects_arbitrary_code() -> None:
    """验证 AST 沙箱对非纯数学表达式及任意代码注入攻击的安全拦截能力。
    
    预期行为：
    - 当传入包含 __import__、os.system 等危险节点时，
      AST 解析器应在语法遍历阶段识别并抛出 ValueError，杜绝安全逃逸。
    """
    with pytest.raises(ValueError):
        calculator_fn({"expression": "__import__('os').system('ls')"}, {})


def test_calculator_division_by_zero() -> None:
    """验证计算器面对数学异常边界时的防御性机制。
    
    预期行为：
    - 当除数为 0 时，准确触发 ZeroDivisionError 异常，
      以便上层 Agent 运行时捕获该 tool_error 并反馈给模型自愈。
    """
    with pytest.raises(ZeroDivisionError):
        calculator_fn({"expression": "10 / 0"}, {})


# =====================================================================
# 2. Search Mock 信息检索工具单元测试
# =====================================================================

def test_search_mock_matches_keyword() -> None:
    """验证 Mock 检索工具的关键词匹配及事实数据契约输出。
    
    预期行为：
    - 输入包含'北京天气'的检索 query 时，命中天气分支并返回对应事实描述。
    - 返回字典中包含结构化的 query 与 results 列表。
    """
    res = search_fn({"query": "北京天气"}, {})
    assert "results" in res
    assert any("天气" in r or "气温" in r for r in res["results"])


# =====================================================================
# 3. Todo 待办事项状态看板单元测试（Working Memory 读写契约）
# =====================================================================

def test_todo_add_list_clear() -> None:
    """验证待办工具的有状态持久化特性及状态机流转闭环。
    
    契约断言维度：
    1. 动态挂载：直接依赖传入的独立 working_memory 字典，杜绝跨会话全局变量污染。
    2. 动作流转：依次执行 add -> list -> clear，验证状态单调变更。
    3. 契约校验：严格基于 tools.py 实际返回的 current_todos 规范校验状态流。
    """
    # 初始化独立的会话工作记忆池
    working_memory: dict[str, Any] = {}

    # 1. 动作阶段一：连续追加待办项 (action='add')
    res_add1 = todo_fn({"action": "add", "item": "带伞"}, working_memory)
    assert res_add1["current_todos"] == ["带伞"], "首次追加待办状态不一致"

    res_add2 = todo_fn({"action": "add", "item": "买牛奶"}, working_memory)
    assert res_add2["current_todos"] == ["带伞", "买牛奶"], "二次追加待办状态未成功累积"

    # 2. 动作阶段二：列出待办列表 (action='list')
    listed = todo_fn({"action": "list"}, working_memory)
    assert listed["current_todos"] == ["带伞", "买牛奶"], "查询待办与实际内存看板状态不符"

    # 3. 动作阶段三：清空待办列表 (action='clear')
    cleared = todo_fn({"action": "clear"}, working_memory)
    assert cleared["current_todos"] == [], "清空动作后返回值 current_todos 应为空列表"
    assert working_memory["todos"] == [], "working_memory 底层看板数组未被彻底清空"