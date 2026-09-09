from __future__ import annotations

"""全量测试运行套件（run_all_tests.py）。

功能：
1. 单元与集成测试（Pytest 自动化回归）：覆盖解析器、工具沙箱、会话隔离与运行时 Mock。
2. 端到端实测（Real API E2E）：连接真实大模型，自动执行 6 阶段跨窗口核心用例并进行强隔离断言。

运行方式：
  python run_all_tests.py
"""

import sys
import time
from pathlib import Path

# 确保项目根路径挂载
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from agent.llm import LLMClient
from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_tools

# 6 个跨会话端到端核心验收用例
E2E_CASES = [
    # 阶段 1：窗口 1 复合指令与状态持久化
    ("user-a-window-1", "帮我查一下北京天气，并把'带伞'记到待办里"),
    ("user-a-window-1", "我之前让你记的待办是什么？"),
    # 阶段 2：窗口 2 会话物理隔离验证
    ("user-a-window-2", "帮我把'写周报'记到待办，然后告诉我今天天气"),
    ("user-a-window-2", "我之前让你记的待办是什么？"),
    # 阶段 3：切回窗口 1 持续累加与工具计算
    ("user-a-window-1", "再记一条'买牛奶'，然后列出我所有待办"),
    ("user-a-window-1", "算上买运动鞋预算 500*0.8，一共需要准备多少钱？"),
]


def run_unit_tests() -> bool:
    """第一部分：调用 Pytest 运行 tests/ 目录下的所有单元测试"""
    print("\n" + "=" * 70)
    print("📋 [Part 1/2] 开始运行单元测试与集成测试 (Pytest)")
    print("=" * 70)

    # 屏蔽可能冲突的 anyio 插件，开启详细输出
    pytest_args = ["tests", "-v", "-p", "no:anyio"]
    ret_code = pytest.main(pytest_args)

    if ret_code == 0:
        print("\n✅ 所有单元测试通过！")
        return True
    else:
        print(f"\n❌ 单元测试存在失败用例 (Exit code: {ret_code})")
        return False


def run_e2e_real_api_tests() -> bool:
    """第二部分：调用真实 LLM API 执行跨会话隔离与工具复合测试"""
    print("\n" + "=" * 70)
    print("🚀 [Part 2/2] 开始执行端到端真机验收测试 (6 个跨窗口用例)")
    print("=" * 70)

    # 1. 挂载工具池
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)

    # 2. 初始化持久化与运行时
    store = SessionStore()
    runtime = AgentRuntime(
        llm=LLMClient(),
        registry=registry,
        store=store,
        max_turns=8,
        max_messages=20,
    )

    # 3. 逐条执行用例
    for idx, (sid, query) in enumerate(E2E_CASES, 1):
        print(f"\n[{idx}/6] 会话视窗: {sid}")
        print(f"  🧑 User  > {query}")

        start_t = time.time()
        res = runtime.run(sid, query)
        elapsed = round(time.time() - start_t, 2)

        print(f"  🤖 Agent > {res.get('answer', '')} (耗时: {elapsed}s)")

        # 打印工具调用链路与关键追踪
        for event in res.get("trace", []):
            evt = event.get("event")
            if evt == "tool_call":
                print(f"     🛠️ [Tool Call] {event.get('tool')}: {event.get('arguments')}")
            elif evt == "tool_result":
                summary = str(event.get("result"))[:60]
                print(f"     📥 [Tool Result] {event.get('tool')}: {summary}")
            elif evt in ("parse_error", "tool_error"):
                print(f"     ⚠️ [{evt}] {event.get('error')}")

    # 4. 强隔离状态核验与断言
    print("\n" + "-" * 70)
    print("🔍 检验 Session 间状态强隔离 (Working Memory 断言):")

    s1 = store.get("user-a-window-1")
    s2 = store.get("user-a-window-2")

    s1_todos = s1.working_memory.get("todos", []) if s1 else []
    s2_todos = s2.working_memory.get("todos", []) if s2 else []

    print(f"  · 窗口 1 待办看板: {s1_todos}")
    print(f"  · 窗口 2 待办看板: {s2_todos}")

    try:
        # 1. 验证窗口 1 的累加持久化状态
        assert any("带伞" in item for item in s1_todos), "窗口 1 缺少待办: '带伞'"
        assert any("买牛奶" in item for item in s1_todos), "窗口 1 缺少待办: '买牛奶'"

        # 2. 核心隔离断言：窗口 2 绝不能被窗口 1 污染
        assert any("写周报" in item for item in s2_todos), "窗口 2 缺少待办: '写周报'"
        assert not any("带伞" in item or "买牛奶" in item for item in s2_todos), (
            f"❌ 窗口 2 发生数据穿透！受到了窗口 1 的污染: {s2_todos}"
        )

        print("\n✅ 端到端状态隔离断言全部通过！窗口 1 与窗口 2 绝对物理隔离。")
        return True
    except AssertionError as err:
        print(f"\n❌ 端到端断言失败: {err}")
        return False


def main() -> None:
    print("=" * 70)
    print("🧪 Agent Harness 全链路自动化评测启动")
    print("=" * 70)

    unit_ok = run_unit_tests()
    e2e_ok = run_e2e_real_api_tests()

    print("\n" + "=" * 70)
    print("📊 最终评测结论汇总:")
    print(f"  1. 单元测试 (Pytest)      : {'PASS ✅' if unit_ok else 'FAIL ❌'}")
    print(f"  2. 端到端真机测试 (E2E)    : {'PASS ✅' if e2e_ok else 'FAIL ❌'}")
    print("=" * 70)

    if not (unit_ok and e2e_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()