from __future__ import annotations

import sys
from pathlib import Path

# 确保在任意工作目录下执行 python main.py 均能稳定加载 agent 模块
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import LLMClient
from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_tools


def main() -> None:
    # 1. 组装工具注册中心
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)

    # 2. 初始化持久化仓库与运行时
    store = SessionStore()
    runtime = AgentRuntime(
        llm=LLMClient(),
        registry=registry,
        store=store,
        max_turns=8,
        max_messages=20,
    )

    session_id = "cli-session"
    tools_list = ", ".join(registry.names())

    print("=" * 60)
    print(f"🤖 Agent CLI 控制台已就绪 (Session: {session_id})")
    print(f"🛠️ 已挂载可用工具: [{tools_list}]")
    print("💡 在终端直接输入内容与 Agent 对话，输入 'exit' 或 'quit' 退出。")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("\n会话结束，已安全退出。")
                break

            result = runtime.run(session_id, user_input)
            print(f"\nAgent > {result['answer']}")

            # 终端简明打印工具调用与关键事件流
            for event in result.get("trace", []):
                evt = event.get("event")
                if evt == "tool_call":
                    print(f"   🛠️ [调用工具] {event.get('tool')}({event.get('arguments')})")
                elif evt == "tool_result":
                    res_str = str(event.get("result"))[:80]
                    print(f"   📥 [工具返回] {event.get('tool')} -> {res_str}")
                elif evt == "tool_error":
                    print(f"   ❌ [工具异常] {event.get('tool')}: {event.get('error')}")
                elif evt == "parse_error":
                    print(f"   ⚠️ [格式自愈] {event.get('error')}")

        except (KeyboardInterrupt, EOFError):
            print("\n检测到退出中断信号，已终止。")
            break


if __name__ == "__main__":
    main()