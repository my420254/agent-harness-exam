from __future__ import annotations

"""真实 LLM 错误处理演示：用 deepseek-v4-flash 跑几个"故意刁难"的任务，
展示 Agent 在模型输出异常时如何接住并恢复。

运行：python demo_errors.py
"""

from agent.llm import LLMClient
from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.tools import build_tools

# 这些 case 是实测会触发错误的（解析失败、工具参数错误、模型不守 JSON 约定）
HARD_CASES: list[tuple[str, str]] = [
    ("case1-算非数字", "帮我算一下'苹果加香蕉'等于多少？"),
    ("case2-错误动作", "把'写代码'记到待办，用 delete 这个动作"),
    ("case3-复杂多工具", "先算 3*7，再查北京天气，然后把结果和天气都记到待办"),
    ("case4-不让用工具", "直接回答 1+1 等于几，不要调用任何工具，也不要用 JSON 格式"),
]


def main() -> None:
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    runtime = AgentRuntime(llm=LLMClient(), registry=registry)

    for name, question in HARD_CASES:
        print(f"\n{'=' * 60}\n{name}: {question}\n{'=' * 60}")
        result = runtime.run(name, question)
        print(f"最终回答: {result['answer']}")
        print("执行链:")
        for event in result["trace"]:
            if event["event"] == "llm_call":
                print(f"  · llm_call  latency={event.get('latency_ms')}ms")
            elif event["event"] == "parse_error":
                print(f"  · ⚠️ parse_error: {event.get('error')}")
            elif event["event"] == "tool_error":
                print(f"  · ⚠️ tool_error: {event.get('tool')} -> {event.get('error')}")
            elif event["event"] == "tool_call":
                print(f"  · tool_call: {event.get('tool')} {event.get('arguments')}")
            elif event["event"] == "tool_result":
                print(f"  · tool_result: {event.get('tool')} -> {str(event.get('result'))[:60]}")
            elif event["event"] == "answer":
                print(f"  · answer")
        errors = [e for e in result["trace"] if e["event"] in ("parse_error", "tool_error")]
        print(f"→ 本轮触发的错误数: {len(errors)}（全部被 runtime 接住并继续/恢复）")


if __name__ == "__main__":
    main()
