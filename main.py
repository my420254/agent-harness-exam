from __future__ import annotations

"""CLI 入口：交互式多窗口聊天，或跑一个双窗口演示。

用法：
    python main.py               # 交互式 REPL，可切换 session
    python main.py --demo        # 跑"窗口1查天气记待办 / 窗口2写周报记待办"演示
"""

import argparse
import json

from agent.llm import LLMClient
from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.tools import build_tools


def build_runtime() -> AgentRuntime:
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    return AgentRuntime(llm=LLMClient(), registry=registry)


def print_trace(trace: list[dict]) -> None:
    for item in trace:
        event = item["event"]
        if event == "llm_call":
            print(f"  · llm_call model={item.get('model')} latency={item.get('latency_ms')}ms")
        elif event in {"tool_call", "tool_result", "tool_error"}:
            print(f"  · {event} tool={item.get('tool')}")
        elif event == "answer":
            print(f"  · answer")
        elif event in {"parse_error", "llm_error", "max_turns"}:
            print(f"  · {event}")


def demo() -> None:
    runtime = build_runtime()
    print("== 窗口 1：查天气 + 记待办 ==")
    for text in ["帮我查一下北京天气，并把'带伞'记到待办里", "我之前让你记的待办是什么？"]:
        print(f"> {text}")
        result = runtime.run("user-a-window-1", text)
        print(f"< {result['answer']}\n")

    print("== 窗口 2：写周报 + 记待办（应独立于窗口 1）==")
    for text in ["帮我把'写周报'记到待办，然后告诉我今天天气", "我之前让你记的待办是什么？"]:
        print(f"> {text}")
        result = runtime.run("user-a-window-2", text)
        print(f"< {result['answer']}\n")

    print("== 切回窗口 1 继续聊（待办应仍是窗口 1 自己的）==")
    result = runtime.run("user-a-window-1", "再记一条'买牛奶'，然后列出我所有待办")
    print(f"> 再记一条'买牛奶'，然后列出我所有待办")
    print(f"< {result['answer']}\n")


def repl() -> None:
    runtime = build_runtime()
    session_id = input("session id（默认 demo）: ").strip() or "demo"
    print("输入问题回车即可；输入 /trace 查看上一次调用链；输入 /new 切窗口；/quit 退出。")
    last_trace: list[dict] = []
    while True:
        try:
            text = input(f"[{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            session_id = input("新 session id: ").strip() or "demo"
            continue
        if text == "/trace":
            print_trace(last_trace)
            continue
        result = runtime.run(session_id, text)
        last_trace = result["trace"]
        print(result["answer"])
        print_trace(result["trace"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="跑双窗口演示")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        repl()


if __name__ == "__main__":
    main()
