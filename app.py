from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# 确保在任意工作目录下启动均能准确定位 agent 模块
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import spaces

from agent.llm import LLMClient
from agent.registry import ToolRegistry
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_tools


def build_runtime() -> AgentRuntime:
    registry = ToolRegistry()
    for tool in build_tools():
        registry.register(tool)
    return AgentRuntime(llm=LLMClient(), registry=registry, store=SessionStore())


runtime = build_runtime()

MODEL_NAME = getattr(runtime.llm, "model", "deepseek-v4-flash")
REGISTERED_TOOLS = runtime.registry.names()

# 前端会话历史缓存池（与后端 Runtime 的 Session 形成强隔离映射）
SESSION_HISTORIES: dict[str, list[dict[str, str]]] = {}


def extract_text_content(content: Any) -> str:
    """提取 Gradio 传来的复杂嵌套内容，消除 [{'text': '...', 'type': 'text'}]"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def format_trace(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "暂无调用链数据"

    lines: list[str] = []
    for idx, item in enumerate(trace, 1):
        event = item.get("event", "unknown")
        if event == "llm_call":
            lines.append(
                f"[{idx}] 🧠 LLM 推理 | 模型: {item.get('model', MODEL_NAME)} | 耗时: {item.get('latency_ms')}ms"
            )
        elif event == "thinking":
            thought_snippet = str(item.get("text", "")).replace("\n", " ")[:100]
            lines.append(f"[{idx}] 💭 思考过程 | {thought_snippet}...")
        elif event == "tool_call":
            args_str = json.dumps(item.get("arguments", {}), ensure_ascii=False)
            lines.append(f"[{idx}] 🛠️ 触发工具 | {item.get('tool')}({args_str})")
        elif event == "tool_result":
            lines.append(f"[{idx}] 📥 工具返回 | {item.get('tool')} -> {item.get('result')}")
        elif event == "tool_error":
            lines.append(f"[{idx}] ❌ 工具报错 | {item.get('tool')}: {item.get('error')}")
        elif event == "answer":
            lines.append(f"[{idx}] 💬 生成最终回复")
        elif event in {"parse_error", "llm_error", "max_turns"}:
            lines.append(f"[{idx}] ⚠️ 循环边界事件: {event}")
        else:
            lines.append(f"[{idx}] · {event}")
    return "\n".join(lines)


@spaces.GPU
def run_agent_turn(user_input: str, session_id: str):
    sid = session_id.strip() or "user-a-window-1"

    # 获取或初始化属于该窗口独立的前端消息历史
    if sid not in SESSION_HISTORIES:
        SESSION_HISTORIES[sid] = []
    current_history = SESSION_HISTORIES[sid]

    if not user_input.strip():
        return sid, "", current_history, "请输入有效内容"

    try:
        result = runtime.run(sid, user_input)
        answer = result.get("answer", "")
        trace = result.get("trace", [])
    except Exception as e:
        answer = f"运行时执行异常: {str(e)}"
        trace = [{"event": "system_error", "error": str(e)}]

    # 存入纯文本规范字典
    clean_user = extract_text_content(user_input)
    clean_answer = extract_text_content(answer)
    current_history.append({"role": "user", "content": clean_user})
    current_history.append({"role": "assistant", "content": clean_answer})

    return sid, "", current_history, format_trace(trace)


def on_session_change(sid: str):
    """当用户切换 session_id 输入框时，自动渲染该 Session 的独立聊天流"""
    clean_sid = sid.strip() or "user-a-window-1"
    history = SESSION_HISTORIES.get(clean_sid, [])
    return history


with gr.Blocks(title="Agent Harness Console") as demo:
    gr.Markdown(
        f"""
        # 🤖 自研 Agent Harness 控制台
        
        **【系统元数据】**
        * **驱动基座模型**：`{MODEL_NAME}`
        * **已挂载工具池**：`{", ".join(REGISTERED_TOOLS)}`
        * **状态机说明**：前端视窗与后端 Runtime 的 `session_id` 保持强隔离映射，切换窗口自动隔离对话气泡与待办上下文。
        """
    )

    with gr.Accordion("🎯 核心验收测试集（点击按钮一键切窗口并运行）", open=True):
        gr.Markdown("**阶段 1：窗口 1 复合指令与状态持久化**")
        with gr.Row():
            btn_t1 = gr.Button("用例 1-1：窗口 1 查天气并记待办 '带伞'", size="sm")
            btn_t2 = gr.Button("用例 1-2：窗口 1 追问刚才记了什么", size="sm")

        gr.Markdown("**阶段 2：窗口 2 会话物理隔离验证（核心断言）**")
        with gr.Row():
            btn_t3 = gr.Button("用例 2-1：窗口 2 记待办 '写周报' 并查天气", size="sm")
            btn_t4 = gr.Button("用例 2-2：窗口 2 追问待办（断言：不显示窗口 1 的待办）", size="sm")

        gr.Markdown("**阶段 3：切回窗口 1 持续累加与工具计算**")
        with gr.Row():
            btn_t5 = gr.Button("用例 3-1：切回窗口 1 追加待办 '买牛奶' 并列出所有", size="sm")
            btn_t6 = gr.Button("用例 3-2：工具与纯推理复合任务（预算 500*0.8 计算）", size="sm")

    # 异常自愈与对抗测试面板
    with gr.Accordion("⚠️ 异常自愈与对抗测试集（测试 AST 拦截、格式容错与参数纠偏）", open=False):
        gr.Markdown("**点击下方按钮，使用独立会话触发对应异常，并在右侧 Trace 查看自愈过程：**")
        with gr.Row():
            btn_err1 = gr.Button("对抗 1：算非数字 ('苹果加香蕉')", size="sm", variant="secondary")
            btn_err2 = gr.Button("对抗 2：错误动作 (诱导用 delete 记待办)", size="sm", variant="secondary")
        with gr.Row():
            btn_err3 = gr.Button("对抗 3：复杂多工具连续调用", size="sm", variant="secondary")
            btn_err4 = gr.Button("对抗 4：不让用工具 & 禁用 JSON", size="sm", variant="secondary")

    with gr.Row():
        with gr.Column(scale=4):
            session_id_box = gr.Textbox(
                label="当前 Session ID (会话环境隔离键)",
                value="user-a-window-1",
                placeholder="修改此 ID 即可切换不同窗口",
            )
        with gr.Column(scale=8):
            gr.Markdown(
                "> **隔离保障**：切换到 `user-a-window-2` 时，页面对话区仅展示窗口 2 的记录，绝不出现窗口 1 的气泡。"
            )

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="Agent 对话交互区 (已绑定当前 Session)",
                height=480,
                value=[],
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    label="手动发送自定义指令",
                    placeholder="输入指令回车，或直接点击上方预设用例快捷验证...",
                    scale=9,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Column(scale=5):
            trace_box = gr.Textbox(
                label="调用链 Trace / 执行流水 (可观测性日志)",
                interactive=False,
                lines=22,
                placeholder="此处实时回显 llm_call 耗时、tool_call 传参及 tool_result 返回...",
            )

    # 普通手动输入
    submit_btn.click(
        fn=run_agent_turn,
        inputs=[msg_box, session_id_box],
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    msg_box.submit(
        fn=run_agent_turn,
        inputs=[msg_box, session_id_box],
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )

    # 手动输入框修改 Session ID 触发聊天记录动态切换
    session_id_box.change(
        fn=on_session_change,
        inputs=[session_id_box],
        outputs=[chatbot],
    )

    # 快捷按钮行为（严格按窗口分流执行）
    btn_t1.click(
        fn=lambda: run_agent_turn("帮我查一下北京天气，并把'带伞'记到待办里", "user-a-window-1"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_t2.click(
        fn=lambda: run_agent_turn("我之前让你记的待办是什么？", "user-a-window-1"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_t3.click(
        fn=lambda: run_agent_turn("帮我把'写周报'记到待办，然后告诉我今天天气", "user-a-window-2"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_t4.click(
        fn=lambda: run_agent_turn("我之前让你记的待办是什么？", "user-a-window-2"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_t5.click(
        fn=lambda: run_agent_turn("再记一条'买牛奶'，然后列出我所有待办", "user-a-window-1"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_t6.click(
        fn=lambda: run_agent_turn("算上买运动鞋预算 500*0.8，一共需要准备多少钱？", "user-a-window-1"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )

    # 异常自愈对抗按钮（自动分配带时间戳的独立会话，防止污染正常窗口）
    btn_err1.click(
        fn=lambda: run_agent_turn("帮我算一下'苹果加香蕉'等于多少？", f"err-calc-{int(time.time())}"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_err2.click(
        fn=lambda: run_agent_turn("把'写代码'记到待办，用 delete 这个动作", f"err-action-{int(time.time())}"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_err3.click(
        fn=lambda: run_agent_turn("先算 3*7，再查北京天气，然后把结果和天气都记到待办", f"case-multi-{int(time.time())}"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )
    btn_err4.click(
        fn=lambda: run_agent_turn("直接回答 1+1 等于几，不要调用任何工具，也不要用 JSON 格式", f"err-nojson-{int(time.time())}"),
        outputs=[session_id_box, msg_box, chatbot, trace_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())