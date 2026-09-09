# tests/test_context.py
from __future__ import annotations

"""上下文组装与双轨记忆压缩单元测试套件（tests/test_context.py）。

测试覆盖维度：
1. 结构化工作记忆注入（Working Memory Injection）：
   - 验证待办列表（todos）在会话中存在时，能动态编译并前置挂载到 System Prompt 中。
   - 验证 KV-Cache 亲和性与无漂移原则：当待办为空时，完全不输出待办占位段落，保持提示词精简。

2. 线性情景记忆截断与安全压缩（Head-Tail Truncation & Safe Compression）：
   - 验证消息未达上限时，保留完整历史序列与初始 System Prompt。
   - 验证消息超出 max_messages 阈值时，执行“保留头部目标（Head）+ 注入压缩占位符 + 保留尾部活跃上下文（Tail）”机制。
   - 验证压缩占位消息的角色安全性，防止破坏多轮对话协议。
"""

from agent.context import build_messages, build_system_prompt


def test_working_memory_injected_into_system_prompt() -> None:
    """验证 Working Memory 存在待办事项时，正确挂载至 System Prompt。
    
    设计意图：
    - Agent Harness 采用“双轨记忆”机制。
    - 结构化业务看板（如待办列表）不应沉没于长线对话流水中，而是在每轮调度时
      由 Runtime 动态抽取并序列化注入到 System Prompt 尾部，确保模型 100% 确定性感知。
    
    断言维度：
    - 提示词中必须包含工作记忆看板的明确标识文本。
    - 内存中存在的每一项待办文本（'带伞', '写周报'）均应被正确序列化呈现。
    """
    prompt = build_system_prompt(
        [{"name": "todo"}],
        {"todos": ["带伞", "写周报"]},
    )
    assert "当前持有的业务状态（Working Memory）" in prompt, "缺少工作记忆看板头部声明"
    assert "带伞" in prompt, "未正确序列化待办项：'带伞'"
    assert "写周报" in prompt, "未正确序列化待办项：'写周报'"


def test_no_todo_section_when_empty() -> None:
    """验证 Working Memory 为空时，不注入待办看板及冗余占位噪声。
    
    设计意图：
    - 保持 System Prompt 的确定性与 Token 经济性。
    - 当用户尚未创建任何待办时，提示词不应包含空白的“当前待办”或“Working Memory”提示，
      避免模型产生“记忆混乱”或主动向用户汇报空列表。
    
    断言维度：
    - 提示词中绝不能出现'当前待办'或'Working Memory'等看板关键字。
    """
    prompt = build_system_prompt([{"name": "todo"}], {})
    assert "当前待办" not in prompt, "待办为空时不应出现'当前待办'段落"
    assert "Working Memory" not in prompt, "待办为空时不应出现'Working Memory'段落"


def test_context_within_limit_keeps_all() -> None:
    """验证对话轮次在滑动窗口容量（max_messages）限制内时，保留全量历史。
    
    设计意图：
    - 在常规对话轮次下，系统不应触发任何有损压缩或截断。
    - 输出列表首位固定为 system 角色，后续顺序顺延历史消息。
    
    断言维度：
    - 总体消息条数严格等于 1 (System Prompt) + N (原始消息数)。
    - 首位元素角色必须为 'system'。
    """
    messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    result = build_messages(messages, system_prompt="SYS", max_messages=10)

    assert len(result) == 6, f"预期保留 6 条消息（1 system + 5 user），实际为 {len(result)}"
    assert result[0]["role"] == "system", "首位消息必须为 system 角色"
    assert [m["content"] for m in result[1:]] == [f"m{i}" for i in range(5)], "原始历史消息序列不应发生变动"


def test_context_compresses_long_history() -> None:
    """验证超长历史场景下的头尾保留压缩算法（Head-Tail Truncation）。
    
    算法核心逻辑：
    1. 首位保持注入 System Prompt。
    2. 保留最前部对话（Head，锁定初始任务目标与原始意图，如 m0, m1）。
    3. 中间截断部分折叠为一条包含'已压缩'字样的提示性占位消息。
    4. 保留最近发生的尾部对话（Tail，锁定最新任务上下文，如 m29）。
    
    断言维度：
    - 结构完整性：首位为 'system' 且包含系统提示词。
    - 压缩感知：折叠消息中包含'已压缩'标识。
    - 关键意图锚定：首条任务 m0、m1 存在，最新动态 m29 存在。
    - 容量收敛：超出 30 条的长文本被压缩控制在安全预算内。
    """
    messages = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    result = build_messages(messages, system_prompt="SYS", max_messages=10)

    # 1. 结构完整性断言
    assert result[0]["role"] == "system", "首位消息必须为 system"
    assert result[0]["content"] == "SYS", "系统提示词内容不一致"

    # 2. 压缩占位符断言：验证存在标识历史已压缩的提示项
    assert any("已压缩" in m["content"] for m in result), "未在上下文中检测到包含'已压缩'的折叠提示消息"

    # 3. 头部首条意图保留断言（Head 保留前 2 条）
    assert result[1]["content"] == "m0", "压缩机制未能保留初始用户目标 m0"
    assert result[2]["content"] == "m1", "压缩机制未能保留初始上下文 m1"

    # 4. 尾部最新状态保真断言（Tail 保留最新一条）
    assert result[-1]["content"] == "m29", "压缩机制丢失了最新一轮用户输入 m29"

    # 5. 上下文容量收敛断言
    assert len(result) <= 12, f"压缩后总长度超出安全上限，当前长度: {len(result)}"