# 2026 Agent Harness 技术笔试 — 要求拆解与完成方案

## 一、题目总览

### Part 1：Vibe Coding — 从零实现一个最小可用 Agent

| # | 要求 | 我的实现要点 |
|---|---|---|
| 1 | 从零完成，不依赖 langgraph/openhands/openclaw | 纯 Python 自写 `Runtime` 主循环，只用 `httpx` 调 LLM API，无任何 agent 框架 |
| 2 | 基本 Loop | `接收输入 → LLM 决策(回复/工具) → 调工具 → 据结果判断继续 loop 还是返回` |
| 3 | 至少 3 个工具 | `calculator`（真实算）、`search`（mock 返回）、`todo`（session 内持久化待办） |
| 4 | 工具注册机制 | 每个工具含 `name/description/parameters(JSON Schema)`，LLM 据 Schema 自主选 |
| 5 | LLM 输出解析 | 解析 `<thinking>`、`{"action":"tool"...}`、`{"action":"answer"...}` 三种形态 |
| 6 | session 管理 | 用户 A 窗口 1/窗口 2 各自独立 session，可随时接着聊、互不影响 |
| 7 | context 有效管理 | 最大轮次限制、记住状态、支持追问（纯对话 + 带工具）、过长基础压缩 |
| 8 | 异常处理 + trace | 工具调用/LLM 调用 try-except + 结构化 trace 日志 |
| 9 | 测试用例 | 覆盖解析器/工具/注册/session/context/完整 loop（mock LLM） |
| 10 | 真实 LLM API | OpenAI-compatible（本机 vLLM Qwen / DeepSeek），环境变量可配 |
| 11 | README | 运行方式、系统设计、memory 召回时机与放置方式说明 |
| 12 | AI Prompt 与问题解决记录 | 单独文档记录开发中用到的 prompt 和踩坑解决 |

### Part 2：架构设计题（5 模块各选 1 题）

| 模块 | 选题 |
|---|---|
| 一 Context/Performance | 题 1（首 token 5-10s → 2s 的优化方案） |
| 二 Memory | 题 2（memory 经典框架与发展趋势、头部玩家做法） |
| 三 Task | 题 2（每天 9 点根据昨天聊天做复盘总结，怎么设计） |
| 四 Tool/Session Runtime | 题 2（session busy 时新消息/异步工具完成事件怎么处理） |
| 五 Agent Runtime 架构对比 | 题 1（Claude Code 工具输出 vs OpenAI function calling） |

## 二、技术选型

- Python 3.10+，标准库 + `httpx`（唯一第三方运行依赖）
- LLM：OpenAI-compatible 接口（`LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` 环境变量）
- 测试：`pytest`
- 无 agent 框架、无 langgraph，主循环自行实现

## 三、目录结构

```
agent-harness-exam/
├── README.md                       # 运行方式、系统设计、memory 召回时机与放置方式
├── requirements.txt
├── .env.example
├── main.py                         # CLI 入口（多窗口演示）
├── agent/
│   ├── __init__.py
│   ├── llm.py                      # OpenAI-compatible 客户端（含解析）
│   ├── runtime.py                  # 核心 Agent Runtime（自实现 loop）
│   ├── registry.py                 # 工具注册机制
│   ├── tools.py                    # calculator / search / todo
│   ├── parser.py                   # LLM 输出解析（thinking/tool/answer）
│   ├── session.py                  # session 多窗口管理
│   └── context.py                  # context 管理 + 基础压缩
├── tests/
│   ├── test_parser.py
│   ├── test_tools.py
│   ├── test_registry.py
│   ├── test_session.py
│   ├── test_context.py
│   └── test_runtime.py             # mock LLM 跑完整 loop
└── docs/
    ├── AI_PROMPT_AND_SOLVING.md    # AI Prompt 与问题解决记录
    └── architecture_design.md      # 5 模块架构设计题答案
```

## 四、核心设计要点

### 1. Agent Loop（自实现，`runtime.py`）
```
while turns < max_turns:
    messages = context.build(session)          # 组装上下文
    output = llm.chat(messages)                 # 调 LLM
    parsed = parser.parse(output)               # 提取 thinking/tool/answer
    if parsed.action == "answer":               # 最终答案 → 返回用户
        return parsed.answer
    if parsed.action == "tool":                 # 工具调用 → 执行 → 把结果塞回上下文继续
        result = registry.call(parsed.tool, parsed.arguments)
        session.append_tool_result(...)
        continue
```
LLM 每轮根据「系统 prompt + 对话历史 + 工具结果」决定是继续调工具还是给最终答案。

### 2. 工具注册（`registry.py`）
- `Tool` dataclass：`name / description / parameters(JSON Schema) / fn`
- `ToolRegistry.register(tool)`、`registry.schema()` 生成给 LLM 的工具清单、`registry.call(name, args)` 统一执行 + 异常捕获。

### 3. LLM 输出解析（`parser.py`）
要求模型输出两种 JSON 动作之一（外层可带 `<thinking>`）：
- `{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}`
- `{"action":"answer","answer":"..."}`
解析逻辑：正则抓 `<thinking>...</thinking>`，再从文本里找第一个 `{...}` JSON 块，容错处理（去 markdown 围栏、截断到合法 JSON）。

### 4. Session 管理（`session.py`）
- `SessionStore`：`{session_id: Session}`，Session 内含 `messages`、`tool_state`（如 todo 列表）、`turn_count`。
- 用户 A 窗口 1（`session="a-1"`）和窗口 2（`session="a-2"`）互不相干，各自可续聊。

### 5. Context 管理（`context.py`）
- **放置方式**：系统 prompt（角色 + 工具 Schema + 当前工具态如 todo）→ 历史消息 → 本轮用户输入。
- **追问**：纯对话追问靠保留历史消息；带工具追问靠把上轮工具结果留在上下文中。
- **memory 召回时机**：`todo` 这类结构化状态在**每轮开始前**注入系统 prompt（不放消息流里，避免噪声）；对话记忆靠保留最近 N 轮消息。
- **基础压缩**：超过 `max_messages` 时，把最旧的历史压缩成一条 `[较早对话摘要]`（可用 LLM 摘要，简单实现用截断保留首尾）。

### 6. 异常处理 + trace
- 工具执行 try-except，失败返回结构化错误并让 LLM 决定下一步。
- LLM 调用超时/异常捕获，返回友好错误。
- `TraceLogger` 记录 `llm_call / tool_call / tool_result / answer` 事件，带时间戳和 token 计数。

### 7. 测试
- 解析器：thinking/tool/answer/畸形 JSON 的容错。
- 工具：calculator 正确性、search mock、todo 增删查。
- 注册：schema 生成、调用、未注册工具报错。
- session：多窗口隔离、可续聊。
- context：max turns 截断、压缩、追问。
- runtime：用 `FakeLLM`（按轮次返回预置输出）跑完整 loop 与工具调用。

## 五、真实 LLM API

`.env.example`：
```
LLM_API_BASE=https://api.deepseek.com   # 或 http://192.168.27.250:18003/v1（本机 vLLM Qwen）
LLM_API_KEY=...
LLM_MODEL=deepseek-v4-flash            # 或 Qwen3.6-27B
```

## 六、架构设计题答案

见 `docs/architecture_design.md`，5 个模块各 1 题，重点写"方案 + 权衡 + 为什么"，不堆字数。
