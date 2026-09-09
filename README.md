# Minimal Agent Harness（2026 Agent 技术笔试 · Vibe Coding 题）

从零实现的最小可用 Agent：不依赖 langgraph / openhands / openclaw 等任何 agent 框架，
核心 Runtime 主循环自行实现。核心运行只依赖 `httpx` 调真实 LLM API；可选 Gradio UI 依赖 `gradio`。

> 🌐 **在线 Demo（Hugging Face Space）**：https://huggingface.co/spaces/my420254/agent-harness-exam
> （点开即可在网页里直接体验多窗口隔离与异常自愈，无需本地环境）

## 一、运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置真实 LLM API（OpenAI-compatible）
cp .env.example .env
# 编辑 .env：填 LLM_API_BASE / LLM_API_KEY / LLM_MODEL
# 也支持本机 vLLM Qwen：LLM_API_BASE=http://192.168.27.250:18003/v1  LLM_MODEL=Qwen3.6-27B
# 或 DeepSeek：LLM_API_BASE=https://api.deepseek.com  LLM_MODEL=deepseek-v4-flash

# 3. CLI 交互式聊天（多窗口）
python main.py

# 4. Gradio Web 控制台（含核心用例 + 异常自愈对抗用例 + Trace 面板）
python app.py

# 5. 全量测试：单元测试 + 真实 API 端到端（6 个跨窗口用例 + 强隔离断言）
python run_all_tests.py

# 6. 只跑单元测试
pytest tests/ -q
```

## 二、系统设计

### 1. Agent Loop（`agent/runtime.py`，自行实现）

```
接收用户输入
  -> 组装上下文（system prompt + 工具清单 + 待办 + 历史）
  -> LLM 决策：输出 "工具调用" 或 "最终答案"
  -> 若工具调用：执行 -> 结果塞回上下文 -> 回到 LLM 决策
  -> 若最终答案：返回用户，结束
```

- 最多 `max_turns` 轮；超限给兜底回答。
- 工具执行异常、LLM 异常、解析异常都在 runtime 层捕获并记录 trace。

### 2. 工具注册机制（`agent/registry.py`）

每个工具是 `{name, description, parameters(JSON Schema), fn}`；`ToolRegistry.schema()`
把工具清单序列化成 JSON 注入 system prompt，LLM 依据名称、描述、参数 Schema 自主决定
调用哪个工具、传什么参数。工具函数统一签名 `fn(arguments, working_memory)`。已实现三个工具：

| 工具 | 说明 |
|---|---|
| `calculator` | 真实计算（ast 白名单安全求值，支持四则/括号/幂） |
| `search` | mock 检索，按关键词返回文本 |
| `todo` | session 内持久化待办（add/list/clear） |

### 3. LLM 输出解析（`agent/parser.py`）

约定模型输出（外层可带 `<thinking>`）：
```json
{"action":"tool","tool":"calculator","arguments":{"expression":"1+2"}}
{"action":"answer","answer":"结果是 3"}
```
解析器容错：兼容 `<think>/<thinking>/<thought>` 思考标签、去 markdown 围栏、从噪声里抠 JSON、
JSON 截断时补 `}` 恢复、arguments 被双重序列化时自动解包。

### 4. session 管理（`agent/session.py`）

`SessionStore` 以 `session_id` 为键；用户 A 的窗口 1 和窗口 2 各自持有独立的
`messages`（情景记忆）与 `working_memory`（工作记忆看板），可随时接着聊、互不影响。
内置 FIFO 容量上限，防止会话池无限膨胀。

### 5. context 管理（`agent/context.py`）

- **最大轮次**：`max_messages` 限制进入 prompt 的历史消息数；
- **追问**：纯对话追问靠保留历史；带工具追问靠把上轮工具结果留在消息里；
- **基础压缩**：历史过长时，保留最旧 2 条（锁定初始意图）+ 最近消息，中间折叠为摘要占位。

## 三、memory 的召回时机与放置方式

| memory 类型 | 召回时机 | 放置方式 | 为什么 |
|---|---|---|---|
| 结构化短期状态（todo，即 `working_memory`） | **每轮开始前** | **系统 prompt** | 模型每轮都能"记得"，又不污染对话消息流、不被压缩截断 |
| 对话历史（含工具结果） | 每轮组装 context 时 | 消息列表 | 支持连续追问与带工具追问 |
| 超长历史 | 超过 `max_messages` 时 | 折叠为一条摘要占位 | 基础压缩，防止上下文爆掉 |

核心取舍：**结构性、需要精确回读的状态（如 todo）放系统提示；叙述性、需要顺序的（对话、工具结果）放消息流。**

## 四、异常处理与 trace

- 工具执行失败（未注册/参数错/除零）→ 捕获并作为错误消息塞回上下文，让 LLM 决定下一步；
- LLM 网络/超时/HTTP 错误 → 捕获，返回友好错误；
- 解析失败 → 把错误提示塞回上下文让模型重试；
- `trace` 记录 `llm_call / thinking / tool_call / tool_result / tool_error / parse_error / answer / max_turns` 事件（含时间戳、耗时、token 用量、turn 序号）。

## 五、测试

- `tests/` 单元测试（21 个用例）：解析器容错、三个工具、注册机制、session 隔离、context 压缩、完整 loop（用 `FakeLLM` 确定性跑通工具调用→回答、工具异常、解析重试、多窗口隔离）。
- `run_all_tests.py` 全量测试：先跑单元测试，再用真实 LLM API 跑 6 个跨窗口用例并做 session 强隔离断言。
- `app.py` 内置"异常自愈对抗测试集"按钮：算非数字、诱导错误动作、复杂多工具、禁用 JSON，可实时观察 flash 犯错被 runtime 接住。

## 六、文件结构

```
agent/            # 核心模块：runtime / registry / tools / parser / session / context / llm
tests/            # 单元测试
docs/             # REQUIREMENTS / architecture_design(5模块) / AI_PROMPT_AND_SOLVING
main.py           # CLI 交互
app.py            # Gradio Web 控制台（核心用例 + 异常自愈用例）
run_all_tests.py  # 全量测试（单元 + 真实 API E2E）
```
