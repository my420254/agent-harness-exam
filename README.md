# Minimal Agent Harness（2026 Agent 技术笔试 · Vibe Coding 题）

从零实现的最小可用 Agent：不依赖 langgraph / openhands / openclaw 等任何 agent 框架，
核心 Runtime 主循环自行实现，只用一个第三方库 `httpx` 调真实 LLM API。

## 一、运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置真实 LLM API（OpenAI-compatible）
cp .env.example .env
# 编辑 .env：填 LLM_API_BASE / LLM_API_KEY / LLM_MODEL
# 或用环境变量：
export LLM_API_BASE=https://api.deepseek.com
export LLM_API_KEY=sk-xxx
export LLM_MODEL=deepseek-v4-flash
# 也支持本机 vLLM Qwen：LLM_API_BASE=http://192.168.27.250:18003/v1  LLM_MODEL=Qwen3.6-27B

# 3. 交互式聊天（多窗口）
python main.py

# 4. 跑双窗口演示（查天气记待办 / 写周报记待办，验证 session 隔离）
python main.py --demo

# 5. 跑真实模型的错误处理演示（用刁难任务看 flash 犯错被接住）
python demo_errors.py

# 6. 跑测试
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
调用哪个工具、传什么参数。已实现三个工具：

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
解析器容错：正则抓 `<thinking>`、去 markdown 围栏、从噪声里抠 JSON、JSON 截断时补 `}` 恢复。

### 4. session 管理（`agent/session.py`）

`SessionStore` 以 `session_id` 为键；用户 A 的窗口 1（`user-a-1`）和窗口 2（`user-a-2`）
各自持有独立的 `messages` 与 `tool_state`，可随时接着聊、互不影响。

### 5. context 管理（`agent/context.py`）

- **最大轮次**：`max_messages` 限制进入 prompt 的历史消息数；
- **追问**：纯对话追问靠保留历史；带工具追问靠把上轮工具结果留在消息里；
- **基础压缩**：历史过长时，把最旧的中间轮次折叠成一条摘要占位，保留系统提示 + 最近消息。

## 三、memory 的召回时机与放置方式

| memory 类型 | 召回时机 | 放置方式 | 为什么 |
|---|---|---|---|
| 结构化短期状态（todo） | **每轮开始前** | **系统 prompt** | 模型每轮都能"记得"，又不污染对话消息流、不被压缩截断 |
| 对话历史（含工具结果） | 每轮组装 context 时 | 消息列表 | 支持连续追问与带工具追问 |
| 超长历史 | 超过 `max_messages` 时 | 折叠为一条摘要占位 | 基础压缩，防止上下文爆掉 |

核心取舍：**结构性、需要精确回读的状态（如 todo）放系统提示；叙述性、需要顺序的（对话、工具结果）放消息流。** 这样既保证"记住"，又避免把结构化数据塞进对话噪声里。

## 四、异常处理与 trace

- 工具执行失败（未注册/参数错/除零）→ 捕获并作为错误消息塞回上下文，让 LLM 决定下一步；
- LLM 网络/超时/HTTP 错误 → 捕获，返回友好错误；
- 解析失败 → 把错误提示塞回上下文让模型重试；
- `trace` 记录 `llm_call / thinking / tool_call / tool_result / tool_error / parse_error / answer / max_turns` 事件（含时间戳、耗时、token 用量）。

## 五、测试

`tests/` 覆盖：解析器（含畸形/截断 JSON 容错）、三个工具、注册机制、session 隔离与续聊、
context 压缩、完整 loop（用 `FakeLLM` 确定性跑通工具调用→回答、工具异常、解析重试、多窗口隔离）。
共 25 个用例。
