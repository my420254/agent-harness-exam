# AI Prompt 与问题解决记录

> 记录开发这个最小 Agent 时：给 LLM 的核心 Prompt 怎么设计的、以及开发中踩到的问题和解决方式。

## 一、给 Agent 的核心系统 Prompt 设计

实际喂给 LLM 的 system prompt（`agent/context.py::build_system_prompt`）结构：

```
你是一个助手 Agent，只能通过下面的工具获取信息或执行操作。
工具清单（JSON）：[...]
当前待办（todo，如有）：[...]

每轮只输出一个动作，格式如下（可先给 <thinking>...</thinking>，再给 JSON）：
调用工具：{"action":"tool","tool":"<名称>","arguments":{...}}
最终回答：{"action":"answer","answer":"..."}
规则：需要信息就调用工具；信息足够就直接 answer；不要反问，用工具或上下文解决。
```

三个关键设计决策：

1. **工具清单放 system prompt 而不是历史消息**：工具 Schema 是"每轮都需要的稳定信息"，放系统层既能命中前缀缓存，又不占用历史窗口。
2. **待办（结构化状态）也放 system prompt**：见 README「memory 召回时机与放置方式」，结构状态放系统层、叙述状态放消息流。
3. **动作用「JSON + action 字段」二选一**：比让模型自由发挥更可解析，比完整 function calling 更轻、更透明；`thinking` 单独用 `<thinking>` 标签，和动作解耦，方便 trace 和调试。

## 二、问题解决记录

| # | 问题 | 解决 |
|---|---|---|
| 1 | LLM 输出可能带 markdown 围栏、前后废话、JSON 被截断 | 解析器先正则抓 `<thinking>`、去 ```json``` 围栏，再抠第一个 `{...}` 块；截断时逐个补 `}` 重试解析（最多 4 个），保证"买牛奶"这类被截断的工具调用仍能恢复 |
| 2 | `eval(expression)` 有任意代码执行风险 | 用 `ast` 白名单：只允许数字、四则运算、括号、幂、取模，其余 AST 节点一律拒绝 |
| 3 | 除零、未注册工具、参数缺失会导致整轮崩溃 | 工具调用统一在 runtime 层 try-except，异常转成"工具调用失败：xxx"消息塞回上下文，让 LLM 自己决定下一步（不静默吞掉，也 trace 记录） |
| 4 | 多窗口 session 隔离 + 可续聊 | `SessionStore` 按 `session_id` 隔离 `messages` 和 `tool_state`；todo 写进 session 自己的 state，不落全局 |
| 5 | 上下文过长会爆 | 超过 `max_messages` 时，保留系统提示 + 最旧 2 条 + 最近 N 条，中间折叠成一条"已压缩"占位，实现基础压缩 |
| 6 | 测试环境 `pytest` 报 `anyio` 插件冲突 | 是开发机环境里 anyio 插件和旧 pytest 不兼容，与项目无关；用 `pytest -p no:anyio` 绕过，用户干净环境直接 `pytest` 即可 |

## 三、开发中让 LLM 辅助的方式

本题的核心 Runtime、工具注册、解析、session、context 均按上面设计自行实现；LLM 只承担"运行时决策"这一件事。开发阶段用 LLM 辅助做了两件事：

1. **审查代码边界**：让 LLM 逐条对照题目要求（从零实现、3 工具、session 隔离、context 压缩、异常/trace），查漏补缺；
2. **推演测试盲区**：让 LLM 站在面试官角度列出"最容易漏测/最容易被追问"的点，据此补了"截断 JSON 恢复""除零""未知工具""多窗口隔离"这几个用例。
