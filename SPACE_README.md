---
title: Minimal Agent Harness
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🤖 Minimal Agent Harness（从零实现的 ReAct Agent）

一个**不依赖 langgraph / openhands / openclaw** 的最小可用 Agent，核心 Runtime 主循环完全自研，只用一个 `httpx` 调真实 LLM API。

> 这是 2026 Agent Harness 技术笔试的 vibe-coding 题作品：重点看**实现细节**——工具注册、LLM 输出解析容错、多窗口 session 隔离、context 管理、异常自愈与 trace。

## 直接体验（两个面板）

1. **🎯 核心验收测试集**（点按钮一键验证）：
   - 窗口 1 查天气 + 记待办 → 追问待办；
   - 窗口 2 记待办 → 验证**两个窗口完全隔离**；
   - 切回窗口 1 累加待办 + 工具计算 `500*0.8`。
2. **⚠️ 异常自愈对抗测试集**（看模型犯错被接住）：
   - 算"苹果加香蕉"（触发 parse_error + tool_error 双重自愈）；
   - 诱导用 `delete` 记待办（模型读懂 Schema 自适应）；
   - 复杂多工具连续调用；
   - 命令"禁用 JSON"（触发格式畸变自愈）。

也可以在下方的 Session ID 输入框切换窗口，或手动输入任意指令，右侧面板实时回显 **Trace 调用链**（llm_call 耗时 / tool_call 传参 / tool_result 返回 / 异常自愈）。

## 技术要点

| 能力 | 实现 |
|---|---|
| 主循环 | 自实现 ReAct：输入 → LLM 决策(回复/工具) → 调工具 → 据结果继续或返回 |
| 工具注册 | `name / description / JSON Schema`，LLM 基于 Schema 自主决策 |
| 3 个工具 | `calculator`(AST 白名单安全求值) / `search`(mock) / `todo`(session 持久化) |
| LLM 解析 | 容错：`<thinking>` 标签、markdown 围栏、截断 JSON 补 `}` 恢复 |
| session 隔离 | 多窗口 `session_id` 物理隔离 + 可续聊 |
| 记忆 | 双轨：`working_memory`（待办，挂系统提示词）+ 情景记忆（消息流，Head-Tail 压缩） |
| 异常自愈 | parse_error 回喂重试 + tool_error 转观测回喂 |
| 可观测 | 全链路 Trace |

## 代码

完整源码（含单元测试 + 架构设计题答案 + AI Prompt 与问题解决记录）见 GitHub：
https://github.com/my420254/agent-harness-exam
