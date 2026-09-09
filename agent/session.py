from __future__ import annotations

"""会话状态与记忆持久化管理模块（agent/session.py）。

架构设计要点：
1. 【物理沙箱与会话隔离（Session Isolation）】：
   每个会话由唯一的 `session_id` 进行逻辑与物理隔离。
   不同窗口（如 window-1 与 window-2）拥有完全独立的情景记忆（messages）
   与工作记忆看板（working_memory），绝不会发生串号或状态互相污染。

2. 【双轨记忆模型（Dual-Track Memory）】：
   - 情景记忆（Episodic Memory，即 messages）：按时间线线性记录的对话轮次历史，
     后续通过 context.py 实施头尾截断与窗口控制。
   - 工作记忆（Working Memory，即 working_memory）：常驻任务状态看板（如 todo 列表）。
     无论对话历史如何滑动或截断，working_memory 始终保持常驻并挂载于系统提示词中。

3. 【资源约束与内存泄漏防御（OOM Protection）】：
   内置基于容量阈值（FIFO 驱逐）的 SessionStore 淘汰策略，
   避免长期高并发测试下会话字典无上限膨胀引发的内存溢出。
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# 严格约束标准角色枚举，防止非标准角色（如 'model', 'human'）导致 API 拒收
MessageRole = Literal["system", "user", "assistant"]


@dataclass
class Session:
    """单个独立会话实体对象。
    
    保存单次长任务或多轮人机交互中的所有动态上下文与状态看板。
    """
    
    # 窗口/会话的唯一标识字符串（例如 'user-a-window-1'）
    session_id: str
    
    # 线性对话历史记录（情景记忆），使用 field(default_factory=list) 确保每个实例持有独立空列表
    messages: list[dict[str, str]] = field(default_factory=list)
    
    # 【核心重构字段】：全局工作记忆看板，严格命名为 working_memory，杜绝历史包袱 tool_state
    # 供 todo 等有状态工具进行跨轮状态变更与持久化写入
    working_memory: dict[str, Any] = field(default_factory=dict)
    
    # 用户真实交互轮次计数器（仅在 user 角色输入时递增）
    turn_count: int = 0

    def append(self, role: MessageRole, content: str) -> None:
        """向当前会话追加一条对话历史记录。
        
        Args:
            role: 消息角色，仅允许 'system' | 'user' | 'assistant'。
            content: 消息纯文本内容。
        """
        self.messages.append({"role": role, "content": content})
        # 仅针对人类用户的提问进行轮次自增统计
        if role == "user":
            self.turn_count += 1

    def clear_history(self) -> None:
        """清空当前会话的线性对话历史，但保留当前 working_memory 看板状态。
        
        适用于用户主动点击“清空对话气泡”但希望保留待办事项看板的场景。
        """
        self.messages.clear()
        self.turn_count = 0


class SessionStore:
    """内存级会话存储仓库与生命周期调度中心。"""

    def __init__(self, max_sessions: int = 1000) -> None:
        """初始化存储仓库。
        
        Args:
            max_sessions: 最大允许驻留内存的会话上限，防止测试高并发时内存溢出。
        """
        # 会话字典映射：session_id -> Session 实体
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> Session:
        """检索指定会话；若不存在则自动实例化并挂载注册。
        
        Args:
            session_id: 目标会话唯一标识。
            
        Returns:
            对应的 Session 实例。
        """
        session = self._sessions.get(session_id)
        if session is None:
            # 容量安全防御：若会话池已满，优先驱逐最先创建的陈旧会话（FIFO）
            if len(self._sessions) >= self._max_sessions:
                oldest_key = next(iter(self._sessions))
                del self._sessions[oldest_key]

            # 实例化全新独立会话
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """安全查询已存在的会话，若不存在则返回 None（不主动创建）。"""
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        """彻底销毁并清理指定会话（释放其占用的全部历史记录与工作看板）。"""
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        """获取当前正在活跃挂载的会话总数。"""
        return len(self._sessions)