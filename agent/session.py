from __future__ import annotations

"""session 管理：多窗口独立、可随时续聊、互不影响。

用户 A 的窗口 1 与窗口 2 用不同 `session_id` 区分；每个 Session 持有自己的
消息历史与工具状态（如 todo），因此窗口之间完全隔离。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_state: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
