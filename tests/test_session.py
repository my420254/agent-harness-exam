from __future__ import annotations

from agent.session import SessionStore


def test_sessions_are_isolated():
    store = SessionStore()
    a1 = store.get_or_create("user-a-1")
    a2 = store.get_or_create("user-a-2")
    a1.append("user", "查天气")
    a1.tool_state["todos"] = ["带伞"]

    assert store.get_or_create("user-a-2").messages == []
    assert store.get_or_create("user-a-2").tool_state == {}


def test_session_is_resumable():
    store = SessionStore()
    session = store.get_or_create("s1")
    session.append("user", "hello")
    # 再次取回同一 session，历史还在
    again = store.get_or_create("s1")
    assert again.messages[-1] == {"role": "user", "content": "hello"}
    assert again.turn_count == 1


def test_reset_clears_session():
    store = SessionStore()
    store.get_or_create("s1").append("user", "hello")
    store.reset("s1")
    assert store.get_or_create("s1").messages == []
