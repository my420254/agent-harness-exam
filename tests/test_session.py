from __future__ import annotations

"""会话存储与隔离机制单元测试（tests/test_session.py）。

测试覆盖场景：
1. 会话物理隔离（Session Isolation）：验证不同 session_id 之间的消息历史与 working_memory 看板互不干扰。
2. 会话断点续聊（Resumability）：验证通过相同的 session_id 再次检索会话时，历史与轮次计数完全保留。
3. 会话销毁重置（Session Reset）：验证 reset 接口能彻底清理历史与状态，重置后获取到的是干净会话。
"""

from agent.session import SessionStore


def test_sessions_are_isolated() -> None:
    """验证多个并发会话之间的情景记忆与工作记忆看板绝对物理隔离。"""
    store = SessionStore()
    a1 = store.get_or_create("user-a-1")
    a2 = store.get_or_create("user-a-2")

    # 对窗口 1 进行写入操作
    a1.append("user", "查天气")
    # 【统一对齐】：严格使用 working_memory 看板
    a1.working_memory["todos"] = ["带伞"]

    # 断言窗口 2 依然处于纯净的初始化状态，未被窗口 1 污染
    session_a2 = store.get_or_create("user-a-2")
    assert session_a2.messages == []
    assert session_a2.working_memory == {}


def test_session_is_resumable() -> None:
    """验证会话可随时挂起与恢复续聊，历史记录与轮次计数状态保持一致。"""
    store = SessionStore()
    session = store.get_or_create("s1")
    session.append("user", "hello")

    # 模拟用户后续轮次再次通过同一 session_id 访问
    again = store.get_or_create("s1")
    assert again.messages[-1] == {"role": "user", "content": "hello"}
    assert again.turn_count == 1


def test_reset_clears_session() -> None:
    """验证重置销毁会话后，原数据被彻底回收，再次创建为全新空会话。"""
    store = SessionStore()
    store.get_or_create("s1").append("user", "hello")

    # 触发彻底重置
    store.reset("s1")

    # 再次检索应为重新初始化的空会话
    assert store.get_or_create("s1").messages == []