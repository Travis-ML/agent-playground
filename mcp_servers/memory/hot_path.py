"""Helpers the playground uses to record turns into memory."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.raw_turns import record_turn


def on_turn_appended(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_index: int,
    role: str,
    occurred_at: str,
) -> None:
    """Single entry point the playground calls. Wraps record_turn in a try
    so that memory failures never break the chat flow."""
    try:
        record_turn(
            conn,
            conversation_id=conversation_id,
            turn_index=turn_index,
            role=role,
            occurred_at=occurred_at,
        )
    except Exception:
        pass  # memory degrades gracefully
