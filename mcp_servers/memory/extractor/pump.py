"""Pump that drains the pending raw_turn_refs queue, calling extract_for_turn."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp_servers.memory.extractor.worker import extract_for_turn
from mcp_servers.memory.providers.base import LLMClient
from mcp_servers.memory.repo.raw_turns import list_pending


def pump_once(
    *,
    conn: sqlite3.Connection,
    llm: LLMClient,
    conversations_root: str | Path,
    max_batch: int = 50,
) -> int:
    processed = 0
    for rt in list_pending(conn, limit=max_batch):
        extract_for_turn(
            conn=conn, llm=llm,
            conversations_root=conversations_root,
            raw_turn_id=rt.id,
        )
        processed += 1
    return processed
