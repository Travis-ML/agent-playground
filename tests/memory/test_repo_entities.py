"""Tests for entities repo."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.entities import (
    get_by_canonical_name, get_or_create, list_top_importance, touch_seen,
)


def test_get_or_create_inserts_then_returns_existing(conn: sqlite3.Connection) -> None:
    e1 = get_or_create(conn, canonical_name="MCP pool", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    e2 = get_or_create(conn, canonical_name="MCP pool", kind="concept",
                       seen_at="2026-05-12T15:05:00Z")
    assert e1.id == e2.id
    e3 = get_by_canonical_name(conn, "MCP pool")
    assert e3.last_seen == "2026-05-12T15:05:00Z"


def test_list_top_importance_orders_desc(conn: sqlite3.Connection) -> None:
    a = get_or_create(conn, canonical_name="A", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    b = get_or_create(conn, canonical_name="B", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    conn.execute("UPDATE entities SET importance = 0.9 WHERE id = ?", (a.id,))
    conn.execute("UPDATE entities SET importance = 0.1 WHERE id = ?", (b.id,))
    top = list_top_importance(conn, limit=10)
    assert [t.id for t in top[:2]] == [a.id, b.id]


def test_touch_seen_updates_last_seen(conn: sqlite3.Connection) -> None:
    e = get_or_create(conn, canonical_name="X", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    touch_seen(conn, e.id, "2026-05-12T16:00:00Z")
    refreshed = get_by_canonical_name(conn, "X")
    assert refreshed.last_seen == "2026-05-12T16:00:00Z"
