import sqlite3

from mcp_servers.memory.dreamer_runner.entity_resolve import resolve_entity


def test_resolve_entity_creates_then_reuses(conn: sqlite3.Connection) -> None:
    a = resolve_entity(conn, canonical="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    b = resolve_entity(conn, canonical="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    assert a == b
