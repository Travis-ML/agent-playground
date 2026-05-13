import sqlite3

from mcp_servers.memory.hot_path import on_turn_appended


def test_on_turn_appended_writes_row(conn: sqlite3.Connection) -> None:
    on_turn_appended(conn, conversation_id="c1", turn_index=0,
                     role="user", occurred_at="2026-05-12T15:00:01Z")
    row = conn.execute("SELECT id FROM raw_turn_refs").fetchone()
    assert row is not None


def test_on_turn_appended_swallows_errors() -> None:
    # closed connection should NOT raise out of the helper
    bad = sqlite3.connect(":memory:")
    bad.close()
    on_turn_appended(bad, conversation_id="c1", turn_index=0,
                     role="user", occurred_at="2026-05-12T15:00:01Z")
