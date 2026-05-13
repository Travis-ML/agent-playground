import sqlite3

from mcp_servers.memory.repo.raw_turns import (
    list_pending,
    mark_extraction_status,
    record_turn,
)


def test_record_turn_inserts_pending_row(conn: sqlite3.Connection) -> None:
    rt = record_turn(
        conn,
        conversation_id="2026-05-12T15-00-00-aaaa",
        turn_index=0,
        role="user",
        occurred_at="2026-05-12T15:00:01Z",
    )
    assert rt.extraction_status == "pending"
    assert rt.retry_count == 0
    row = conn.execute(
        "SELECT id, role, extraction_status FROM raw_turn_refs"
    ).fetchone()
    assert row["id"] == rt.id
    assert row["role"] == "user"
    assert row["extraction_status"] == "pending"


def test_record_turn_is_idempotent_on_dup_key(conn: sqlite3.Connection) -> None:
    a = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    b = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    assert a.id == b.id


def test_list_pending_returns_only_pending(conn: sqlite3.Connection) -> None:
    a = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    b = record_turn(conn, conversation_id="c1", turn_index=1, role="assistant",
                    occurred_at="2026-05-12T15:00:02Z")
    mark_extraction_status(conn, b.id, "done")
    pending = list_pending(conn)
    assert [p.id for p in pending] == [a.id]
