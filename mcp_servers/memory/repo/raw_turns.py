"""Repo for raw_turn_refs — pointer rows into existing conversations/*.json."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_raw_turn_id
from mcp_servers.memory.models import RawTurnRef


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_turn(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_index: int,
    role: str,
    occurred_at: str,
) -> RawTurnRef:
    existing = conn.execute(
        "SELECT * FROM raw_turn_refs WHERE conversation_id = ? AND turn_index = ?",
        (conversation_id, turn_index),
    ).fetchone()
    if existing is not None:
        return _row_to_model(existing)
    rt_id = new_raw_turn_id()
    recorded_at = _now()
    conn.execute(
        """
        INSERT INTO raw_turn_refs
            (id, conversation_id, turn_index, role, occurred_at, recorded_at, extraction_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (rt_id, conversation_id, turn_index, role, occurred_at, recorded_at),
    )
    return RawTurnRef(
        id=rt_id,
        conversation_id=conversation_id,
        turn_index=turn_index,
        role=role,
        occurred_at=occurred_at,
        recorded_at=recorded_at,
        extraction_status="pending",
    )


def mark_extraction_status(
    conn: sqlite3.Connection,
    raw_turn_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if status == "failed":
        conn.execute(
            (
                "UPDATE raw_turn_refs "
                "SET extraction_status = ?, retry_count = retry_count + 1, "
                "last_error = ? WHERE id = ?"
            ),
            (status, error, raw_turn_id),
        )
    else:
        conn.execute(
            (
                "UPDATE raw_turn_refs "
                "SET extraction_status = ?, last_error = ? WHERE id = ?"
            ),
            (status, error, raw_turn_id),
        )


def list_pending(conn: sqlite3.Connection, *, limit: int = 100) -> list[RawTurnRef]:
    rows = conn.execute(
        (
            "SELECT * FROM raw_turn_refs "
            "WHERE extraction_status = 'pending' ORDER BY recorded_at LIMIT ?"
        ),
        (limit,),
    ).fetchall()
    return [_row_to_model(r) for r in rows]


def _row_to_model(row: sqlite3.Row) -> RawTurnRef:
    return RawTurnRef(
        id=row["id"],
        conversation_id=row["conversation_id"],
        turn_index=row["turn_index"],
        role=row["role"],
        occurred_at=row["occurred_at"],
        recorded_at=row["recorded_at"],
        extraction_status=row["extraction_status"],
        retry_count=row["retry_count"],
        last_error=row["last_error"],
    )
