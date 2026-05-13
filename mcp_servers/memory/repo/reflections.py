"""Reflections repo — recursive synthesized insights."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_reflection_id
from mcp_servers.memory.models import Reflection


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_reflection(
    conn: sqlite3.Connection,
    *,
    summary: str,
    importance: float,
    level: int,
    source_kind: str,
    source_ids: list[str],
    created_in_dream_run: str,
) -> Reflection:
    r_id = new_reflection_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO reflections
            (id, summary, importance, level, source_kind, source_ids,
             created_at, created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (r_id, summary, importance, level, source_kind,
         json.dumps(source_ids), created_at, created_in_dream_run),
    )
    return Reflection(
        id=r_id, summary=summary, importance=importance, level=level,
        source_kind=source_kind, source_ids=source_ids,
        created_at=created_at, created_in_dream_run=created_in_dream_run,
    )


def list_by_level(
    conn: sqlite3.Connection, *, level: int, limit: int = 100,
) -> list[Reflection]:
    rows = conn.execute(
        "SELECT * FROM reflections WHERE level = ? ORDER BY created_at DESC LIMIT ?",
        (level, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def list_recent(
    conn: sqlite3.Connection, *, min_level: int = 1, limit: int = 20,
) -> list[Reflection]:
    rows = conn.execute(
        "SELECT * FROM reflections WHERE level >= ? ORDER BY created_at DESC LIMIT ?",
        (min_level, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Reflection:
    return Reflection(
        id=r["id"], summary=r["summary"], importance=r["importance"],
        level=r["level"], source_kind=r["source_kind"],
        source_ids=json.loads(r["source_ids"]),
        created_at=r["created_at"], created_in_dream_run=r["created_in_dream_run"],
    )
