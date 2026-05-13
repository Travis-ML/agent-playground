"""Repo for entities (people / projects / concepts / tools / files / ...)."""

from __future__ import annotations

import json
import sqlite3

from mcp_servers.memory.ids import new_entity_id
from mcp_servers.memory.models import Entity


def get_or_create(
    conn: sqlite3.Connection, *, canonical_name: str, kind: str, seen_at: str,
) -> Entity:
    row = conn.execute(
        "SELECT * FROM entities WHERE canonical_name = ?", (canonical_name,)
    ).fetchone()
    if row is not None:
        conn.execute(
            "UPDATE entities SET last_seen = ? WHERE id = ?", (seen_at, row["id"]),
        )
        return _row(_refresh(conn, row["id"]))
    e_id = new_entity_id()
    conn.execute(
        """
        INSERT INTO entities
            (id, canonical_name, kind, aliases, summary, first_seen, last_seen, importance)
        VALUES (?, ?, ?, '[]', NULL, ?, ?, 0.5)
        """,
        (e_id, canonical_name, kind, seen_at, seen_at),
    )
    return _row(_refresh(conn, e_id))


def get_by_canonical_name(conn: sqlite3.Connection, name: str) -> Entity | None:
    row = conn.execute(
        "SELECT * FROM entities WHERE canonical_name = ?", (name,)
    ).fetchone()
    return _row(row) if row else None


def get_by_id(conn: sqlite3.Connection, entity_id: str) -> Entity | None:
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return _row(row) if row else None


def touch_seen(conn: sqlite3.Connection, entity_id: str, seen_at: str) -> None:
    conn.execute(
        "UPDATE entities SET last_seen = ? WHERE id = ?", (seen_at, entity_id),
    )


def list_top_importance(
    conn: sqlite3.Connection, *, limit: int = 50, kind: str | None = None,
) -> list[Entity]:
    if kind is None:
        rows = conn.execute(
            "SELECT * FROM entities ORDER BY importance DESC, last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entities WHERE kind = ? ORDER BY importance DESC, last_seen DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
    return [_row(r) for r in rows]


def _refresh(conn: sqlite3.Connection, entity_id: str) -> sqlite3.Row:
    return conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()


def _row(r: sqlite3.Row) -> Entity:
    return Entity(
        id=r["id"], canonical_name=r["canonical_name"], kind=r["kind"],
        aliases=json.loads(r["aliases"]),
        summary=r["summary"], first_seen=r["first_seen"], last_seen=r["last_seen"],
        importance=r["importance"],
    )
