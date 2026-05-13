"""Repo for episodes — atomic episodic events extracted from raw turns."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_episode_id
from mcp_servers.memory.models import Episode


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_episode(
    conn: sqlite3.Connection,
    *,
    actor: str,
    predicate: str,
    subject_entity: str | None,
    object_entity: str | None,
    object_value: str | None,
    summary: str,
    importance: float,
    occurred_at: str,
    source_refs: list[dict],
) -> Episode:
    ep_id = new_episode_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO episodes
            (id, actor, predicate, subject_entity, object_entity, object_value,
             summary, importance, occurred_at, created_at, status, source_refs)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fresh', ?)
        """,
        (ep_id, actor, predicate, subject_entity, object_entity, object_value,
         summary, importance, occurred_at, created_at, json.dumps(source_refs)),
    )
    return Episode(
        id=ep_id, actor=actor, predicate=predicate,
        subject_entity=subject_entity, object_entity=object_entity,
        object_value=object_value, summary=summary, importance=importance,
        occurred_at=occurred_at, created_at=created_at,
        status="fresh", source_refs=source_refs,
    )


def set_status(conn: sqlite3.Connection, episode_id: str, status: str) -> None:
    conn.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id))


def list_by_status(
    conn: sqlite3.Connection, status: str, *, limit: int = 1000
) -> list[Episode]:
    rows = conn.execute(
        "SELECT * FROM episodes WHERE status = ? ORDER BY occurred_at LIMIT ?",
        (status, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Episode:
    return Episode(
        id=r["id"], actor=r["actor"], predicate=r["predicate"],
        subject_entity=r["subject_entity"], object_entity=r["object_entity"],
        object_value=r["object_value"], summary=r["summary"],
        importance=r["importance"], occurred_at=r["occurred_at"],
        created_at=r["created_at"], status=r["status"],
        source_refs=json.loads(r["source_refs"]),
    )
