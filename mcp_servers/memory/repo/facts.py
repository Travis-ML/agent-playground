"""Bi-temporal facts repo. Supersession + time-travel queries live here."""

from __future__ import annotations

import json
import sqlite3

from mcp_servers.memory.ids import new_fact_id
from mcp_servers.memory.models import Fact


def insert_new_fact(
    conn: sqlite3.Connection,
    *,
    subject_entity: str,
    predicate: str,
    object_entity: str | None,
    object_value: str | None,
    valid_from: str,
    learned_at: str,
    source_episode_ids: list[str],
    confidence: float,
    created_in_dream_run: str,
    supersedes: str | None = None,
) -> Fact:
    if object_entity is None and object_value is None:
        raise ValueError("must supply object_entity or object_value")
    f_id = new_fact_id()
    conn.execute(
        """
        INSERT INTO facts
            (id, subject_entity, predicate, object_entity, object_value,
             valid_from, valid_to, learned_at, invalidated_at,
             source_episode_ids, confidence, supersedes, superseded_by,
             created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, NULL, ?)
        """,
        (f_id, subject_entity, predicate, object_entity, object_value,
         valid_from, learned_at, json.dumps(source_episode_ids),
         confidence, supersedes, created_in_dream_run),
    )
    return get_by_id(conn, f_id)  # type: ignore[return-value]


def current_belief(
    conn: sqlite3.Connection, *, subject_entity: str, predicate: str,
) -> Fact | None:
    row = conn.execute(
        """
        SELECT * FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        ORDER BY learned_at DESC LIMIT 1
        """,
        (subject_entity, predicate),
    ).fetchone()
    return _row(row) if row else None


def get_by_id(conn: sqlite3.Connection, fact_id: str) -> Fact | None:
    row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return _row(row) if row else None


def _row(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"], subject_entity=r["subject_entity"], predicate=r["predicate"],
        object_entity=r["object_entity"], object_value=r["object_value"],
        valid_from=r["valid_from"], valid_to=r["valid_to"],
        learned_at=r["learned_at"], invalidated_at=r["invalidated_at"],
        source_episode_ids=json.loads(r["source_episode_ids"]),
        confidence=r["confidence"], supersedes=r["supersedes"],
        superseded_by=r["superseded_by"],
        created_in_dream_run=r["created_in_dream_run"],
    )
