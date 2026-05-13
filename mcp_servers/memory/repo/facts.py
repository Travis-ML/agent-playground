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


def supersede_fact(
    conn: sqlite3.Connection,
    *,
    old_fact_id: str,
    new_object_entity: str | None,
    new_object_value: str | None,
    change_time: str,
    source_episode_ids: list[str],
    confidence: float,
    created_in_dream_run: str,
) -> Fact:
    old = get_by_id(conn, old_fact_id)
    if old is None:
        raise KeyError(old_fact_id)
    if old.superseded_by is not None:
        raise ValueError(f"fact {old_fact_id} already superseded")

    new = insert_new_fact(
        conn,
        subject_entity=old.subject_entity,
        predicate=old.predicate,
        object_entity=new_object_entity,
        object_value=new_object_value,
        valid_from=change_time,
        learned_at=change_time,
        source_episode_ids=source_episode_ids,
        confidence=confidence,
        created_in_dream_run=created_in_dream_run,
        supersedes=old_fact_id,
    )
    conn.execute(
        """
        UPDATE facts
        SET valid_to = ?, invalidated_at = ?, superseded_by = ?
        WHERE id = ?
        """,
        (change_time, change_time, new.id, old_fact_id),
    )
    return new


def list_facts_for_subject_predicate(
    conn: sqlite3.Connection, *,
    subject_entity: str, predicate: str, currently_believed: bool = False,
) -> list[Fact]:
    if currently_believed:
        rows = conn.execute(
            """
            SELECT * FROM facts
            WHERE subject_entity = ? AND predicate = ?
              AND valid_to IS NULL AND invalidated_at IS NULL
            ORDER BY learned_at DESC
            """,
            (subject_entity, predicate),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM facts
            WHERE subject_entity = ? AND predicate = ?
            ORDER BY learned_at
            """,
            (subject_entity, predicate),
        ).fetchall()
    return [_row(r) for r in rows]


def current_belief_as_of(
    conn: sqlite3.Connection,
    *,
    subject_entity: str,
    predicate: str,
    as_of: str,
) -> Fact | None:
    row = conn.execute(
        """
        SELECT * FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_from <= ?
          AND (valid_to IS NULL OR valid_to > ?)
          AND learned_at <= ?
          AND (invalidated_at IS NULL OR invalidated_at > ?)
        ORDER BY learned_at DESC LIMIT 1
        """,
        (subject_entity, predicate, as_of, as_of, as_of, as_of),
    ).fetchone()
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
