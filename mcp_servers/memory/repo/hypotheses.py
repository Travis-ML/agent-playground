"""Hypotheses repo — first-class speculations from the recombine stage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_hypothesis_id
from mcp_servers.memory.models import Hypothesis


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_hypothesis(
    conn: sqlite3.Connection, *,
    statement: str, source_node_ids: list[str],
    confidence: float, created_in_dream_run: str,
) -> Hypothesis:
    h_id = new_hypothesis_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO hypotheses
            (id, statement, source_node_ids, confidence, status,
             created_at, created_in_dream_run)
        VALUES (?, ?, ?, ?, 'open', ?, ?)
        """,
        (h_id, statement, json.dumps(source_node_ids), confidence,
         created_at, created_in_dream_run),
    )
    return Hypothesis(
        id=h_id, statement=statement, source_node_ids=source_node_ids,
        confidence=confidence, status="open",
        resolved_at=None, resolved_by=None, resolution_note=None,
        created_at=created_at, created_in_dream_run=created_in_dream_run,
    )


def resolve(
    conn: sqlite3.Connection,
    hypothesis_id: str,
    *,
    status: str,
    resolved_by: str,
    note: str | None = None,
) -> None:
    if status not in ("corroborated", "refuted", "set_aside"):
        raise ValueError(f"invalid status: {status}")
    conn.execute(
        """
        UPDATE hypotheses
        SET status = ?, resolved_at = ?, resolved_by = ?, resolution_note = ?
        WHERE id = ?
        """,
        (status, _now(), resolved_by, note, hypothesis_id),
    )


def list_by_status(
    conn: sqlite3.Connection, status: str, *, limit: int = 100,
) -> list[Hypothesis]:
    rows = conn.execute(
        "SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at DESC LIMIT ?",
        (status, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Hypothesis:
    return Hypothesis(
        id=r["id"], statement=r["statement"],
        source_node_ids=json.loads(r["source_node_ids"]),
        confidence=r["confidence"], status=r["status"],
        resolved_at=r["resolved_at"], resolved_by=r["resolved_by"],
        resolution_note=r["resolution_note"],
        created_at=r["created_at"], created_in_dream_run=r["created_in_dream_run"],
    )
