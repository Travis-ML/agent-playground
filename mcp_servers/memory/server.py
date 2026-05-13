"""FastMCP server for the memory subsystem.

Run standalone (stdio): `python -m mcp_servers.memory.server`
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.repo.entities import (
    get_by_canonical_name,
)
from mcp_servers.memory.repo.raw_turns import record_turn

_DEFAULT_DB = Path.home() / ".travisml-playground" / "memory.db"


def _open() -> sqlite3.Connection:
    p = Path(os.getenv("TRAVISML_MEMORY_DB", str(_DEFAULT_DB)))
    conn = open_connection(p)
    apply_migrations(conn)
    return conn


def handle_record_turn(
    *, conn: sqlite3.Connection,
    conversation_id: str, turn_index: int, role: str, occurred_at: str,
) -> dict[str, Any]:
    rt = record_turn(
        conn, conversation_id=conversation_id, turn_index=turn_index,
        role=role, occurred_at=occurred_at,
    )
    return {"status": "ok", "raw_turn_id": rt.id}


def handle_search_episodes(
    *, conn: sqlite3.Connection,
    actor: str | None = None, since: str | None = None, until: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    clauses = []
    params: list[Any] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if since:
        clauses.append("occurred_at >= ?")
        params.append(since)
    if until:
        clauses.append("occurred_at <= ?")
        params.append(until)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM episodes {where} ORDER BY occurred_at DESC LIMIT ?", params
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "actor": r["actor"], "predicate": r["predicate"],
            "summary": r["summary"], "importance": r["importance"],
            "occurred_at": r["occurred_at"],
        })
    return {"episodes": out}


def handle_search_facts(
    *, conn: sqlite3.Connection,
    subject_canonical_name: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_invalidated: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    subject_id = None
    if subject_canonical_name:
        e = get_by_canonical_name(conn, subject_canonical_name)
        if e is None:
            return {"facts": []}
        subject_id = e.id
    clauses = ["1 = 1"]
    params: list[Any] = []
    if subject_id:
        clauses.append("subject_entity = ?")
        params.append(subject_id)
    if predicate:
        clauses.append("predicate = ?")
        params.append(predicate)
    if as_of:
        clauses += [
            "valid_from <= ?",
            "(valid_to IS NULL OR valid_to > ?)",
            "learned_at <= ?",
            "(invalidated_at IS NULL OR invalidated_at > ?)",
        ]
        params += [as_of, as_of, as_of, as_of]
    elif not include_invalidated:
        clauses += ["valid_to IS NULL", "invalidated_at IS NULL"]
    params.append(limit)
    sql = f"SELECT * FROM facts WHERE {' AND '.join(clauses)} ORDER BY learned_at DESC LIMIT ?"
    rows = conn.execute(sql, params).fetchall()
    facts = [{
        "id": r["id"], "subject_entity": r["subject_entity"],
        "predicate": r["predicate"], "object_entity": r["object_entity"],
        "object_value": r["object_value"], "valid_from": r["valid_from"],
        "valid_to": r["valid_to"], "learned_at": r["learned_at"],
        "confidence": r["confidence"],
    } for r in rows]
    return {"facts": facts}


def handle_get_entity(
    *, conn: sqlite3.Connection, name: str,
) -> dict[str, Any]:
    ent = get_by_canonical_name(conn, name)
    if ent is None:
        return {"entity": None}
    facts = conn.execute(
        """
        SELECT * FROM facts WHERE subject_entity = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        ORDER BY learned_at DESC LIMIT 10
        """, (ent.id,)
    ).fetchall()
    eps = conn.execute(
        """
        SELECT * FROM episodes
        WHERE subject_entity = ? OR object_entity = ?
        ORDER BY occurred_at DESC LIMIT 10
        """, (ent.id, ent.id)
    ).fetchall()
    return {
        "entity": {
            "id": ent.id, "canonical_name": ent.canonical_name,
            "kind": ent.kind, "summary": ent.summary,
            "first_seen": ent.first_seen, "last_seen": ent.last_seen,
            "importance": ent.importance,
        },
        "recent_facts": [{
            "id": r["id"], "predicate": r["predicate"],
            "object_value": r["object_value"], "object_entity": r["object_entity"],
            "learned_at": r["learned_at"],
        } for r in facts],
        "recent_episodes": [{
            "id": r["id"], "actor": r["actor"], "summary": r["summary"],
            "occurred_at": r["occurred_at"],
        } for r in eps],
    }


mcp = FastMCP("memory")


@mcp.tool()
def record_turn_tool(
    conversation_id: str, turn_index: int, role: str, occurred_at: str,
) -> dict:
    """Hot-path write: append a raw turn ref. Called by the playground after
    every chat-turn append. Returns the new raw_turn_id."""
    with _open() as c:
        return handle_record_turn(
            conn=c, conversation_id=conversation_id, turn_index=turn_index,
            role=role, occurred_at=occurred_at,
        )


@mcp.tool()
def search_episodes(
    actor: str | None = None, since: str | None = None,
    until: str | None = None, limit: int = 20,
) -> dict:
    """Search atomic episodes by actor and/or time range."""
    with _open() as c:
        return handle_search_episodes(
            conn=c, actor=actor, since=since, until=until, limit=limit,
        )


@mcp.tool()
def search_facts(
    subject_canonical_name: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_invalidated: bool = False,
    limit: int = 50,
) -> dict:
    """Search bi-temporal facts. Defaults to currently-believed facts; pass
    `as_of` for time-travel queries."""
    with _open() as c:
        return handle_search_facts(
            conn=c, subject_canonical_name=subject_canonical_name,
            predicate=predicate, as_of=as_of,
            include_invalidated=include_invalidated, limit=limit,
        )


@mcp.tool()
def get_entity(name: str) -> dict:
    """Look up an entity by canonical name; returns dossier with recent
    facts and recent episodes."""
    with _open() as c:
        return handle_get_entity(conn=c, name=name)


if __name__ == "__main__":
    mcp.run()
