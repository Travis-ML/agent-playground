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


@mcp.tool()
def force_dream(cycle: str = "full", model: str = "vllm/local") -> dict:
    """Operator-only: manually trigger a dream cycle. Returns the run id."""
    with _open() as c:
        return handle_force_dream(conn=c, cycle=cycle, model=model)


def handle_force_dream(
    *, conn: sqlite3.Connection,
    cycle: str = "full",
    model: str = "vllm/local",
) -> dict:
    try:
        from mcp_servers.memory.dreamer_runner.runner import run_cycle
        from mcp_servers.memory.dreamer_runner.stages import all_stages
        dr = run_cycle(
            conn=conn, pid=os.getpid(),
            cycle_mode=cycle, trigger_reason="manual",
            model_used=model, stages=all_stages(),
        )
        return {"dream_run_id": dr.id, "status": dr.status}
    except Exception as e:
        return {"dream_run_id": None, "status": "failed", "error": str(e)}


def handle_status(*, conn: sqlite3.Connection) -> dict:
    counts = {}
    for table in ("raw_turn_refs", "episodes", "entities", "facts",
                  "reflections", "hypotheses", "links"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
    last = conn.execute(
        "SELECT id, started_at, ended_at, cycle_mode, status "
        "FROM dream_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return {
        "counts": counts,
        "last_dream_run": dict(last) if last else None,
    }


@mcp.resource("memory://status")
def status_resource() -> str:
    with _open() as c:
        return json.dumps(handle_status(conn=c), indent=2)


# Phase 14: HippoRAG retrieval tools
from mcp_servers.memory.repo.hypotheses import list_by_status as _list_hyp
from mcp_servers.memory.repo.links import list_links_from
from mcp_servers.memory.retrieval.recall import recall as _recall


def handle_recall(
    *, conn: sqlite3.Connection,
    query: str,
    max_results: int = 8,
    kinds: list[str] | None = None,
    embedder=None,
) -> dict:
    memories = _recall(
        conn=conn, query=query, embedder=embedder,
        max_results=max_results, kinds=kinds,
    )
    return {"memories": memories}


def handle_list_hypotheses(
    *, conn: sqlite3.Connection, status: str = "open", limit: int = 10,
) -> dict:
    rows = _list_hyp(conn, status, limit=limit)
    return {"hypotheses": [
        {"id": h.id, "statement": h.statement,
         "confidence": h.confidence, "status": h.status,
         "sources": h.source_node_ids, "created_at": h.created_at}
        for h in rows
    ]}


def handle_traverse_graph(
    *, conn: sqlite3.Connection,
    start_kind: str, start_id: str,
    max_hops: int = 2,
    link_types: list[str] | None = None,
) -> dict:
    seen: set[tuple[str, str]] = {(start_kind, start_id)}
    frontier: list[tuple[str, str]] = [(start_kind, start_id)]
    for _ in range(max_hops):
        next_frontier: list[tuple[str, str]] = []
        for (k, i) in frontier:
            for row in list_links_from(conn, src_kind=k, src_id=i):
                if link_types and row["link_type"] not in link_types:
                    continue
                key = (row["dst_kind"], row["dst_id"])
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append(key)
        frontier = next_frontier
    return {"nodes": [{"node_kind": k, "node_id": i} for (k, i) in seen]}


@mcp.tool()
def recall(query: str, max_results: int = 8, kinds: list[str] | None = None) -> dict:
    """Vector + PageRank-spread retrieval. Returns mixed-kind memories."""
    with _open() as c:
        return handle_recall(conn=c, query=query,
                             max_results=max_results, kinds=kinds)


@mcp.tool()
def list_hypotheses(status: str = "open", limit: int = 10) -> dict:
    """Surface dream-generated speculations."""
    with _open() as c:
        return handle_list_hypotheses(conn=c, status=status, limit=limit)


@mcp.tool()
def traverse_graph(
    start_kind: str, start_id: str,
    max_hops: int = 2, link_types: list[str] | None = None,
) -> dict:
    """Graph walk from a known node, optionally filtered by link types."""
    with _open() as c:
        return handle_traverse_graph(
            conn=c, start_kind=start_kind, start_id=start_id,
            max_hops=max_hops, link_types=link_types,
        )


if __name__ == "__main__":
    mcp.run()
