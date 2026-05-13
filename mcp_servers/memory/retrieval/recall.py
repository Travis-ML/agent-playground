"""HippoRAG-style recall: vector top-K seed → personalized PageRank spread."""

from __future__ import annotations

import sqlite3
from typing import Any

from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.retrieval.pagerank import personalized_pagerank
from mcp_servers.memory.retrieval.vector_search import top_k


def _hydrate(conn: sqlite3.Connection, kind: str, node_id: str) -> dict | None:
    """Load a node from its table, filtering out archived/invalidated/closed nodes."""
    if kind == "episode":
        r = conn.execute(
            "SELECT id, actor, summary, importance, occurred_at, status "
            "FROM episodes WHERE id = ? AND status != 'archived'",
            (node_id,),
        ).fetchone()
    elif kind == "fact":
        r = conn.execute(
            "SELECT id, subject_entity, predicate, object_entity, object_value, "
            "       valid_from, valid_to, learned_at, confidence "
            "FROM facts WHERE id = ? AND invalidated_at IS NULL",
            (node_id,),
        ).fetchone()
    elif kind == "reflection":
        r = conn.execute(
            "SELECT id, summary, level, importance, created_at "
            "FROM reflections WHERE id = ?", (node_id,),
        ).fetchone()
    elif kind == "entity":
        r = conn.execute(
            "SELECT id, canonical_name, kind, summary, importance "
            "FROM entities WHERE id = ?", (node_id,),
        ).fetchone()
    elif kind == "hypothesis":
        r = conn.execute(
            "SELECT id, statement, status, confidence "
            "FROM hypotheses WHERE id = ? AND status = 'open'", (node_id,),
        ).fetchone()
    else:
        return None
    if r is None:
        return None
    out = dict(r)
    out["node_kind"] = kind
    out["node_id"] = node_id
    return out


def recall(
    *,
    conn: sqlite3.Connection,
    query: str,
    embedder: EmbeddingProvider | None = None,
    max_results: int = 8,
    kinds: list[str] | None = None,
    include_hypotheses: bool = False,
    seed_top_k: int = 20,
    damping: float = 0.85,
) -> list[dict[str, Any]]:
    """HippoRAG-style retrieval: vector seed → personalized PageRank spread.

    Args:
        conn: Database connection.
        query: Text query to embed.
        embedder: EmbeddingProvider; defaults to SentenceTransformersProvider if None.
        max_results: Max results to return (default 8).
        kinds: Optional list of node kinds to seed on (default: all except hypotheses).
        include_hypotheses: If True, include hypotheses in seeding (default False).
        seed_top_k: Number of seed results from vector search (default 20).
        damping: PageRank damping factor (default 0.85).

    Returns:
        List of dicts with relevance scores; includes only non-archived/non-invalidated nodes.
    """
    if embedder is None:
        from mcp_servers.memory.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        embedder = SentenceTransformersProvider()

    seed_kinds = kinds
    if seed_kinds is None and not include_hypotheses:
        seed_kinds = ["episode", "fact", "reflection", "entity"]

    q_vec = embedder.embed(query)
    seeds = top_k(conn, query_vec=q_vec, k=seed_top_k, kinds=seed_kinds)

    if not seeds:
        return []

    pers = {(k, i): max(0.0, s) for (k, i, s) in seeds if s > 0.0}
    scores = personalized_pagerank(
        conn=conn, personalization=pers, damping=damping, max_iter=200,
    )

    if not scores:
        # graph is empty / nothing reachable — fall back to seed order
        ranked = [(k, i, s) for (k, i, s) in seeds]
    else:
        ranked = sorted(
            scores.items(), key=lambda kv: kv[1], reverse=True,
        )
        ranked = [(k, i, s) for ((k, i), s) in ranked]

    out: list[dict[str, Any]] = []
    for kind, nid, score in ranked:
        if not include_hypotheses and kind == "hypothesis":
            continue
        hydrated = _hydrate(conn, kind, nid)
        if hydrated is None:
            continue
        hydrated["relevance"] = float(score)
        out.append(hydrated)
        if len(out) >= max_results:
            break
    return out
