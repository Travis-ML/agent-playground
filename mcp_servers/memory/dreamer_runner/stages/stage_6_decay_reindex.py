"""Stage ⑥ — archive low-importance nodes, recompute PageRank, refresh cache."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from mcp_servers.memory.dreamer_runner.decay import archive_bottom_percentile
from mcp_servers.memory.retrieval.pagerank import compute_and_store


def _refresh_background_pack_cache(conn: sqlite3.Connection) -> dict:
    # Top 8 entities by current PageRank score (those with kind='entity')
    rows = conn.execute(
        """
        SELECT p.node_id AS id, p.score AS score, e.canonical_name AS name,
               e.summary AS summary
        FROM pagerank_scores p
        LEFT JOIN entities e ON e.id = p.node_id
        WHERE p.node_kind = 'entity'
        ORDER BY p.score DESC
        LIMIT 8
        """
    ).fetchall()
    entities = [dict(r) for r in rows]

    refl_rows = conn.execute(
        "SELECT id, summary, level FROM reflections "
        "WHERE level >= 1 ORDER BY created_at DESC LIMIT 4"
    ).fetchall()
    reflections = [dict(r) for r in refl_rows]

    cache = {"entities": entities, "reflections": reflections}
    conn.execute(
        """
        INSERT INTO dreamer_config (key, value) VALUES ('background_pack_cache', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (json.dumps(cache),),
    )
    return cache


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    archive_percentile: float = 0.05,
    **_: Any,
) -> dict[str, Any]:
    archived = archive_bottom_percentile(
        conn=conn, percentile=archive_percentile,
    )
    pr_nodes = compute_and_store(conn=conn, dream_run_id=dream_run_id)
    cache = _refresh_background_pack_cache(conn)
    return {
        "metrics": {
            "archived": archived,
            "pagerank_nodes": pr_nodes,
            "background_pack_size": len(cache.get("entities", [])),
        }
    }
