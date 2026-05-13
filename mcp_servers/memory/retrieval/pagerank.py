"""Personalized PageRank over the typed weighted link graph."""

from __future__ import annotations

import sqlite3

import networkx as nx

from mcp_servers.memory.repo.links import all_links


def _build_graph(conn: sqlite3.Connection) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in all_links(conn):
        src = f"{row['src_kind']}/{row['src_id']}"
        dst = f"{row['dst_kind']}/{row['dst_id']}"
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += row["weight"]
        else:
            g.add_edge(src, dst, weight=row["weight"])
    return g


def compute_and_store(
    *, conn: sqlite3.Connection, dream_run_id: str, damping: float = 0.85,
) -> int:
    g = _build_graph(conn)
    if g.number_of_nodes() == 0:
        return 0
    scores = nx.pagerank(g, alpha=damping, max_iter=200, weight="weight")
    rows = []
    for node, score in scores.items():
        kind, nid = node.split("/", 1)
        rows.append((kind, nid, float(score), dream_run_id))
    conn.execute("DELETE FROM pagerank_scores")
    conn.executemany(
        "INSERT INTO pagerank_scores (node_kind, node_id, score, computed_in_dream_run) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def personalized_pagerank(
    *,
    conn: sqlite3.Connection,
    personalization: dict[tuple[str, str], float],
    damping: float = 0.85,
    max_iter: int = 50,
) -> dict[tuple[str, str], float]:
    g = _build_graph(conn)
    if g.number_of_nodes() == 0:
        return {}
    p = {f"{k}/{i}": w for (k, i), w in personalization.items() if f"{k}/{i}" in g}
    if not p:
        return {}
    total = sum(p.values()) or 1.0
    p = {k: v / total for k, v in p.items()}
    scores = nx.pagerank(
        g, alpha=damping, personalization=p,
        max_iter=max_iter, weight="weight",
    )
    return {
        (n.split("/", 1)[0], n.split("/", 1)[1]): float(s)
        for n, s in scores.items()
    }
