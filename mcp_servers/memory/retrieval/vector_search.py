"""sqlite-vec backed embedding storage + search."""

from __future__ import annotations

import sqlite3
import struct


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def upsert_embedding(
    conn: sqlite3.Connection, *,
    node_kind: str, node_id: str, embedding: list[float],
) -> None:
    conn.execute(
        "DELETE FROM embeddings WHERE node_kind = ? AND node_id = ?",
        (node_kind, node_id),
    )
    conn.execute(
        "INSERT INTO embeddings (node_kind, node_id, embedding) VALUES (?, ?, ?)",
        (node_kind, node_id, _pack(embedding)),
    )


def has_embedding(conn: sqlite3.Connection, node_kind: str, node_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM embeddings WHERE node_kind = ? AND node_id = ?",
        (node_kind, node_id),
    ).fetchone()
    return row is not None


def top_k(
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    k: int = 20,
    kinds: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Return [(node_kind, node_id, similarity)] ordered by cosine similarity.

    Uses sqlite-vec's MATCH operator for vector search.
    Similarity = 1.0 - distance (lower distance = higher similarity).
    """
    if kinds:
        marks = ",".join("?" * len(kinds))
        sql = (
            f"SELECT node_kind, node_id, distance FROM embeddings "
            f"WHERE embedding MATCH ? AND node_kind IN ({marks}) "
            f"ORDER BY distance LIMIT ?"
        )
        params = [_pack(query_vec), *kinds, k]
    else:
        sql = (
            "SELECT node_kind, node_id, distance FROM embeddings "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
        )
        params = [_pack(query_vec), k]
    rows = conn.execute(sql, params).fetchall()
    # sqlite-vec returns distance (lower = closer); convert to similarity.
    return [(r["node_kind"], r["node_id"], 1.0 - float(r["distance"])) for r in rows]
