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
