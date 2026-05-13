"""Typed weighted links — the glue layer for graph traversal."""

from __future__ import annotations

import sqlite3


def add_link(
    conn: sqlite3.Connection,
    *,
    src_kind: str, src_id: str,
    dst_kind: str, dst_id: str,
    link_type: str,
    weight: float = 1.0,
    dream_run: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO links
            (src_kind, src_id, dst_kind, dst_id, link_type, weight,
             created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (src_kind, src_id, dst_kind, dst_id, link_type, weight, dream_run),
    )


def list_links_from(
    conn: sqlite3.Connection, *, src_kind: str, src_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM links WHERE src_kind = ? AND src_id = ?",
        (src_kind, src_id),
    ).fetchall()


def list_links_to(
    conn: sqlite3.Connection, *, dst_kind: str, dst_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM links WHERE dst_kind = ? AND dst_id = ?",
        (dst_kind, dst_id),
    ).fetchall()


def all_links(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM links").fetchall()
