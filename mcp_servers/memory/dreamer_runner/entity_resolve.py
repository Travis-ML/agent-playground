"""Tiny helper used by stages 3+ to turn canonical name + kind into an entity id."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.entities import get_or_create


def resolve_entity(
    conn: sqlite3.Connection, *,
    canonical: str, kind: str, seen_at: str,
) -> str:
    return get_or_create(
        conn, canonical_name=canonical, kind=kind, seen_at=seen_at,
    ).id
