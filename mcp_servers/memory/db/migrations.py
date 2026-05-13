"""Forward-only SQL migration runner. Migrations live in ./migrations/NNN_*.sql."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(row[0])


def apply_migrations(conn: sqlite3.Connection) -> None:
    current = current_version(conn)
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(path.name.split("_", 1)[0])
        if version <= current:
            continue
        sql = path.read_text()
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )
