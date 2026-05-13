from pathlib import Path

import pytest

from mcp_servers.memory.db.connection import open_connection


def test_open_connection_enables_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    conn = open_connection(db_path)
    cur = conn.execute("PRAGMA journal_mode")
    assert cur.fetchone()[0] == "wal"
    conn.close()


def test_open_connection_loads_sqlite_vec(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    cur = conn.execute("SELECT vec_version()")
    version = cur.fetchone()[0]
    assert isinstance(version, str) and len(version) > 0
    conn.close()


def test_open_connection_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "memory.db"
    conn = open_connection(nested)
    assert nested.parent.is_dir()
    conn.close()
