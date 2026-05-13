from pathlib import Path

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations, current_version


def _tables(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return {r[0] for r in rows}


def test_apply_migrations_creates_full_schema(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    apply_migrations(conn)
    tables = _tables(conn)
    for expected in (
        "schema_version", "raw_turn_refs", "entities", "episodes", "facts",
        "reflections", "hypotheses", "links", "embeddings",
        "pagerank_scores", "dream_runs", "dreamer_config", "dreamer_lock",
    ):
        assert expected in tables, expected
    assert current_version(conn) == 1
    conn.close()


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    apply_migrations(conn)
    apply_migrations(conn)
    assert current_version(conn) == 1
    conn.close()
