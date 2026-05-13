import sqlite3

from mcp_servers.memory.retrieval.vector_search import (
    has_embedding,
    upsert_embedding,
)


def test_upsert_then_check_returns_true(conn: sqlite3.Connection) -> None:
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.1] * 768)
    assert has_embedding(conn, "episode", "ep_1") is True
    assert has_embedding(conn, "episode", "ep_999") is False


def test_upsert_replaces_existing(conn: sqlite3.Connection) -> None:
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.1] * 768)
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.2] * 768)
    # only one row
    rows = conn.execute(
        "SELECT COUNT(*) AS c FROM embeddings WHERE node_id = 'ep_1'"
    ).fetchone()
    assert rows["c"] == 1
