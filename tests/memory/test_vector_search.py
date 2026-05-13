import sqlite3

from mcp_servers.memory.retrieval.vector_search import (
    has_embedding,
    top_k,
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


def test_top_k_returns_ordered_by_similarity(conn: sqlite3.Connection) -> None:
    # Two nodes, one similar to query and one far.
    upsert_embedding(conn, node_kind="episode", node_id="ep_close",
                     embedding=[1.0] + [0.0] * 767)
    upsert_embedding(conn, node_kind="episode", node_id="ep_far",
                     embedding=[0.0] + [1.0] + [0.0] * 766)
    out = top_k(conn, query_vec=[1.0] + [0.0] * 767, k=2)
    assert out[0][1] == "ep_close"
    assert out[1][1] == "ep_far"
    assert out[0][2] >= out[1][2]


def test_top_k_can_filter_by_kinds(conn: sqlite3.Connection) -> None:
    upsert_embedding(conn, node_kind="episode",    node_id="ep_1",
                     embedding=[1.0] + [0.0] * 767)
    upsert_embedding(conn, node_kind="reflection", node_id="re_1",
                     embedding=[1.0] + [0.0] * 767)
    out = top_k(conn, query_vec=[1.0] + [0.0] * 767, k=5, kinds=["episode"])
    assert [n[0] for n in out] == ["episode"]
