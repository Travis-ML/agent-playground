import sqlite3

from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.links import add_link
from mcp_servers.memory.retrieval.recall import recall
from mcp_servers.memory.retrieval.vector_search import upsert_embedding


def _seed(conn: sqlite3.Connection, fixed_embedder):
    ep1 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="mcp",
        summary="MCP pool eventloop death", importance=0.7,
        occurred_at="2026-05-12T15:00:00Z", source_refs=[],
    )
    ep2 = insert_episode(
        conn, actor="agent", predicate="x", subject_entity=None,
        object_entity=None, object_value="diag",
        summary="thread holds stale loop reference", importance=0.7,
        occurred_at="2026-05-12T15:00:01Z", source_refs=[],
    )
    upsert_embedding(conn, node_kind="episode", node_id=ep1.id,
                     embedding=fixed_embedder.embed("MCP pool eventloop death"))
    upsert_embedding(conn, node_kind="episode", node_id=ep2.id,
                     embedding=fixed_embedder.embed("thread holds stale loop reference"))
    add_link(conn, src_kind="episode", src_id=ep1.id,
             dst_kind="episode", dst_id=ep2.id,
             link_type="caused", weight=1.0)
    return ep1, ep2


def test_recall_returns_seed_results(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    ep1, _ = _seed(conn, fixed_embedder)
    out = recall(
        conn=conn, query="MCP pool eventloop death",
        embedder=fixed_embedder, max_results=2,
    )
    assert len(out) >= 1
    assert any(r["node_id"] == ep1.id for r in out)


def test_recall_spreads_via_pagerank(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    ep1, ep2 = _seed(conn, fixed_embedder)
    out = recall(
        conn=conn, query="MCP pool", embedder=fixed_embedder, max_results=4,
    )
    ids = {r["node_id"] for r in out}
    # PageRank-spread should pull in ep2 even though its text is different
    assert ep1.id in ids
    assert ep2.id in ids
