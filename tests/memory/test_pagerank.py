import sqlite3

from mcp_servers.memory.retrieval.pagerank import compute_and_store
from mcp_servers.memory.repo.links import add_link


def test_compute_pagerank_writes_scores_for_all_nodes(
    conn: sqlite3.Connection,
) -> None:
    # build a tiny graph: A -> B -> C, A -> C
    add_link(conn, src_kind="entity", src_id="A",
             dst_kind="entity", dst_id="B", link_type="see_also", weight=1.0)
    add_link(conn, src_kind="entity", src_id="B",
             dst_kind="entity", dst_id="C", link_type="see_also", weight=1.0)
    add_link(conn, src_kind="entity", src_id="A",
             dst_kind="entity", dst_id="C", link_type="see_also", weight=1.0)

    n = compute_and_store(conn=conn, dream_run_id="dr_x")
    assert n == 3
    rows = conn.execute(
        "SELECT node_id, score FROM pagerank_scores"
    ).fetchall()
    scores = {r["node_id"]: r["score"] for r in rows}
    # C has more inbound than A or B
    assert scores["C"] >= scores["A"]
    assert scores["C"] >= scores["B"]
