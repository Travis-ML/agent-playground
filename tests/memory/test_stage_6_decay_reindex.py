import sqlite3

from mcp_servers.memory.dreamer_runner.stages.stage_6_decay_reindex import run
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.links import add_link


def test_stage_6_runs_decay_and_pagerank(conn: sqlite3.Connection) -> None:
    for i in range(20):
        ep = insert_episode(
            conn, actor="user", predicate="x", subject_entity=None,
            object_entity=None, object_value=str(i), summary=f"e{i}",
            importance=i / 20.0,
            occurred_at=f"2026-04-{(i % 28) + 1:02d}T00:00:00Z",
            source_refs=[],
        )
        conn.execute("UPDATE episodes SET status = 'consolidated' WHERE id = ?", (ep.id,))
    add_link(conn, src_kind="episode", src_id="ep_a",
             dst_kind="episode", dst_id="ep_b",
             link_type="see_also", weight=1.0)

    out = run(conn=conn, dream_run_id="dr_x", ctx={})
    assert "archived" in out["metrics"]
    assert "pagerank_nodes" in out["metrics"]
    cfg = conn.execute(
        "SELECT value FROM dreamer_config WHERE key = 'background_pack_cache'"
    ).fetchone()
    assert cfg is not None
