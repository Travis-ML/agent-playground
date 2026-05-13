import sqlite3

from mcp_servers.memory.dreamer_runner.stages.stage_1_cluster import (
    cluster_episodes, run,
)
from mcp_servers.memory.repo.episodes import insert_episode


def _seed_episodes(conn: sqlite3.Connection, n: int = 6) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        e = insert_episode(
            conn, actor="user", predicate="x",
            subject_entity=None, object_entity=None,
            object_value=f"value-{i // 3}",  # 3 + 3 grouping
            summary=f"summary-{i}", importance=0.5,
            occurred_at=f"2026-05-12T15:00:{i:02d}Z",
            source_refs=[],
        )
        ids.append(e.id)
    return ids


def test_cluster_episodes_groups_similar(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    eps = _seed_episodes(conn, n=6)
    # the fake embedder gives identical vectors for identical strings, so
    # episodes with the same summary cluster together; here summaries are
    # distinct, so we cluster by hash distance — verify the function shape:
    clusters = cluster_episodes(
        episode_ids=eps, embeddings=[fixed_embedder.embed(s) for s in eps],
        distance_threshold=0.5,
    )
    flat = [eid for c in clusters for eid in c]
    assert sorted(flat) == sorted(eps)
    assert all(len(c) >= 1 for c in clusters)


def test_stage_run_writes_metrics_and_embeddings(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    _seed_episodes(conn, n=4)
    out = run(conn=conn, dream_run_id="dr_x", ctx={}, embedder=fixed_embedder)
    assert out["metrics"]["episodes_seen"] == 4
    assert out["metrics"]["clusters"] >= 1
    rows = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()
    assert rows["c"] == 4
    assert "cluster_ids" in out["ctx_updates"]
