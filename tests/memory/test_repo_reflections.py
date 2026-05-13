import sqlite3

from mcp_servers.memory.repo.reflections import (
    insert_reflection,
    list_by_level,
    list_recent,
)


def test_insert_and_list_by_level(conn: sqlite3.Connection) -> None:
    r1 = insert_reflection(
        conn, summary="user prefers terse output",
        importance=0.8, level=1, source_kind="episode_cluster",
        source_ids=["ep_a", "ep_b"], created_in_dream_run="dr_1",
    )
    r2 = insert_reflection(
        conn, summary="prefers brevity in commit messages too",
        importance=0.7, level=2, source_kind="reflection_cluster",
        source_ids=[r1.id], created_in_dream_run="dr_1",
    )
    assert [r.id for r in list_by_level(conn, level=1)] == [r1.id]
    assert [r.id for r in list_by_level(conn, level=2)] == [r2.id]


def test_list_recent(conn: sqlite3.Connection) -> None:
    r1 = insert_reflection(
        conn, summary="first reflection",
        importance=0.5, level=1, source_kind="episode_cluster",
        source_ids=[], created_in_dream_run="dr_1",
    )
    r2 = insert_reflection(
        conn, summary="second reflection",
        importance=0.6, level=1, source_kind="episode_cluster",
        source_ids=[], created_in_dream_run="dr_1",
    )
    recent = list_recent(conn, min_level=1, limit=10)
    assert len(recent) == 2
    recent_ids = {r.id for r in recent}
    assert recent_ids == {r1.id, r2.id}
