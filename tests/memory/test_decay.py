import sqlite3

from mcp_servers.memory.dreamer_runner.decay import archive_bottom_percentile
from mcp_servers.memory.repo.episodes import insert_episode


def test_archive_marks_bottom_percentile(conn: sqlite3.Connection) -> None:
    for i in range(20):
        ep = insert_episode(
            conn, actor="user", predicate="x", subject_entity=None,
            object_entity=None, object_value=str(i), summary=f"e{i}",
            importance=(i / 20.0),
            occurred_at=f"2026-04-{(i % 28) + 1:02d}T00:00:00Z",
            source_refs=[],
        )
        if i < 10:
            conn.execute("UPDATE episodes SET status = 'consolidated' WHERE id = ?", (ep.id,))
    n = archive_bottom_percentile(conn=conn, percentile=0.10)
    assert n >= 1
    archived = conn.execute(
        "SELECT COUNT(*) AS c FROM episodes WHERE status = 'archived'"
    ).fetchone()["c"]
    assert archived == n
