import sqlite3

from mcp_servers.memory.repo.episodes import (
    insert_episode,
    list_by_status,
    set_status,
)


def test_insert_and_list_fresh(conn: sqlite3.Connection) -> None:
    ep = insert_episode(
        conn,
        actor="user",
        predicate="reported_problem",
        subject_entity=None,
        object_entity=None,
        object_value="mcp pool eventloop death",
        summary="user reports the MCP pool keeps dying",
        importance=0.7,
        occurred_at="2026-05-12T15:00:01Z",
        source_refs=[{"raw_turn_ref_id": "rt_abc"}],
    )
    assert ep.status == "fresh"
    out = list_by_status(conn, "fresh")
    assert [o.id for o in out] == [ep.id]


def test_set_status_consolidates(conn: sqlite3.Connection) -> None:
    ep = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="foo", summary="s",
        importance=0.1, occurred_at="2026-05-12T15:00:01Z",
        source_refs=[],
    )
    set_status(conn, ep.id, "consolidated")
    assert list_by_status(conn, "fresh") == []
    assert [e.id for e in list_by_status(conn, "consolidated")] == [ep.id]
