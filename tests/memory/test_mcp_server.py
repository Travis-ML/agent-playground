import sqlite3

from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.facts import insert_new_fact
from mcp_servers.memory.server import (
    handle_get_entity,
    handle_record_turn,
    handle_search_episodes,
    handle_search_facts,
)


def test_handle_record_turn_writes_row(conn: sqlite3.Connection) -> None:
    out = handle_record_turn(
        conn=conn, conversation_id="c1", turn_index=0,
        role="user", occurred_at="2026-05-12T15:00:01Z",
    )
    assert out["status"] == "ok"
    assert out["raw_turn_id"].startswith("rt_")


def test_handle_search_episodes_filters_by_actor(conn: sqlite3.Connection) -> None:
    insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                   object_entity=None, object_value="alpha", summary="a",
                   importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                   source_refs=[])
    insert_episode(conn, actor="agent", predicate="x", subject_entity=None,
                   object_entity=None, object_value="beta", summary="b",
                   importance=0.5, occurred_at="2026-05-12T15:00:01Z",
                   source_refs=[])
    out = handle_search_episodes(conn=conn, actor="user", limit=10)
    assert [e["summary"] for e in out["episodes"]] == ["a"]


def test_handle_search_facts_default_returns_only_current(
    conn: sqlite3.Connection,
) -> None:
    user = get_or_create(conn, canonical_name="U", kind="person",
                        seen_at="2026-05-12T15:00:00Z").id
    insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=None, object_value="python",
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_x",
    )
    out = handle_search_facts(conn=conn, subject_canonical_name="U")
    assert len(out["facts"]) == 1
    assert out["facts"][0]["object_value"] == "python"


def test_handle_get_entity_returns_dossier(conn: sqlite3.Connection) -> None:
    user = get_or_create(
        conn, canonical_name="MCP pool", kind="concept",
        seen_at="2026-05-12T15:00:00Z",
    )
    out = handle_get_entity(conn=conn, name="MCP pool")
    assert out["entity"]["id"] == user.id
    assert out["entity"]["canonical_name"] == "MCP pool"
    assert isinstance(out["recent_facts"], list)
    assert isinstance(out["recent_episodes"], list)


def test_handle_status_reports_counts(conn: sqlite3.Connection) -> None:
    from mcp_servers.memory.server import handle_status

    out = handle_status(conn=conn)
    assert out["counts"]["raw_turn_refs"] == 0
    assert out["counts"]["episodes"] == 0
    assert "last_dream_run" in out
    assert out["last_dream_run"] is None
