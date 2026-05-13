import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.facts import insert_new_fact, get_by_id


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _ctx_with_cluster(conn) -> dict:
    e = insert_episode(conn, actor="user", predicate="x",
                       subject_entity=None, object_entity=None,
                       object_value="hi", summary="user uses python",
                       importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                       source_refs=[])
    return {
        "cluster_ids": [[e.id]],
        "episode_index": {e.id: e},
    }


def test_extract_creates_new_fact(conn: sqlite3.Connection) -> None:
    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python",
        "object_entity_kind": "concept",
        "confidence": 0.9,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_test", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_added"] == 1
    rows = conn.execute(
        "SELECT * FROM facts WHERE invalidated_at IS NULL AND valid_to IS NULL"
    ).fetchall()
    assert len(rows) == 1


def test_extract_supersedes_when_object_changes(conn: sqlite3.Connection) -> None:
    travis = get_or_create(conn, canonical_name="Travis", kind="person",
                           seen_at="2026-05-12T15:00:00Z").id
    py_old = get_or_create(conn, canonical_name="Python 3.13", kind="concept",
                           seen_at="2026-05-12T15:00:00Z").id
    insert_new_fact(
        conn, subject_entity=travis, predicate="uses",
        object_entity=py_old, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_a",
    )

    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python 3.14",
        "object_entity_kind": "concept",
        "confidence": 0.95,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_b", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_added"] == 1
    assert out["metrics"]["facts_superseded"] == 1
    current = conn.execute(
        """
        SELECT object_entity FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        """,
        (travis, "uses"),
    ).fetchone()
    assert current is not None  # there is one current belief


def test_extract_reinforces_matching_value(conn: sqlite3.Connection) -> None:
    travis = get_or_create(conn, canonical_name="Travis", kind="person",
                           seen_at="2026-05-12T15:00:00Z").id
    py = get_or_create(conn, canonical_name="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z").id
    f = insert_new_fact(
        conn, subject_entity=travis, predicate="uses",
        object_entity=py, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.7, created_in_dream_run="dr_a",
    )
    before = get_by_id(conn, f.id).confidence

    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python",
        "object_entity_kind": "concept",
        "confidence": 0.9,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_c", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_reinforced"] == 1
    after = get_by_id(conn, f.id).confidence
    assert after > before
