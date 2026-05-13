import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import (
    run as run_extract,
)
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_no_two_currently_believed_facts_for_same_subject_predicate(
    conn: sqlite3.Connection,
) -> None:
    ep1 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="-", summary="travis uses python 3.13",
        importance=0.5, occurred_at="2026-04-01T00:00:00Z", source_refs=[],
    )
    ep2 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="-", summary="travis upgraded to 3.14",
        importance=0.5, occurred_at="2026-05-01T00:00:00Z", source_refs=[],
    )

    llm = MagicMock()
    llm.stream_chat.side_effect = [
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "uses",
            "object_kind": "entity", "object": "Python 3.13",
            "object_entity_kind": "concept", "confidence": 0.9,
            "valid_from_hint": "2026-04-01T00:00:00Z",
        }]}),
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "uses",
            "object_kind": "entity", "object": "Python 3.14",
            "object_entity_kind": "concept", "confidence": 0.95,
            "valid_from_hint": "2026-05-01T00:00:00Z",
        }]}),
    ]

    ctx1 = {"cluster_ids": [[ep1.id]], "episode_index": {ep1.id: ep1}}
    run_extract(conn=conn, dream_run_id="dr_a", ctx=ctx1, llm=llm)
    ctx2 = {"cluster_ids": [[ep2.id]], "episode_index": {ep2.id: ep2}}
    run_extract(conn=conn, dream_run_id="dr_b", ctx=ctx2, llm=llm)

    rows = conn.execute(
        """
        SELECT subject_entity, predicate, COUNT(*) AS c
        FROM facts
        WHERE valid_to IS NULL AND invalidated_at IS NULL
        GROUP BY subject_entity, predicate
        HAVING c > 1
        """
    ).fetchall()
    assert rows == []
