import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _cluster(conn, importance: float):
    e = insert_episode(
        conn, actor="user", predicate="prefers",
        subject_entity=None, object_entity=None, object_value="brevity",
        summary="user prefers brevity", importance=importance,
        occurred_at="2026-05-12T15:00:00Z", source_refs=[],
    )
    return {"cluster_ids": [[e.id]], "episode_index": {e.id: e}}


def test_reflect_creates_level_1_above_threshold(
    conn: sqlite3.Connection,
) -> None:
    ctx = _cluster(conn, importance=0.9)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "insight": "user values terse output across contexts",
        "importance": 0.85,
        "supporting_event_ids": [list(ctx["episode_index"].keys())[0]],
    })
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 1
    rows = conn.execute("SELECT level, summary FROM reflections").fetchall()
    assert rows[0]["level"] == 1


def test_reflect_skips_low_importance_clusters(
    conn: sqlite3.Connection,
) -> None:
    ctx = _cluster(conn, importance=0.1)
    llm = MagicMock()
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 0
    llm.stream_chat.assert_not_called()


def test_reflect_skips_when_insight_is_null(conn: sqlite3.Connection) -> None:
    ctx = _cluster(conn, importance=0.9)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"insight": None, "importance": 0.0,
                                            "supporting_event_ids": []})
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 0
