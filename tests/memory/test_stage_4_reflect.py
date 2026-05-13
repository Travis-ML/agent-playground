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


def test_run_recursive_pass_creates_level_2(conn: sqlite3.Connection, fixed_embedder) -> None:
    from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import (
        run_recursive_pass,
    )
    from mcp_servers.memory.repo.reflections import insert_reflection

    r1 = insert_reflection(conn, summary="user prefers brevity", importance=0.8,
                           level=1, source_kind="episode_cluster",
                           source_ids=["ep_a"], created_in_dream_run="dr_a")
    r2 = insert_reflection(conn, summary="user dislikes verbose explanations",
                           importance=0.8, level=1, source_kind="episode_cluster",
                           source_ids=["ep_b"], created_in_dream_run="dr_a")

    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "insight": "user has a strong overall preference for brevity",
        "importance": 0.9,
        "supporting_event_ids": [r1.id, r2.id],
    })

    n = run_recursive_pass(conn=conn, dream_run_id="dr_b",
                           input_level=1, llm=llm,
                           embedder=fixed_embedder,
                           distance_threshold=2.0)
    assert n >= 1
    rows = conn.execute(
        "SELECT level FROM reflections WHERE level = 2"
    ).fetchall()
    assert len(rows) >= 1
