import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_5_recombine import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _seed(conn: sqlite3.Connection, n: int = 5):
    eps = []
    for i in range(n):
        e = insert_episode(
            conn, actor="user", predicate="x",
            subject_entity=None, object_entity=None, object_value=str(i),
            summary=f"event {i}", importance=0.5,
            occurred_at=f"2026-05-12T15:00:{i:02d}Z",
            source_refs=[],
        )
        eps.append(e)
    conn.execute("UPDATE episodes SET status = 'consolidated'")
    return eps


def test_recombine_writes_hypotheses_for_non_none(
    conn: sqlite3.Connection,
) -> None:
    _seed(conn, n=6)
    llm = MagicMock()
    llm.stream_chat.side_effect = [
        _stream({"statement": "event 0 and event 3 may share a cause",
                 "confidence": 0.4}),
        _stream({"statement": None, "confidence": 0.0}),
        _stream({"statement": "events relate to shared workflow",
                 "confidence": 0.5}),
    ]
    out = run(conn=conn, dream_run_id="dr_x", ctx={}, llm=llm,
              k_triplets=3, seed=42)
    rows = conn.execute(
        "SELECT statement, status FROM hypotheses ORDER BY created_at"
    ).fetchall()
    assert all(r["status"] == "open" for r in rows)
    assert out["metrics"]["hypotheses_added"] == len(rows)
    assert out["metrics"]["triplets_sampled"] == 3


def test_recombine_returns_zero_when_too_few_nodes(
    conn: sqlite3.Connection,
) -> None:
    _seed(conn, n=2)
    out = run(conn=conn, dream_run_id="dr_x", ctx={}, llm=MagicMock(),
              k_triplets=3, seed=42)
    assert out["metrics"]["triplets_sampled"] == 0
