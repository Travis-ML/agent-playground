import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_2_consolidate import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_consolidate_marks_duplicates_and_survivors(
    conn: sqlite3.Connection,
) -> None:
    a = insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                       object_entity=None, object_value="alpha", summary="A",
                       importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                       source_refs=[])
    b = insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                       object_entity=None, object_value="alpha-2", summary="A2",
                       importance=0.5, occurred_at="2026-05-12T15:00:01Z",
                       source_refs=[])
    c = insert_episode(conn, actor="user", predicate="y", subject_entity=None,
                       object_entity=None, object_value="beta", summary="B",
                       importance=0.5, occurred_at="2026-05-12T15:00:02Z",
                       source_refs=[])
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "groups": [{"survivor": a.id, "duplicates": [b.id]}],
    })

    ctx = {"cluster_ids": [[a.id, b.id, c.id]],
           "episode_index": {a.id: a, b.id: b, c.id: c}}
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm)

    statuses = dict(conn.execute(
        "SELECT id, status FROM episodes"
    ).fetchall())
    assert statuses[a.id] == "consolidated"
    assert statuses[c.id] == "consolidated"
    assert statuses[b.id] == "consolidated"  # duplicate, also marked
    # duplicates should be linked into survivor
    rows = conn.execute(
        "SELECT * FROM links WHERE link_type = 'consolidated_into'"
    ).fetchall()
    assert any(r["src_id"] == b.id and r["dst_id"] == a.id for r in rows)
    assert out["metrics"]["clusters_processed"] == 1
    assert out["metrics"]["duplicates_collapsed"] == 1
