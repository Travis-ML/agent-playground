import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

from mcp_servers.memory.extractor.worker import extract_for_turn
from mcp_servers.memory.providers.base import (  # noqa: F401
    MessageComplete,
    TextDelta,
    Usage,
)
from mcp_servers.memory.repo.episodes import list_by_status
from mcp_servers.memory.repo.raw_turns import record_turn


def _make_fake_stream(payload: dict) -> Iterator:
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=10, output_tokens=10), stop_reason="end_turn")


def test_extract_for_turn_writes_episodes(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # seed conversation file
    page = tmp_path / "basic_chat"
    page.mkdir(parents=True)
    (page / "c1.json").write_text(json.dumps({
        "id": "c1", "page": "basic_chat",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "MCP pool keeps dying"}]},
        ],
    }))
    rt = record_turn(
        conn, conversation_id="c1", turn_index=0, role="user",
        occurred_at="2026-05-12T15:00:01Z",
    )

    fake = MagicMock()
    fake.stream_chat.return_value = _make_fake_stream({
        "episodes": [{
            "actor": "user", "predicate": "reported_problem",
            "subject": None, "object": "MCP pool eventloop death",
            "summary": "user reports MCP pool keeps dying",
            "importance": 0.8,
        }],
    })

    extract_for_turn(
        conn=conn,
        llm=fake,
        conversations_root=tmp_path,
        raw_turn_id=rt.id,
    )

    eps = list_by_status(conn, "fresh")
    assert len(eps) == 1
    assert eps[0].summary == "user reports MCP pool keeps dying"
    row = conn.execute(
        "SELECT extraction_status FROM raw_turn_refs WHERE id = ?", (rt.id,)
    ).fetchone()
    assert row["extraction_status"] == "done"
