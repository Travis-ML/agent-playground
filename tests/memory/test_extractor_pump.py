import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from mcp_servers.memory.extractor.pump import pump_once
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.raw_turns import record_turn


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_pump_once_processes_all_pending(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    page = tmp_path / "basic_chat"; page.mkdir(parents=True)
    (page / "c.json").write_text(json.dumps({
        "id": "c", "page": "basic_chat",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        ],
    }))
    record_turn(conn, conversation_id="c", turn_index=0, role="user",
                occurred_at="2026-05-12T15:00:01Z")
    record_turn(conn, conversation_id="c", turn_index=1, role="assistant",
                occurred_at="2026-05-12T15:00:02Z")

    llm = MagicMock()
    llm.stream_chat.side_effect = lambda **kw: _stream({"episodes": []})

    processed = pump_once(conn=conn, llm=llm, conversations_root=tmp_path)

    assert processed == 2
    rows = conn.execute(
        "SELECT extraction_status FROM raw_turn_refs"
    ).fetchall()
    assert all(r["extraction_status"] == "done" for r in rows)
