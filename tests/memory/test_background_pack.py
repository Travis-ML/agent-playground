import json
import sqlite3

from mcp_servers.memory.retrieval.background_pack import build_pack


def test_build_pack_uses_cached_snapshot_when_present(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "INSERT INTO dreamer_config (key, value) VALUES ('background_pack_cache', ?)",
        (json.dumps({
            "entities": [
                {"id": "en_1", "name": "MCP pool",
                 "summary": "the MCP server pool",
                 "score": 0.13},
            ],
            "reflections": [
                {"id": "re_1", "summary": "user prefers brevity",
                 "level": 1},
            ],
        }),),
    )
    md = build_pack(conn=conn)
    assert "MCP pool" in md
    assert "user prefers brevity" in md


def test_build_pack_empty_when_no_cache_and_no_data(
    conn: sqlite3.Connection,
) -> None:
    md = build_pack(conn=conn)
    assert "no prior memory" in md.lower()
