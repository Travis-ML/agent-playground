import json
from pathlib import Path

from mcp_servers.memory.repo.raw_turn_content import read_turn_text


def _seed_conv(root: Path, conv_id: str, turns: list[dict]) -> None:
    page_dir = root / "basic_chat"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{conv_id}.json").write_text(json.dumps({
        "schema_version": 1, "id": conv_id, "page": "basic_chat",
        "messages": turns,
    }))


def test_read_turn_text_joins_text_blocks(tmp_path: Path) -> None:
    _seed_conv(tmp_path, "c1", [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]},
    ])
    assert read_turn_text(tmp_path, "c1", 0) == "hello\nworld"


def test_read_turn_text_skips_non_text_blocks(tmp_path: Path) -> None:
    _seed_conv(tmp_path, "c1", [
        {"role": "assistant", "content": [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
        ]},
    ])
    assert read_turn_text(tmp_path, "c1", 0) == "thinking..."
