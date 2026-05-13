"""Resolve raw turn text by reading the playground's conversations/*.json files."""

from __future__ import annotations

import json
from pathlib import Path


def read_turn_text(
    conversations_root: str | Path,
    conversation_id: str,
    turn_index: int,
    *,
    page: str = "basic_chat",
) -> str:
    path = Path(conversations_root) / page / f"{conversation_id}.json"
    data = json.loads(path.read_text())
    msg = data["messages"][turn_index]
    parts: list[str] = []
    for block in msg.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
