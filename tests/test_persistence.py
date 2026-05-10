"""Tests for conversation persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.persistence import (
    SCHEMA_VERSION,
    ConversationStore,
    ConversationSummary,
)


def _config(provider: str = "anthropic") -> dict:
    return {
        "provider": provider,
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "temperature": 1.0,
        "system_prompt": {"source": None, "text": "hi"},
        "tools": {"local": [], "mcp": [], "builtin": []},
        "mcp_servers_enabled": [],
    }


def test_new_conversation_writes_file_with_schema(tmp_conversations_root: Path) -> None:
    store = ConversationStore(tmp_conversations_root)
    conv = store.new("basic_chat", _config())
    conv.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                          "content": [{"type": "text", "text": "hi"}]})

    files = list((tmp_conversations_root / "basic_chat").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["page"] == "basic_chat"
    assert data["config"]["provider"] == "anthropic"
    assert len(data["messages"]) == 1


def test_round_trip_canonical_fixture(
    canonical_conversation: dict,
    tmp_conversations_root: Path,
) -> None:
    """Loading a v1 fixture and saving it produces semantically equal JSON."""
    page_dir = tmp_conversations_root / "basic_chat"
    page_dir.mkdir()
    fpath = page_dir / f"{canonical_conversation['id']}.json"
    fpath.write_text(json.dumps(canonical_conversation))

    store = ConversationStore(tmp_conversations_root)
    loaded = store.load(canonical_conversation["id"])
    assert loaded.data["messages"] == canonical_conversation["messages"]
    assert loaded.data["config"] == canonical_conversation["config"]


def test_list_returns_summaries_sorted_descending(tmp_conversations_root: Path) -> None:
    store = ConversationStore(tmp_conversations_root)
    a = store.new("basic_chat", _config())
    a.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                      "content": [{"type": "text", "text": "first"}]})
    b = store.new("basic_chat", _config(provider="openai"))
    b.append_message({"role": "user", "ts": "2026-01-02T00:00:00Z",
                      "content": [{"type": "text", "text": "second"}]})

    summaries = store.list("basic_chat")
    assert len(summaries) == 2
    assert summaries[0].started_at >= summaries[1].started_at
    assert all(isinstance(s, ConversationSummary) for s in summaries)
    assert any(s.first_user_message == "first" for s in summaries)
    assert any(s.first_user_message == "second" for s in summaries)


def test_atomic_write_does_not_leave_partial_file(
    tmp_conversations_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a write failing mid-flight: original file should be untouched
    if it existed, and no .tmp file should remain."""
    store = ConversationStore(tmp_conversations_root)
    conv = store.new("basic_chat", _config())
    conv.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                          "content": [{"type": "text", "text": "first"}]})
    page_dir = tmp_conversations_root / "basic_chat"

    # Force os.replace to fail
    import os
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        conv.append_message({"role": "user", "ts": "2026-01-01T00:00:01Z",
                              "content": [{"type": "text", "text": "second"}]})

    # Restore so cleanup works
    monkeypatch.setattr(os, "replace", real_replace)

    leftover_tmp = list(page_dir.glob("*.tmp"))
    assert leftover_tmp == [], f"Leftover tmp files: {leftover_tmp}"
