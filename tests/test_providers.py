"""Tests for provider client implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.providers.base import (
    ChatMessage,
    MessageComplete,
    TextBlock,
    TextDelta,
)

# ---------------- Anthropic ----------------


def _replay_anthropic_stream(events_file: Path):
    """Yield event dicts as the anthropic SDK's stream iterator would yield
    raw events. The AnthropicClient adapter translates these into our
    StreamEvent types."""
    with events_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def test_anthropic_client_basic_stream(
    fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from playground.providers import anthropic_client as ac

    events = list(_replay_anthropic_stream(fixtures_dir / "anthropic_basic_response.jsonl"))

    class _FakeStream:
        def __init__(self, evs):
            self._evs = evs

        def __enter__(self):
            return iter(self._evs)

        def __exit__(self, *a):
            pass

    class _FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream(events)

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()

    monkeypatch.setattr(ac, "Anthropic", _FakeAnthropic)

    client = ac.AnthropicClient(model="claude-sonnet-4-6", api_key="test")
    out = list(
        client.stream_chat(
            messages=[ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=1.0,
        )
    )

    deltas = [e for e in out if isinstance(e, TextDelta)]
    completes = [e for e in out if isinstance(e, MessageComplete)]
    assert "".join(d.text for d in deltas) == "Hello there"
    assert len(completes) == 1
    assert completes[0].usage.output_tokens == 7
    assert completes[0].stop_reason == "end_turn"
