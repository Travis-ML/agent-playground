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


# ---------------- OpenAI ----------------

def test_openai_client_basic_stream(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers import openai_client as oc

    events = []
    with (fixtures_dir / "openai_basic_response.jsonl").open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    class _FakeStream:
        def __init__(self, evs):
            self._evs = [type("Chunk", (), {"model_dump": lambda self, e=e: e})() for e in evs]
        def __iter__(self): return iter(self._evs)

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeStream(events)

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs): self.chat = _FakeChat()

    monkeypatch.setattr(oc, "OpenAI", _FakeOpenAI)

    client = oc.OpenAIClient(model="gpt-4o", api_key="test")
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
    assert completes[0].stop_reason == "stop"


# ---------------- LM Studio ----------------

def test_lmstudio_client_subclasses_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://example.invalid:1234/v1")
    from playground.providers.lmstudio_client import LMStudioClient
    from playground.providers.openai_client import OpenAIClient

    c = LMStudioClient(model="local-model")
    assert isinstance(c, OpenAIClient)
    assert c.name == "lmstudio"


def test_discover_lmstudio_models_returns_empty_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from playground.providers.lmstudio_client import discover_lmstudio_models

    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1/v1")  # unlikely to be live
    models = discover_lmstudio_models(timeout=0.1)
    assert models == []
