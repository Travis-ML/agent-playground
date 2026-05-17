"""Tests for Gemma-style prompted tool calling in LMStudioClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from playground.providers.base import (
    ChatMessage,
    MessageComplete,
    TextBlock,
    TextDelta,
    ToolCallComplete,
    ToolDefinition,
)
from playground.providers.lmstudio_client import (
    LMStudioClient,
    _format_tool_descriptions,
    _parse_prompted_tool_calls,
)


@pytest.fixture(autouse=True)
def _base_url(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://x/v1")
    monkeypatch.delenv("LMSTUDIO_TOOL_PROMPTING", raising=False)


# ---- _should_use_prompted_tools --------------------------------------------


def test_should_use_prompted_for_gemma_model() -> None:
    c = LMStudioClient(model="google/gemma-4-27b-it")
    assert c._should_use_prompted_tools() is True


def test_should_not_use_prompted_for_llama_model() -> None:
    c = LMStudioClient(model="meta/llama-3-70b-instruct")
    assert c._should_use_prompted_tools() is False


def test_env_override_force_prompted(monkeypatch) -> None:
    monkeypatch.setenv("LMSTUDIO_TOOL_PROMPTING", "prompted")
    c = LMStudioClient(model="meta/llama-3-70b-instruct")
    assert c._should_use_prompted_tools() is True


def test_env_override_force_native(monkeypatch) -> None:
    monkeypatch.setenv("LMSTUDIO_TOOL_PROMPTING", "native")
    c = LMStudioClient(model="google/gemma-4-27b-it")
    assert c._should_use_prompted_tools() is False


def test_env_override_auto_falls_back_to_model_check(monkeypatch) -> None:
    monkeypatch.setenv("LMSTUDIO_TOOL_PROMPTING", "auto")
    assert LMStudioClient(model="gemma-4-9b").  _should_use_prompted_tools() is True
    assert LMStudioClient(model="hermes-3-70b")._should_use_prompted_tools() is False


# ---- _format_tool_descriptions ---------------------------------------------


def test_format_tool_descriptions_includes_name_desc_schema() -> None:
    tools = [
        ToolDefinition(
            name="echo",
            description="Echo back",
            input_schema={
                "type": "object",
                "properties": {"s": {"type": "string"}},
                "required": ["s"],
            },
        ),
    ]
    out = _format_tool_descriptions(tools)
    assert "name: echo" in out
    assert "description: Echo back" in out
    assert '"s"' in out and '"required"' in out


def test_format_tool_descriptions_handles_multiple_tools() -> None:
    tools = [
        ToolDefinition(name="a", description="A", input_schema={"type": "object"}),
        ToolDefinition(name="b", description="",  input_schema={"type": "object"}),
    ]
    out = _format_tool_descriptions(tools)
    assert "name: a" in out and "name: b" in out
    # tool b has no description: that line should not appear for b
    b_block = out.split("- name: b")[1]
    assert "description:" not in b_block


# ---- _parse_prompted_tool_calls --------------------------------------------


def test_parse_extracts_blocks_and_text() -> None:
    text = (
        "Let me think.\n"
        '<tool_call>{"name": "echo", "arguments": {"s": "hi"}}</tool_call>\n'
        "And here is more text.\n"
        '<tool_call>{"name": "reverse", "arguments": {"s": "abc"}}</tool_call>\n'
    )
    plain, calls = _parse_prompted_tool_calls(text)
    assert "Let me think" in plain
    assert "And here is more text" in plain
    assert "<tool_call>" not in plain
    assert calls == [
        {"name": "echo",    "arguments": {"s": "hi"}},
        {"name": "reverse", "arguments": {"s": "abc"}},
    ]


def test_parse_skips_invalid_json_block() -> None:
    text = "Hi\n<tool_call>not json</tool_call>\nBye"
    plain, calls = _parse_prompted_tool_calls(text)
    assert "Hi" in plain and "Bye" in plain
    assert calls == []


def test_parse_skips_block_without_name_key() -> None:
    text = '<tool_call>{"args": {"x": 1}}</tool_call>'
    _, calls = _parse_prompted_tool_calls(text)
    assert calls == []


def test_parse_no_blocks_returns_text_unchanged() -> None:
    plain, calls = _parse_prompted_tool_calls("just plain text")
    assert plain == "just plain text"
    assert calls == []


def test_parse_handles_missing_arguments_key() -> None:
    text = '<tool_call>{"name": "ping"}</tool_call>'
    plain, calls = _parse_prompted_tool_calls(text)
    assert calls == [{"name": "ping", "arguments": {}}]


# ---- stream_chat end-to-end with prompted tools ----------------------------


def _fake_completion(content: str, *, finish: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def test_prompted_stream_chat_emits_text_then_tool_calls() -> None:
    c = LMStudioClient(model="google/gemma-4-27b-it")
    c._client.chat.completions.create = MagicMock(  # type: ignore[assignment]
        return_value=_fake_completion(
            "Thinking...\n"
            '<tool_call>{"name": "echo", "arguments": {"s": "hi"}}</tool_call>',
        )
    )

    events = list(c.stream_chat(
        [ChatMessage(role="user", content=[TextBlock(type="text", text="please echo hi")])],
        system="be helpful",
        tools=[ToolDefinition(name="echo", description="Echo",
                              input_schema={"type": "object"})],
        max_tokens=512,
        temperature=0.0,
    ))

    texts = [e for e in events if isinstance(e, TextDelta)]
    tool_calls = [e for e in events if isinstance(e, ToolCallComplete)]
    finals = [e for e in events if isinstance(e, MessageComplete)]

    assert any("Thinking" in t.text for t in texts)
    assert "<tool_call>" not in texts[0].text
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "echo"
    assert tool_calls[0].input == {"s": "hi"}
    assert tool_calls[0].id.startswith("tc_")
    assert len(finals) == 1
    assert finals[0].stop_reason == "tool_use"


def test_prompted_stream_chat_strips_tools_from_api_call() -> None:
    c = LMStudioClient(model="google/gemma-4-27b-it")
    c._client.chat.completions.create = MagicMock(  # type: ignore[assignment]
        return_value=_fake_completion("hello"),
    )

    list(c.stream_chat(
        [ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
        system="hi",
        tools=[ToolDefinition(name="echo", description="x",
                              input_schema={"type": "object"})],
        max_tokens=10, temperature=0.0,
    ))

    kwargs = c._client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert kwargs["stream"] is False
    # System prompt must be augmented with the tool-call instructions
    system_msg = next(m for m in kwargs["messages"] if m["role"] == "system")
    assert "<tool_call>" in system_msg["content"]
    assert "echo" in system_msg["content"]
    # Original system prompt should be preserved
    assert "hi\n\n" in system_msg["content"]


def test_prompted_stream_chat_no_tool_call_in_response() -> None:
    c = LMStudioClient(model="google/gemma-4-27b-it")
    c._client.chat.completions.create = MagicMock(  # type: ignore[assignment]
        return_value=_fake_completion("I don't need any tools for this."),
    )

    events = list(c.stream_chat(
        [ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
        system=None,
        tools=[ToolDefinition(name="echo", description="",
                              input_schema={"type": "object"})],
        max_tokens=10, temperature=0.0,
    ))

    assert [e for e in events if isinstance(e, ToolCallComplete)] == []
    final = next(e for e in events if isinstance(e, MessageComplete))
    assert final.stop_reason == "stop"


def test_non_gemma_with_tools_takes_native_path() -> None:
    c = LMStudioClient(model="meta/llama-3-70b-instruct")
    create = MagicMock(return_value=_fake_completion("text only"))
    c._client.chat.completions.create = create  # type: ignore[assignment]

    list(c.stream_chat(
        [ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
        system=None,
        tools=[ToolDefinition(name="echo", description="",
                              input_schema={"type": "object"})],
        max_tokens=10, temperature=0.0,
    ))

    # Native path keeps tools in the API call
    kwargs = create.call_args.kwargs
    assert "tools" in kwargs and len(kwargs["tools"]) == 1
    assert kwargs["stream"] is False


def test_no_tools_passes_through_to_streaming_parent(monkeypatch) -> None:
    c = LMStudioClient(model="google/gemma-4-27b-it")
    called = {"flag": False}

    def fake_super_stream(*args, **kwargs):
        called["flag"] = True
        yield MessageComplete(usage=MagicMock(input_tokens=0, output_tokens=0,
                                              cache_read_tokens=0, cache_creation_tokens=0),
                              stop_reason="stop")

    # Patch the parent class's stream_chat (the unbound function)
    from playground.providers import openai_client
    monkeypatch.setattr(openai_client.OpenAIClient, "stream_chat", fake_super_stream)

    list(c.stream_chat(
        [ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
        system=None, tools=[], max_tokens=10, temperature=0.0,
    ))
    assert called["flag"] is True
