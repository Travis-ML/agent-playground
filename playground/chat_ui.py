"""Reusable Streamlit components for chat rendering."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import streamlit as st

from playground.providers.base import (
    ChatMessage,
    MessageComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallComplete,
    ToolResultBlock,
    ToolUseBlock,
)


def render_message(msg: ChatMessage) -> None:
    """Render a finalized assistant or user message in the transcript."""
    with st.chat_message(msg.role):
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text:
                    st.markdown(block.text)
            elif isinstance(block, ToolUseBlock):
                # Phase 7 expands this to a collapsible block
                st.caption(f"→ tool call: `{block.name}`")
            elif isinstance(block, ToolResultBlock):
                st.caption(f"← tool result for `{block.tool_use_id}`")


def render_text_stream(events: Iterator[StreamEvent]) -> tuple[str, Any]:
    """Drive st.write_stream from a TextDelta iterator. Returns (full_text, last_event_or_none).

    The last_event_or_none lets callers inspect MessageComplete.usage etc.
    """
    last_non_text: Any = None
    text_buf: list[str] = []

    def _gen() -> Iterator[str]:
        nonlocal last_non_text
        for ev in events:
            if isinstance(ev, TextDelta):
                text_buf.append(ev.text)
                yield ev.text
            else:
                last_non_text = ev

    st.write_stream(_gen())
    return "".join(text_buf), last_non_text


def render_tool_call_block(
    *,
    name: str,
    source: dict[str, str],
    input: dict[str, Any],
    result_text: str | None,
    duration_ms: int | None,
    is_error: bool,
) -> None:
    """Collapsible block showing a tool call's name, source, input, output."""
    src_label = source.get("kind", "?")
    if src_label == "mcp":
        src_label = f"mcp/{source.get('server', '?')}"
    head = f"⚙ {name} · {src_label}"
    if duration_ms is not None:
        head += f" · {duration_ms}ms"
    if is_error:
        head = "⚠ " + head
    with st.expander(head, expanded=False):
        st.markdown("**Input**")
        st.code(json.dumps(input, indent=2, ensure_ascii=False), language="json")
        if result_text is not None:
            st.markdown("**Result**")
            st.code(result_text, language="json")


def stream_assistant_turn(
    client_stream: Callable[[], Any],
    *,
    on_text: Callable[[str], None],
) -> tuple[str, list[ToolCallComplete], MessageComplete | None]:
    """Drive a single assistant streaming turn.

    Returns (final_text, tool_calls, message_complete).
    """
    buf: list[str] = []
    tool_calls: list[ToolCallComplete] = []
    final: MessageComplete | None = None
    for ev in client_stream():
        if isinstance(ev, TextDelta):
            buf.append(ev.text)
            on_text(ev.text)
        elif isinstance(ev, ToolCallComplete):
            tool_calls.append(ev)
        elif isinstance(ev, MessageComplete):
            final = ev
    return "".join(buf), tool_calls, final


def data_to_chat_messages(conv_data: dict) -> list[ChatMessage]:
    """Reconstruct ChatMessage list from a persisted conversation dict."""
    out: list[ChatMessage] = []
    for m in conv_data.get("messages", []):
        blocks: list = []
        for b in m.get("content", []):
            if b.get("type") == "text":
                blocks.append(TextBlock(type="text", text=b["text"]))
            elif b.get("type") == "tool_use":
                blocks.append(ToolUseBlock(
                    type="tool_use", id=b["id"], name=b["name"],
                    input=b.get("input", {}), source=b.get("source", {}),
                ))
            elif b.get("type") == "tool_result":
                blocks.append(ToolResultBlock(
                    type="tool_result", tool_use_id=b["tool_use_id"],
                    content=b.get("content", []), is_error=b.get("is_error", False),
                    duration_ms=b.get("duration_ms"),
                ))
        out.append(ChatMessage(role=m["role"], content=blocks))
    return out
