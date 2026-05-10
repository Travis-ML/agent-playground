"""Reusable Streamlit components for chat rendering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from playground.providers.base import (
    ChatMessage,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)


def render_message(msg: ChatMessage) -> None:
    """Render a finalized assistant or user message in the transcript."""
    avatar = "🧑" if msg.role == "user" else "◐"
    with st.chat_message(msg.role, avatar=avatar):
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
