"""Shared LLM helpers for dream stages — JSON-mode calls + minimal parsing."""

from __future__ import annotations

import json

from mcp_servers.memory.providers.base import (
    ChatMessage, LLMClient, MessageComplete, TextBlock, TextDelta,
)


def _collect(events) -> str:
    out: list[str] = []
    for ev in events:
        if isinstance(ev, TextDelta):
            out.append(ev.text)
        elif isinstance(ev, MessageComplete):
            break
    return "".join(out)


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def call_json_llm(
    *,
    llm: LLMClient,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    events = llm.stream_chat(
        messages=[ChatMessage(role="user", content=[TextBlock(type="text", text=user)])],
        system=system, tools=[],
        max_tokens=max_tokens, temperature=temperature,
    )
    text = _strip_fences(_collect(events))
    return json.loads(text)
