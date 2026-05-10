"""Anthropic provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic

from playground.providers.base import (
    ChatMessage,
    LLMClient,
    MessageComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        api_messages = [_to_anthropic_message(m) for m in messages]
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        usage = Usage(input_tokens=0, output_tokens=0)
        stop_reason = ""
        in_flight_tools: dict[int, dict[str, Any]] = {}  # block_index → {id, name, partial}

        with self._client.messages.stream(**kwargs) as stream:
            for ev in stream:
                etype = ev["type"] if isinstance(ev, dict) else getattr(ev, "type", None)
                ev_dict = ev if isinstance(ev, dict) else _to_dict(ev)

                if etype == "message_start":
                    u = ev_dict.get("message", {}).get("usage", {})
                    usage.input_tokens = u.get("input_tokens", 0)
                    usage.cache_read_tokens = u.get("cache_read_input_tokens", 0) or 0
                    usage.cache_creation_tokens = (
                        u.get("cache_creation_input_tokens", 0) or 0
                    )

                elif etype == "content_block_start":
                    block = ev_dict.get("content_block", {})
                    if block.get("type") == "tool_use":
                        in_flight_tools[ev_dict["index"]] = {
                            "id": block["id"],
                            "name": block["name"],
                            "partial": "",
                        }

                elif etype == "content_block_delta":
                    delta = ev_dict.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield TextDelta(text=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        idx = ev_dict["index"]
                        if idx in in_flight_tools:
                            in_flight_tools[idx]["partial"] += delta.get("partial_json", "")
                            yield ToolCallDelta(
                                id=in_flight_tools[idx]["id"],
                                name=in_flight_tools[idx]["name"],
                                partial_input_json=in_flight_tools[idx]["partial"],
                            )

                elif etype == "content_block_stop":
                    idx = ev_dict["index"]
                    if idx in in_flight_tools:
                        t = in_flight_tools.pop(idx)
                        try:
                            parsed = json.loads(t["partial"]) if t["partial"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallComplete(id=t["id"], name=t["name"], input=parsed)

                elif etype == "message_delta":
                    d = ev_dict.get("delta", {})
                    if d.get("stop_reason"):
                        stop_reason = d["stop_reason"]
                    u = ev_dict.get("usage", {})
                    if "output_tokens" in u:
                        usage.output_tokens = u["output_tokens"]

                elif etype == "message_stop":
                    pass

        yield MessageComplete(usage=usage, stop_reason=stop_reason)


def _to_anthropic_message(m: ChatMessage) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": [_block_to_dict(b) for b in m.content],
    }


def _block_to_dict(b: TextBlock | ToolUseBlock | ToolResultBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": b.is_error,
        }
    raise TypeError(f"Unknown block type: {type(b)}")


def _to_dict(ev: Any) -> dict[str, Any]:
    """Best-effort coerce a Pydantic-ish stream event to a plain dict."""
    if hasattr(ev, "model_dump"):
        return ev.model_dump()
    if hasattr(ev, "dict"):
        return ev.dict()
    return dict(ev) if hasattr(ev, "__iter__") else {}
