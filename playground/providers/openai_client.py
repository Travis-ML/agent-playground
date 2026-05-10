"""OpenAI provider implementation. Also used by LMStudioClient via custom base_url."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

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


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_to_openai_messages(messages))

        api_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            or None
        )

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=api_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = Usage(input_tokens=0, output_tokens=0)
        stop_reason = ""
        # tool_call_id → {"id":..., "name":..., "args":""}
        in_flight: dict[str, dict[str, str]] = {}

        for chunk in stream:
            ev = _maybe_dump(chunk)
            choices = ev.get("choices", [])
            if choices:
                ch = choices[0]
                delta = ch.get("delta", {}) or {}
                if (text := delta.get("content")):
                    yield TextDelta(text=text)
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    key = str(idx)
                    if key not in in_flight:
                        in_flight[key] = {"id": tc.get("id", key), "name": "", "args": ""}
                    fn = tc.get("function", {}) or {}
                    if fn.get("name"):
                        in_flight[key]["name"] = fn["name"]
                    if fn.get("arguments"):
                        in_flight[key]["args"] += fn["arguments"]
                        yield ToolCallDelta(
                            id=in_flight[key]["id"],
                            name=in_flight[key]["name"],
                            partial_input_json=in_flight[key]["args"],
                        )
                if ch.get("finish_reason"):
                    stop_reason = ch["finish_reason"]
                    for t in in_flight.values():
                        try:
                            parsed = json.loads(t["args"]) if t["args"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallComplete(id=t["id"], name=t["name"], input=parsed)
                    in_flight.clear()

            if (u := ev.get("usage")):
                usage.input_tokens = u.get("prompt_tokens", 0)
                usage.output_tokens = u.get("completion_tokens", 0)

        yield MessageComplete(usage=usage, stop_reason=stop_reason)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            text_parts = [b.text for b in m.content if isinstance(b, TextBlock)]
            tool_results = [b for b in m.content if isinstance(b, ToolResultBlock)]
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": "".join(
                            c.get("text", "") for c in tr.content if c.get("type") == "text"
                        ),
                    }
                )
        else:  # assistant
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in m.content
                if isinstance(b, ToolUseBlock)
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
    return out


def _maybe_dump(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if isinstance(chunk, dict):
        return chunk
    return {}
