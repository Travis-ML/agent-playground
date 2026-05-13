"""LM Studio provider — OpenAI-compatible local endpoint."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

from playground.providers.base import (
    ChatMessage,
    MessageComplete,
    StreamEvent,
    TextDelta,
    ToolCallComplete,
    ToolDefinition,
    Usage,
)
from playground.providers.openai_client import OpenAIClient, _to_openai_messages

# Note: this client talks to any OpenAI-compatible local inference server
# at LMSTUDIO_BASE_URL. The "lmstudio" name is historical — point it at
# LM Studio, vLLM, llama.cpp's OpenAI server, or anything compatible.


class LMStudioClient(OpenAIClient):
    name = "lmstudio"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        # api_key required by SDK but ignored by LM Studio — placeholder is fine.
        super().__init__(model=model, api_key="lm-studio", base_url=url)

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        # vLLM's streaming tool-call parsers are buggy across many releases
        # (raises "Not being used, manual parsing in serving_chat.py"). When
        # tools are present, use a non-streaming completion and synthesize
        # events at the end. Text-only turns keep full streaming.
        if not tools:
            yield from super().stream_chat(
                messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return

        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_to_openai_messages(messages))
        api_tools = [
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

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=api_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        choice = resp.choices[0]
        msg = choice.message
        if getattr(msg, "content", None):
            yield TextDelta(text=msg.content)
        for tc in (msg.tool_calls or []):
            raw_args = tc.function.arguments or ""
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed = {}
            yield ToolCallComplete(id=tc.id, name=tc.function.name, input=parsed)

        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        )
        yield MessageComplete(usage=usage, stop_reason=choice.finish_reason or "")


def discover_lmstudio_models(base_url: str | None = None, timeout: float = 1.0) -> list[str]:
    """Hit /v1/models to discover what's loaded. Returns [] if unreachable."""
    url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    try:
        resp = httpx.get(f"{url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
