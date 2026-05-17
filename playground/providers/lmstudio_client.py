"""LM Studio provider — OpenAI-compatible local endpoint."""

from __future__ import annotations

import json
import os
import re
import secrets
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

# --- Prompted tool-calling support ---------------------------------------

# Models without a native vLLM tool-call parser (Gemma 4, etc.) need their
# tools injected into the system prompt and parsed out of the reply text.
# Format mirrors the Hermes / Nous-style <tool_call> XML tag, which is the
# most widely-supported open-source convention.

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

_PROMPT_TEMPLATE = """You have access to the following tools. To call a tool, emit a
block in this exact format on its own line(s):

<tool_call>
{{"name": "<tool_name>", "arguments": {{<json args>}}}}
</tool_call>

Rules:
- Emit a <tool_call> block ONLY when you want to call a tool.
- Always emit valid JSON inside the block.
- You may emit multiple <tool_call> blocks in one reply.
- Text outside <tool_call> blocks is shown to the user.

Available tools:

{tools}"""


def _format_tool_descriptions(tools: list[ToolDefinition]) -> str:
    lines: list[str] = []
    for t in tools:
        lines.append(f"- name: {t.name}")
        if t.description:
            lines.append(f"  description: {t.description}")
        lines.append(f"  parameters: {json.dumps(t.input_schema, separators=(',', ':'))}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _parse_prompted_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split a prompted-tool-call response into (plain_text, tool_calls).

    Plain text has all <tool_call> blocks stripped. Tool calls are
    successfully-parsed JSON payloads with at least a `name` key; malformed
    blocks are silently dropped.
    """
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        calls.append({
            "name": payload["name"],
            "arguments": payload.get("arguments") or {},
        })
    plain = _TOOL_CALL_RE.sub("", text).strip()
    return plain, calls


# --- Client --------------------------------------------------------------


class LMStudioClient(OpenAIClient):
    name = "lmstudio"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        # api_key required by SDK but ignored by LM Studio — placeholder is fine.
        super().__init__(model=model, api_key="lm-studio", base_url=url)

    def _should_use_prompted_tools(self) -> bool:
        override = os.getenv("LMSTUDIO_TOOL_PROMPTING", "auto").lower()
        if override == "prompted":
            return True
        if override == "native":
            return False
        # auto: model-name heuristic. Gemma has no native vLLM parser as of
        # mid-2026; other families do.
        return "gemma" in self.model.lower()

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        if not tools:
            # Text-only — full streaming via parent.
            yield from super().stream_chat(
                messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return

        if self._should_use_prompted_tools():
            yield from self._prompted_tools_chat(
                messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return

        yield from self._native_tools_chat(
            messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _native_tools_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        # vLLM's streaming tool-call parsers are buggy across many releases
        # (raises "Not being used, manual parsing in serving_chat.py"). Use
        # a non-streaming completion and synthesize events at the end.
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

    def _prompted_tools_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        # Build the augmented system prompt; drop tools from the API call.
        tool_section = _PROMPT_TEMPLATE.format(tools=_format_tool_descriptions(tools))
        augmented_system = f"{system}\n\n{tool_section}" if system else tool_section

        api_messages: list[dict] = [{"role": "system", "content": augmented_system}]
        api_messages.extend(_to_openai_messages(messages))

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        choice = resp.choices[0]
        raw_text = (choice.message.content or "") if getattr(choice, "message", None) else ""
        plain, calls = _parse_prompted_tool_calls(raw_text)

        if plain:
            yield TextDelta(text=plain)
        for c in calls:
            yield ToolCallComplete(
                id=f"tc_{secrets.token_hex(6)}",
                name=c["name"],
                input=c["arguments"],
            )

        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        )
        # When the model actually called tools, signal the chat loop to dispatch.
        stop_reason = "tool_use" if calls else (choice.finish_reason or "")
        yield MessageComplete(usage=usage, stop_reason=stop_reason)


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
