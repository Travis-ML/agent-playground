"""LLMClient protocol and shared types.

The protocol is intentionally narrow: stream a chat response with optional
tools, yield typed events the UI can consume without branching on provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# --- Message types (Anthropic-native shape) ----------------------------

@dataclass
class TextBlock:
    type: Literal["text"]
    text: str


@dataclass
class ToolUseBlock:
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    source: dict[str, str] = field(default_factory=dict)  # {"kind": "local"|"mcp"|"builtin", ...}


@dataclass
class ToolResultBlock:
    type: Literal["tool_result"]
    tool_use_id: str
    content: list[dict[str, Any]]   # always list of {"type": "text", "text": ...}
    is_error: bool = False
    duration_ms: int | None = None


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


# --- Tool definitions passed to providers -------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]   # JSON Schema


# --- Stream events the UI consumes -------------------------------------

@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    """Provider has started emitting a tool call. Multiple deltas may arrive
    before ToolCallComplete; UI accumulates them. For v1 we render once
    complete, so deltas can simply update an in-flight buffer."""
    id: str
    name: str
    partial_input_json: str   # may be empty until the final delta


@dataclass
class ToolCallComplete:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class MessageComplete:
    usage: Usage
    stop_reason: str = ""   # "end_turn" | "tool_use" | "max_tokens" | provider-specific


StreamEvent = TextDelta | ToolCallDelta | ToolCallComplete | MessageComplete


# --- The protocol -------------------------------------------------------

class LLMClient(Protocol):
    """All three provider implementations satisfy this protocol."""

    name: str
    model: str

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        """Yield events as the model streams a response."""
        ...
