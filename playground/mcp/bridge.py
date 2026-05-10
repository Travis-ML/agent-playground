"""MCP tool schema ↔ provider tool format conversion."""

from __future__ import annotations

from typing import Any

from playground.providers.base import ToolDefinition


def mcp_tool_to_provider_format(td: ToolDefinition, provider: str) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "name": td.name,
            "description": td.description,
            "input_schema": td.input_schema,
        }
    if provider in ("openai", "lmstudio"):
        return {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.input_schema,
            },
        }
    raise ValueError(f"Unsupported provider for tool conversion: {provider!r}")


def mcp_tools_to_provider_format(tds: list[ToolDefinition], provider: str) -> list[dict[str, Any]]:
    return [mcp_tool_to_provider_format(t, provider) for t in tds]
