"""Tests for the MCP ↔ provider tool-format bridge."""

from playground.mcp.bridge import mcp_tool_to_provider_format
from playground.providers.base import ToolDefinition


def test_anthropic_format_passthrough():
    td = ToolDefinition(
        name="save_note",
        description="Save a note.",
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    out = mcp_tool_to_provider_format(td, provider="anthropic")
    assert out == {
        "name": "save_note",
        "description": "Save a note.",
        "input_schema": td.input_schema,
    }


def test_openai_format_wraps_in_function_envelope():
    td = ToolDefinition(
        name="save_note",
        description="Save a note.",
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    out = mcp_tool_to_provider_format(td, provider="openai")
    assert out == {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note.",
            "parameters": td.input_schema,
        },
    }
