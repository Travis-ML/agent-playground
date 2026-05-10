"""Echo example tool."""

from __future__ import annotations

from playground.tools import register_tool


@register_tool
def echo(text: str) -> str:
    """Echo the input text back unchanged."""
    return text
