"""Tests for the local tool registry."""

from __future__ import annotations

import pytest


def test_register_tool_infers_schema_from_signature_and_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from playground.tools import _RESET_FOR_TESTS, get_local_tools, register_tool

    _RESET_FOR_TESTS()

    @register_tool
    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: first integer
            b: second integer
        """
        return a + b

    tools = get_local_tools()
    assert any(t.name == "add" for t in tools)
    add_tool = next(t for t in tools if t.name == "add")
    assert add_tool.description.startswith("Add two integers")
    assert add_tool.input_schema["type"] == "object"
    assert "a" in add_tool.input_schema["properties"]
    assert "b" in add_tool.input_schema["properties"]
    assert sorted(add_tool.input_schema["required"]) == ["a", "b"]


def test_call_local_tool_dispatches_to_registered_function() -> None:
    from playground.tools import _RESET_FOR_TESTS, call_local_tool, register_tool

    _RESET_FOR_TESTS()

    @register_tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hi {name}"

    result = call_local_tool("greet", {"name": "world"})
    assert result == "hi world"


def test_unknown_local_tool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.tools import _RESET_FOR_TESTS, call_local_tool

    _RESET_FOR_TESTS()
    with pytest.raises(KeyError):
        call_local_tool("does_not_exist", {})
