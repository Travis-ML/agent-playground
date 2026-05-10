"""Local tool registry — @register_tool decorator + schema inference."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from playground.providers.base import ToolDefinition

_REGISTRY: dict[str, tuple[Callable[..., Any], ToolDefinition]] = {}


def register_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function as a local tool. Schema is inferred from its
    signature and docstring."""
    name = fn.__name__
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        py_type = hints.get(pname, str)
        properties[pname] = {"type": _python_to_json_type(py_type)}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    description = inspect.getdoc(fn) or ""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    _REGISTRY[name] = (fn, ToolDefinition(name=name, description=description, input_schema=schema))
    return fn


def get_local_tools() -> list[ToolDefinition]:
    """Return all registered tools' definitions."""
    return [td for _fn, td in _REGISTRY.values()]


def call_local_tool(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call. Raises KeyError if unregistered."""
    if name not in _REGISTRY:
        raise KeyError(f"Local tool not registered: {name!r}")
    fn, _ = _REGISTRY[name]
    return fn(**args)


def _RESET_FOR_TESTS() -> None:
    _REGISTRY.clear()


def _python_to_json_type(t: type) -> str:
    if t is int:
        return "integer"
    if t is float:
        return "number"
    if t is bool:
        return "boolean"
    if t is list:
        return "array"
    if t is dict:
        return "object"
    return "string"
