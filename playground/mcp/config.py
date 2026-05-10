"""Loader for mcp.json (Claude Desktop / Claude Code format)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True


def load_mcp_config(path: str | Path = "mcp.json") -> dict[str, MCPServerConfig]:
    """Load and parse mcp.json. Missing file returns {}."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        data = json.load(f)
    servers = data.get("mcpServers", {})
    out: dict[str, MCPServerConfig] = {}
    for name, entry in servers.items():
        out[name] = MCPServerConfig(
            name=name,
            command=entry["command"],
            args=list(entry.get("args", [])),
            env=dict(entry.get("env", {})),
            description=entry.get("description", ""),
            enabled=bool(entry.get("enabled", True)),
        )
    return out
