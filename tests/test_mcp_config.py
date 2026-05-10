"""Tests for mcp.json loader."""

from pathlib import Path

from playground.mcp.config import MCPServerConfig, load_mcp_config


def test_load_disabled_server_is_kept(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        """
        {
          "mcpServers": {
            "notes": {
              "command": "python",
              "args": ["x.py"],
              "description": "test",
              "enabled": false
            }
          }
        }
        """
    )
    cfg = load_mcp_config(p)
    assert "notes" in cfg
    notes = cfg["notes"]
    assert isinstance(notes, MCPServerConfig)
    assert notes.command == "python"
    assert notes.args == ["x.py"]
    assert notes.description == "test"
    assert notes.enabled is False


def test_enabled_defaults_to_true(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        """
        {
          "mcpServers": {
            "fs": {
              "command": "npx",
              "args": ["@x/server-fs"]
            }
          }
        }
        """
    )
    cfg = load_mcp_config(p)
    assert cfg["fs"].enabled is True
    assert cfg["fs"].description == ""


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    cfg = load_mcp_config(tmp_path / "does-not-exist.json")
    assert cfg == {}
