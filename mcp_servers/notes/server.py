"""Notes — bundled example MCP server.

Persists scratch notes to ~/.travisml-playground/notes.json.
Run standalone: python mcp_servers/notes/server.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

NOTES_FILE = Path.home() / ".travisml-playground" / "notes.json"
mcp = FastMCP("notes")


def _load() -> dict[str, str]:
    if not NOTES_FILE.exists():
        return {}
    try:
        return json.loads(NOTES_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _save(notes: dict[str, str]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, indent=2, ensure_ascii=False))


@mcp.tool()
def list_notes() -> list[dict]:
    """List all saved notes (title + first 80-char preview)."""
    notes = _load()
    return [{"title": k, "preview": v[:80]} for k, v in notes.items()]


@mcp.tool()
def save_note(title: str, content: str) -> str:
    """Save a note. Overwrites if a note with the same title exists.

    Args:
        title: short identifier
        content: free-form text body
    """
    notes = _load()
    notes[title] = content
    _save(notes)
    return f"Saved note {title!r}"


@mcp.tool()
def delete_note(title: str) -> str:
    """Delete a note by title."""
    notes = _load()
    if title not in notes:
        return f"No note titled {title!r}"
    del notes[title]
    _save(notes)
    return f"Deleted {title!r}"


if __name__ == "__main__":
    mcp.run()
