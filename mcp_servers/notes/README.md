# notes (bundled MCP server)

Three tools for agent-side scratch notes, persisted to
`~/.travisml-playground/notes.json`:

- `list_notes()` → all saved notes (title + 80-char preview)
- `save_note(title, content)` → upsert
- `delete_note(title)` → delete by title

Run standalone for debugging:
```bash
python mcp_servers/notes/server.py
```

The Playground spawns this via `mcp.json` — you don't normally run it by
hand.
