# MCP servers

Bundled MCP servers and a place to develop your own.

## Bundled
- `notes/` — agent scratch notes (see [notes/README.md](notes/README.md))

## Writing your own
1. Create a directory under `mcp_servers/<your-server>/`.
2. Use the `mcp` Python SDK's `FastMCP` (see notes server for reference).
3. Add an entry to `../mcp.json`:

```json
{
  "mcpServers": {
    "your-server": {
      "command": "python",
      "args": ["mcp_servers/your-server/server.py"],
      "description": "What it does",
      "enabled": true
    }
  }
}
```

The Playground will spawn it on Basic Chat load.
