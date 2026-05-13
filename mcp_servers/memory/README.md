# Memory + Dreaming MCP

Bundled MCP server that gives the playground agent persistent memory
across conversations. See the design doc:

- `docs/superpowers/specs/2026-05-11-memory-dreaming-mcp-design.md`

## Quick start

1. Configure `LMSTUDIO_BASE_URL` to point at your local OpenAI-compatible
   inference server (vLLM, LM Studio, etc.).
2. The Streamlit Dreaming page (`pages/2_Dreaming.py`) starts the dreamer
   daemon and exposes operator controls.
3. The MCP server is registered in `../../mcp.json`.

## Layout

- `server.py` — FastMCP stdio server (hot-path writes + retrieval tools)
- `dreamer.py` — CLI entry for the daemon (`python -m mcp_servers.memory.dreamer serve`)
- `db/` — SQLite schema, migrations, connection helpers
- `repo/` — per-entity data access (one module per table)
- `embeddings/` — `EmbeddingProvider` protocol + implementations
- `extractor/` — atomic episode extraction worker
- `retrieval/` — vector + PageRank recall
- `dreamer_runner/` — daemon lifecycle + six dream stages
- `prompts_lib/` — LLM prompt templates for extraction & dreaming
