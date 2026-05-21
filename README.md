# TravisML Agent Playground

Branded multi-provider agent harness — chat, tools, MCP servers, and
auto-saved transcripts — for experimenting with agentic systems against
Anthropic, OpenAI, and locally-hosted models (vLLM, LM Studio, llama.cpp, anything OpenAI-compatible).

## Setup

```bash
# 1. Activate the project venv (Python 3.14)
source .agent-playground/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
$EDITOR .env   # set ANTHROPIC_API_KEY / OPENAI_API_KEY / LOCAL_BASE_URL
```

## Run

```bash
streamlit run app.py
```

Opens a browser on http://localhost:8501. Click **Basic Chat** in the
sidebar.

## Smoke-test without the GUI

```bash
python -m playground.smoke --provider anthropic --prompt "Say hi"
```

## Run the test suite

```bash
pytest
```

## Add a local tool

Create a Python file under `playground/tools/examples/` (or anywhere; just
make sure it gets imported):

```python
from playground.tools import register_tool

@register_tool
def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]
```

Add the import to `playground/tools/examples/__init__.py`. Refresh the
chat page — the tool appears in "Local tools" and the LLM can call it.

## Add an MCP server

Edit `mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/fs"],
      "description": "Read/write files under /tmp/fs",
      "enabled": true
    }
  }
}
```

Click **Reload mcp.json** in Basic Chat's sidebar.

## Layout

- `app.py` — Home page
- `pages/1_Basic_Chat.py` — the MVP chat page
- `playground/` — shared package: providers, tools, prompts, MCP, persistence
- `mcp_servers/notes/` — bundled MCP server (agent scratch notes)
- `conversations/` — auto-saved JSON transcripts (gitignored)
- `tests/` — pytest suite
- `docs/superpowers/` — design specs and implementation plans

## Memory + Dreaming

The bundled `memory` MCP server gives the playground agent persistent
cross-conversation memory. A separate background dreamer process runs a
six-stage consolidation cycle that produces a bi-temporal knowledge
graph plus speculative hypotheses you can curate from the new
**Dreaming** page.

Quick start:

1. Set `LOCAL_BASE_URL` to your local OpenAI-compatible inference server
   (vLLM, LM Studio, llama.cpp, etc.).
2. `streamlit run app.py` — the memory server starts automatically.
3. Send a few messages in Basic Chat. Then open **Dreaming** → click
   **Start daemon** and **Dream now (full)**.

Design: `docs/superpowers/specs/2026-05-11-memory-dreaming-mcp-design.md`.

## Spec

Full v1 design: [docs/superpowers/specs/2026-05-09-travisml-agent-playground-design.md](docs/superpowers/specs/2026-05-09-travisml-agent-playground-design.md).
