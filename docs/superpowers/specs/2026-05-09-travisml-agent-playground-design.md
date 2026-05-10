# TravisML Agent Playground — Design Spec

**Date:** 2026-05-09
**Status:** Approved for v1 implementation
**Owner:** Travis Lelle

## 1. Overview

A branded, browser-based harness for experimenting with agentic systems —
chat, tools, memory, prompts, and MCP servers — across multiple model providers.
Built for engineers who want to see what their agents are actually doing.

The v1 surface is intentionally narrow: a Home page and a single Basic Chat
page that exercises the full stack (multi-provider, system prompts, local +
MCP tools, MCP prompts and resources, auto-saved transcripts). Additional
experiment pages (Tool Workshop, Multi-Agent, Memory Studio, Prompt Evals)
are designed-for but not built; each will be its own brainstorm and spec.

### Goals

- Get a usable multi-provider chat working fast (~2-3 days of focused work).
- Make every agent interaction inspectable: streaming output, collapsible
  tool-call blocks, full transcripts saved to disk as JSON.
- Treat MCP as a first-class citizen: develop your own MCP servers and test
  them against any provider with one code path.
- Carry the TravisML brand (matching travisml.ai) with a light/dark toggle.
- Make adding new experiment pages cheap — drop a file in `pages/`, reuse
  shared infrastructure from the `playground/` package.

### Non-goals (v1)

- Hosted deployment, auth, multi-user.
- Pixel-perfect UI fidelity to the brand mockups (Streamlit constraints — we
  target ~85-90%).
- Anthropic-specific features that aren't on Basic Chat (prompt caching,
  Claude "dreaming"/agent skills, the native MCP API parameter). These belong
  on dedicated pages built later.
- Exhaustive test coverage of UI components (Streamlit widgets are tested
  by Streamlit). Focused tests on logic-heavy code are *in scope* — see §10.

## 2. Tech stack

- **Python 3.14**, project managed via `pyproject.toml` (uv-friendly).
- **Streamlit** as the GUI framework (multipage app pattern).
- **`anthropic`** SDK, **`openai`** SDK (also used for LMStudio via custom
  `base_url`), **`mcp`** SDK (Python MCP client + server framework).
- Conversations persisted as plain JSON files under `conversations/`. No
  database in v1.

## 3. Folder layout

```
agent-playground/
├── .env                          # Secrets — gitignored
├── .env.example                  # Template — committed
├── .gitignore                    # .env, conversations/, .superpowers/, __pycache__/, .venv*/
├── .streamlit/
│   └── config.toml               # Base theme: emerald on porcelain
├── pyproject.toml                # Project metadata + deps
├── README.md                     # Setup, run, write your first agent
├── providers.toml                # Provider catalog (committed)
├── mcp.json                      # MCP server config (committed)
│
├── app.py                        # Streamlit entry — Home page
├── pages/
│   └── 1_Basic_Chat.py           # The MVP page
│
├── playground/                   # Shared Python package — used by every page
│   ├── __init__.py
│   ├── branding.py               # inject_brand_css(), render_brand_wordmark(), render_theme_toggle()
│   ├── chat_ui.py                # render_messages(), render_tool_call_block(), streaming helpers
│   ├── persistence.py            # ConversationStore, Conversation, ConversationSummary
│   ├── providers/
│   │   ├── base.py               # LLMClient protocol, ChatMessage, ToolCall, ToolResult
│   │   ├── anthropic_client.py
│   │   ├── openai_client.py
│   │   ├── lmstudio_client.py    # OpenAI SDK + LMSTUDIO_BASE_URL, no key required
│   │   └── registry.py           # get_client(provider, model, **overrides), list_available_providers()
│   ├── tools/
│   │   ├── __init__.py           # @register_tool, ToolRegistry, get_tool_schemas()
│   │   └── examples/
│   │       ├── echo.py           # Trivial echo tool
│   │       └── get_current_time.py
│   ├── prompts/
│   │   ├── library/              # *.md files — one per saved system prompt
│   │   │   └── default.md
│   │   └── loader.py             # list_prompts(), load_prompt(name)
│   └── mcp/
│       ├── config.py             # load_mcp_config() → dict[str, MCPServerConfig]
│       ├── client.py             # MCPClientPool (cached in st.session_state)
│       └── bridge.py             # mcp_tools_to_provider_format(), MCP messages ↔ playground messages
│
├── mcp_servers/                  # MCP servers shipped with the playground
│   ├── README.md                 # How to write your own
│   └── notes/                    # Bundled example
│       ├── server.py             # FastMCP — list_notes, save_note, delete_note
│       └── README.md
│
├── conversations/                # Auto-saved JSON, gitignored
│   └── basic_chat/
│       └── 2026-05-09T22-30-15-a3f7.json
│
├── tests/                        # Pytest suite (logic-heavy code only)
│   ├── conftest.py               # Shared fixtures, mock provider responses
│   ├── test_persistence.py       # Conversation save/load round-trip, schema
│   ├── test_mcp_bridge.py        # MCP tool schema → provider format
│   ├── test_tools_registry.py    # @register_tool, schema inference
│   ├── test_providers.py         # Each LLMClient against mocked HTTP responses
│   └── fixtures/                 # Captured JSON conversations, recorded API responses
│
└── docs/
    └── superpowers/specs/        # This file lives here
```

The existing empty `anthropic/` and `openai/` directories at the project root
will be removed; their concerns are absorbed by `playground/providers/`.

## 4. Configuration

Three layers, from rarely touched to per-message:

### 4.1 Secrets — `.env` (gitignored)

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LMSTUDIO_BASE_URL=http://localhost:1234/v1   # Override for non-default
```

`.env.example` ships with the keys present but blank. Loaded via
`python-dotenv` at app startup.

### 4.2 Provider catalog — `providers.toml` (committed)

```toml
[anthropic]
models = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]
default_model = "claude-sonnet-4-6"
default_max_tokens = 4096
default_temperature = 1.0
capabilities = ["tools", "streaming", "prompt_caching", "mcp_native"]

[openai]
models = ["gpt-4o", "gpt-4o-mini", "o1"]
default_model = "gpt-4o"
default_max_tokens = 4096
default_temperature = 1.0
capabilities = ["tools", "streaming"]

[lmstudio]
# Models discovered at runtime from /v1/models — no static list
default_max_tokens = 2048
default_temperature = 0.7
capabilities = ["tools", "streaming"]
```

`capabilities` drives UI affordances. For example, `prompt_caching` and
`mcp_native` are present on Anthropic but not used by Basic Chat — they exist
so future pages know what's available without hardcoding provider names.

### 4.3 Runtime UI state — `st.session_state` per page

Provider, model, temperature, max_tokens, current system prompt, MCP server
toggles, theme. Resets per browser session.

### 4.4 Reproducibility — captured in saved conversation JSON

Every saved conversation snapshots the full config at run-time, so
re-opening a year-old conversation tells you exactly what the agent was
configured with. See §8 for schema.

## 5. Branding

Matches travisml.ai — palette "Emerald on Porcelain" (light, default) and
"Emerald on Ink" (dark). Implementation has two layers:

### 5.1 `.streamlit/config.toml` — Streamlit's built-in theme

```toml
[theme]
base = "light"
primaryColor = "#047A5E"
backgroundColor = "#F4F1EA"
secondaryBackgroundColor = "#EDE9DE"
textColor = "#0F1E16"
font = "sans serif"
```

Picks up colors for built-in widgets (chat messages, buttons, sidebar
chrome). Handles roughly 60% of the look.

### 5.2 `playground/branding.py` — CSS injection

Three exported functions:

- `inject_brand_css()` — emits `<style>` block that loads Fraunces / Sora /
  JetBrains Mono Google Fonts, sets CSS variables based on current theme,
  styles headings (Fraunces with italic emerald `em` for emphasis), restyles
  Streamlit chrome with brand variables.
- `render_brand_wordmark()` — renders the "TravisML / *Playground*" wordmark
  in the sidebar.
- `render_theme_toggle()` — renders a `st.sidebar.toggle` for dark mode at
  the bottom of the sidebar; on change updates `st.session_state.theme` and
  calls `st.rerun()`.

### 5.3 Page convention

Every page (including `app.py`) starts with:

```python
from playground.branding import inject_brand_css, render_brand_wordmark, render_theme_toggle
inject_brand_css()
render_brand_wordmark()
render_theme_toggle()
```

### 5.4 Palettes

**Light — Emerald on Porcelain**
- bg-void `#F4F1EA`, bg-deep `#EDE9DE`, bg-panel `#E6E1D2`, bg-elevated `#DDD8C8`
- line `rgba(15,30,22,0.10)`, line-strong `rgba(15,30,22,0.24)`
- text-100 `#0F1E16`, text-200 `#2A3D32`, text-300 `#5A6D62`, text-400 `#8B9C92`
- accent `#047A5E`, accent-bright `#0BA37A`

**Dark — Emerald on Ink**
- bg-void `#0D1612`, bg-deep `#131C17`, bg-panel `#1A241D`, bg-elevated `#222D25`
- line `rgba(244,241,234,0.08)`, line-strong `rgba(244,241,234,0.20)`
- text-100 `#F4F1EA`, text-200 `#D4D0C2`, text-300 `#9AA89E`, text-400 `#6E7C73`
- accent `#0BA37A`, accent-bright `#14C490` (brighter emerald — needed for contrast on dark backgrounds)

Theme defaults to light and persists within the session (not across browser
sessions in v1).

## 6. Provider abstraction

### 6.1 `LLMClient` protocol

`playground/providers/base.py` defines a Protocol with one method:

```python
def stream_chat(
    self,
    messages: list[ChatMessage],
    *,
    system: str | None,
    tools: list[ToolDefinition],
    max_tokens: int,
    temperature: float,
) -> Iterator[StreamEvent]: ...
```

`StreamEvent` is a tagged union covering `TextDelta`, `ToolCallDelta`,
`ToolCallComplete`, `MessageComplete(usage)`. Provider-specific events get
normalized into this shared shape so `chat_ui.py` doesn't branch on provider.

### 6.2 Implementations

- **`AnthropicClient`** — uses `anthropic.messages.stream()`. Maps native
  content blocks to `StreamEvent`s. Captures `usage.cache_read_tokens` when
  present (for the day we add a caching toggle).
- **`OpenAIClient`** — uses `openai.chat.completions.create(stream=True)`.
  Translates OpenAI's tool-call delta format into `ToolCallDelta` events.
- **`LMStudioClient`** — same `openai` SDK with `base_url=LMSTUDIO_BASE_URL`,
  `api_key="lm-studio"` (a placeholder — LM Studio ignores it). On startup,
  hits `/v1/models` to populate the model dropdown dynamically.

### 6.3 Registry

```python
# playground/providers/registry.py
def get_client(provider: str, model: str, **overrides) -> LLMClient: ...
def list_available_providers() -> list[str]: ...      # Filters by env vars present
def list_models(provider: str) -> list[str]: ...      # Static for anthropic/openai, dynamic for lmstudio
```

A provider only appears in the dropdown if its required env var is set
(e.g., LMStudio shows only if `LMSTUDIO_BASE_URL` is reachable; Anthropic
shows only if `ANTHROPIC_API_KEY` is set).

## 7. Tools + MCP

Three sources of tools, all rendered the same way in the transcript:

| Source | Mechanism |
|--------|-----------|
| **Local** | Python functions decorated with `@register_tool` in `playground/tools/` |
| **MCP** | Tools exposed by connected MCP servers, discovered via `tools/list` |
| **Builtin** | One tool: `read_mcp_resource(uri)` — synthesized when any MCP server is connected |

### 7.1 Local tool registry

```python
# playground/tools/__init__.py
@register_tool
def echo(text: str) -> str:
    """Echo the input text back."""
    return text
```

The decorator infers JSON Schema from the function signature and docstring.
`get_tool_schemas()` returns the list ready to pass to a provider.

### 7.2 MCP — single code path for all providers

The playground is itself an MCP client (using the `mcp` Python SDK). For each
enabled server in `mcp.json`, it spawns/connects (stdio or HTTP), aggregates
tools/prompts/resources, and passes tool definitions to whatever LLM is
selected. Anthropic's native API `mcp_servers` parameter is **not** used in
v1 — when you want to test that specific surface, it'll be on its own page.

#### `mcp.json` format (matches Claude Desktop / Claude Code)

```json
{
  "mcpServers": {
    "notes": {
      "command": "python",
      "args": ["mcp_servers/notes/server.py"],
      "description": "Bundled — agent scratch notes",
      "enabled": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/playground-fs"],
      "description": "Read/write files under /tmp/playground-fs",
      "enabled": false
    }
  }
}
```

Two extras beyond the standard format: `description` (sidebar caption) and
`enabled` (default `true`).

#### `playground/mcp/client.py` surface

```python
class MCPClientPool:
    """Manages connections to multiple MCP servers across one Streamlit session."""
    def start(self, servers: dict[str, MCPServerConfig]) -> None: ...
    def shutdown(self) -> None: ...

    def list_tools(self, servers: list[str]) -> list[Tool]: ...
    def list_prompts(self, servers: list[str]) -> list[Prompt]: ...
    def list_resources(self, servers: list[str]) -> list[Resource]: ...

    async def call_tool(self, server: str, tool: str, args: dict) -> ToolResult: ...
    async def get_prompt(self, server: str, prompt: str, args: dict) -> list[Message]: ...
    async def read_resource(self, server: str, uri: str) -> ResourceContent: ...
```

The pool gets cached in `st.session_state` so subprocesses survive Streamlit
reruns. A "Reload mcp.json" button shuts down + restarts the pool.

### 7.3 UI surface in Basic Chat sidebar

Three sections, only visible if at least one MCP server is connected:

```
─── MCP Servers ───
[✓] notes            (bundled)
[ ] filesystem       (disabled in mcp.json)
[Reload mcp.json]

─── MCP Prompts ───
Server: [notes ▾]
Prompt: [summarize_notes ▾]
  → arg: max_count [10    ]
[Use as user message] [Use as system prompt]

─── MCP Resources ───
Server: [filesystem ▾]
  ☐ file:///tmp/playground-fs/notes.md      (text/markdown · 4.2 KB)
  ☐ file:///tmp/playground-fs/data.json     (application/json · 12 KB)
[Attach selected to next message]   [Refresh list]
```

- **Tools** are auto-attached when their server is checked.
- **Prompts** call `prompts/get` with filled args; "Use as user message"
  appends to the conversation, "Use as system prompt" replaces it (with a
  destructive-action confirmation).
- **Resources** can be user-attached (selected resources prepended as
  `<resource uri="..." mimeType="...">…</resource>` blocks before the next
  user message) or LLM-pulled via the builtin `read_mcp_resource(uri)` tool.

### 7.4 Bundled `mcp_servers/notes/` server

A real, working MCP server (~80 LOC) using `FastMCP`. Three tools:

```python
@mcp.tool()
def list_notes() -> list[dict]: ...

@mcp.tool()
def save_note(title: str, content: str) -> str: ...

@mcp.tool()
def delete_note(title: str) -> str: ...
```

Notes persist to `~/.travisml-playground/notes.json`. The server doubles as
a template — `mcp_servers/README.md` walks through writing your own based
on it.

## 8. Persistence

### 8.1 File layout

```
conversations/
└── basic_chat/                              # one folder per page
    ├── 2026-05-09T22-30-15-a3f7.json        # ISO timestamp + 4-char short id
    └── 2026-05-09T22-34-02-9b1c.json
```

Naturally chronological; short id disambiguates same-second runs.

### 8.2 JSON schema (v1)

```json
{
  "schema_version": 1,
  "id": "2026-05-09T22-30-15-a3f7",
  "page": "basic_chat",
  "started_at": "2026-05-09T22:30:15Z",
  "ended_at": "2026-05-09T22:34:02Z",

  "config": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "temperature": 1.0,
    "system_prompt": {
      "source": "library/default.md",
      "text": "You are a helpful assistant."
    },
    "tools": {
      "local": ["echo", "get_current_time"],
      "mcp": [
        {"server": "notes", "tools": ["list_notes", "save_note", "delete_note"]}
      ],
      "builtin": ["read_mcp_resource"]
    },
    "mcp_servers_enabled": ["notes"]
  },

  "messages": [
    {
      "role": "user",
      "ts": "2026-05-09T22:30:18Z",
      "content": [{"type": "text", "text": "What notes do I have?"}]
    },
    {
      "role": "assistant",
      "ts": "2026-05-09T22:30:21Z",
      "content": [
        {"type": "text", "text": "Let me check..."},
        {
          "type": "tool_use",
          "id": "toolu_01abc",
          "name": "list_notes",
          "source": {"kind": "mcp", "server": "notes"},
          "input": {}
        }
      ],
      "usage": {"input_tokens": 250, "output_tokens": 42, "cache_read_tokens": 0}
    },
    {
      "role": "user",
      "ts": "2026-05-09T22:30:21Z",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "toolu_01abc",
        "content": [{"type": "text", "text": "[]"}],
        "duration_ms": 8,
        "is_error": false
      }]
    }
  ],

  "events": [
    {"ts": "2026-05-09T22:30:30Z", "type": "resource_attached",
     "server": "filesystem", "uri": "file:///tmp/foo.md"},
    {"ts": "2026-05-09T22:31:02Z", "type": "prompt_inserted",
     "server": "notes", "prompt": "summarize_notes", "args": {"max_count": 10}}
  ]
}
```

Notes on the schema:

- Mostly Anthropic-native message shape (typed content blocks). It's the
  most expressive; OpenAI translates in cleanly. Avoids inventing a new
  format.
- `source` on `tool_use` blocks disambiguates `local` / `mcp` / `builtin`
  origin — replays of old conversations can trace tools to their source.
- `events` array captures non-message actions that affect reproducibility:
  resource attachments, MCP prompt insertions, server toggles mid-run.
- `schema_version: 1` — forward-compatible; we'll bump and migrate when the
  schema changes.
- `usage` per assistant turn captures tokens including cache reads.

### 8.3 Module API

```python
# playground/persistence.py

@dataclass
class ConversationSummary:
    id: str
    page: str
    started_at: datetime
    ended_at: datetime | None
    provider: str
    model: str
    message_count: int
    first_user_message: str   # truncated to ~80 chars

class ConversationStore:
    def __init__(self, root: Path = Path("conversations")): ...
    def new(self, page: str, config: dict) -> Conversation: ...
    def list(self, page: str | None = None) -> list[ConversationSummary]: ...
    def load(self, conv_id: str) -> Conversation: ...

class Conversation:
    """Open conversation — saves to disk after every append."""
    def append_message(self, msg: dict) -> None: ...
    def add_event(self, event: dict) -> None: ...
    def end(self) -> None: ...
```

`Conversation.append_message()` uses temp-file + atomic rename so a crash
never leaves a half-written conversation.

`ConversationStore.list()` parses each JSON file's top-level keys to build
summaries (cheap until ~hundreds of files; we'll add a per-page
`_index.jsonl` or graduate to SQLite when we feel pain).

## 9. Pages

### 9.1 Home (`app.py`)

- Brand wordmark, hero ("Build, test, debug *agents*").
- Three provider status cards (Anthropic / OpenAI / LMStudio) with connection
  state derived from env vars and `/v1/models` reachability for LMStudio.
- MCP server list from `mcp.json`.
- Theme toggle in sidebar.

### 9.2 Basic Chat (`pages/1_Basic_Chat.py`)

The MVP page — exercises every system in the harness.

**Sidebar (top to bottom):**
- Brand wordmark + version
- Page selector (Streamlit default)
- Provider dropdown → model dropdown
- Temperature slider, max_tokens input
- System prompt: free-text editor seeded from a dropdown (`prompts/library/`)
- MCP Servers checkbox list (only if any configured)
- MCP Prompts widget (only if any connected)
- MCP Resources widget (only if any connected)
- History section: list past conversations on this page
- Theme toggle (bottom)

**Main pane:**
- Streaming chat transcript using `st.chat_message`
- Tool-call rendering via `playground.chat_ui.render_tool_call_block` —
  collapsible expander showing tool name, source, JSON input, JSON result,
  duration in ms
- Chat input at bottom (Streamlit `st.chat_input`)

**Behaviors:**
- Each turn streams tokens via `st.write_stream`.
- Tool-use loop runs until the LLM stops emitting tool calls (no manual
  user approval per call in v1 — bounded by max iterations to prevent
  runaway loops; default cap of 10).
- Every turn is appended to a `Conversation` object, which auto-saves.
- Loading a past conversation puts the page in **read-only mode** with a
  "Fork from here" button that creates a new conversation pre-seeded with
  the loaded transcript.

## 10. Testing

Two complementary surfaces — neither aims for 100% coverage; both aim to
catch regressions in the parts that actually break things.

### 10.1 Programmatic — pytest suite under `tests/`

Focused tests on logic-heavy code paths. Streamlit widgets are out of scope
(they're tested by Streamlit; trying to test our usage of them is brittle).
Target areas:

- **Persistence** (`test_persistence.py`) — `Conversation` round-trip
  save/load, schema validation, atomic-write behavior, `ConversationStore.list()`
  summary parsing.
- **MCP bridge** (`test_mcp_bridge.py`) — translation of MCP tool schemas to
  Anthropic and OpenAI tool formats; tool-call dispatch routing
  (local vs MCP vs builtin).
- **Tool registry** (`test_tools_registry.py`) — `@register_tool` schema
  inference from function signatures + docstrings.
- **Provider clients** (`test_providers.py`) — each `LLMClient`
  implementation against recorded HTTP responses (using `respx` or similar).
  Verify `StreamEvent` normalization across the three providers.

Run with `pytest`. CI not required for v1, but the suite is structured so
adding GitHub Actions is a single workflow file later.

### 10.2 Programmatic — smoke-test CLI

A tiny CLI that runs a turn end-to-end without Streamlit:

```bash
python -m playground.smoke --provider anthropic --model claude-sonnet-4-6 \
    --prompt "Hello, can you list the notes?" --mcp-server notes
```

Use it to:
- Verify the foundation works without launching the GUI
- Quickly test new tools or MCP servers
- Reproduce conversation bugs from a saved JSON

### 10.3 Manual — interactive exploratory testing

The GUI itself is the primary manual test surface. Each build-order step
ends with a small "smoke checklist" — concrete things to verify by hand
(see §11). Saved conversation JSON files double as manual-test fixtures:
when something looks wrong, the conversation is replayable.

## 11. Build order

Each step ends with something runnable. Tests are added alongside the code
they cover, not at the end as a separate phase.

1. **Skeleton** — `pyproject.toml`, deps, `.env.example`, `.gitignore`,
   `.streamlit/config.toml`, providers/MCP config loaders, `app.py` with
   brand wordmark + theme toggle, `tests/` directory + `conftest.py`.
   Smoke check: `streamlit run app.py` shows the home page.
2. **Multi-provider chat** — `LLMClient` protocol + Anthropic / OpenAI /
   LMStudio implementations, provider+model dropdown, bare streaming chat
   (no tools), saves JSON. **Tests:** `test_providers.py` with recorded
   HTTP responses; `test_persistence.py` save/load round-trip.
   Smoke check: `python -m playground.smoke --provider anthropic --prompt hi`
   returns a streamed response.
3. **System prompt editor + library** — sidebar textarea + dropdown reading
   `prompts/library/`. Smoke check: switching prompts updates next request.
4. **Local tools** — `@register_tool`, two examples, collapsible tool-call
   rendering. **Tests:** `test_tools_registry.py` schema inference.
   Smoke check: agent calls `get_current_time`, result rendered inline.
5. **MCP tools** — client pool, `mcp.json` loader, bundled `notes` server,
   sidebar checkboxes per server. **Tests:** `test_mcp_bridge.py` schema
   translation. Smoke check: agent saves and lists notes via MCP.
6. **MCP prompts** — sidebar dropdown + insertion as user/system message.
7. **MCP resources** — sidebar list + attach + `read_mcp_resource` builtin.
8. **History sidebar** — list past conversations, load read-only, fork.
   Smoke check: re-open a conversation from step 2; "Fork from here" works.
9. **Polish** — README (install + first-run + add-a-tool walkthrough),
   example tool comments, `mcp.json` comments, README screenshots, run full
   pytest suite green.

## 12. Out of scope (explicit, deferred to later versions)

- Additional pages (Tool Workshop, Multi-Agent, Memory Studio, Prompt
  Evals) — each will be its own brainstorm + spec.
- Anthropic native `mcp_servers` API parameter — wait until there's a page
  that specifically needs to test it.
- Anthropic prompt caching toggles, Claude "dreaming"/agent skills — surface
  on dedicated pages, not Basic Chat.
- Streaming tool-call inputs (rendering JSON as it builds) — show after
  tool call completes.
- SQLite storage / conversation index file — direct file scan until pain.
- Theme persistence across browser sessions — session-only is fine for v1.
- Auth, multi-user, hosted deployment — local `streamlit run` only.
- Streamlit-widget tests, end-to-end browser tests, CI pipeline. Focused
  pytest suite + smoke-test CLI are in scope (see §10).

## 13. Open questions / things to revisit after first use

- **Tool-call iteration cap (default 10)** — may need tuning once real
  agents are running. Should it be a sidebar slider per page?
- **Resource attachment encoding** — `<resource>` XML-style blocks before
  user text is one approach; could also use Anthropic's native document
  content blocks. Revisit if the LLM struggles to attribute attached content.
- **Empty-state UX** — what does the chat page look like before any
  conversation has happened? Currently just an empty transcript and the
  sidebar. May want a quick-start panel.
- **Cost telemetry** — `usage` is captured per turn but not surfaced in the
  UI. Consider a small token/cost counter in the sidebar once we have
  several conversations to compare.
