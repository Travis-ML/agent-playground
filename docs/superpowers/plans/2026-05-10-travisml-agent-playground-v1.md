# TravisML Agent Playground v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-provider, MCP-aware Streamlit chat harness branded as TravisML Agent Playground (matching travisml.ai), with auto-saved JSON transcripts and a focused pytest suite.

**Architecture:** Streamlit multipage app. A `playground/` Python package holds shared infrastructure (providers, tools, MCP, persistence, branding). Pages under `pages/` are thin scripts that compose the package. Three providers (Anthropic, OpenAI, LMStudio) sit behind one `LLMClient` protocol. MCP integration uses the Python `mcp` SDK as a client; one example MCP server (`notes`) ships in `mcp_servers/`.

**Tech Stack:** Python 3.14 · Streamlit · `anthropic`, `openai`, `mcp` SDKs · `python-dotenv` · `pytest` + `respx` · `tomli` (3.11+ has `tomllib` in stdlib — Python 3.14 confirmed) · uv-friendly `pyproject.toml`.

**Source spec:** [docs/superpowers/specs/2026-05-09-travisml-agent-playground-design.md](../specs/2026-05-09-travisml-agent-playground-design.md)

---

## Conventions used in this plan

- **TDD where it pays off** — logic-heavy code (persistence, registry, bridge, providers) gets a failing test first. UI/Streamlit wiring uses **manual smoke checks** instead, because Streamlit widgets are tested by Streamlit and reproducing their runtime is brittle.
- **Commit cadence:** commit after each task. Use Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`).
- **All paths relative to repo root** `/Users/travislelle/agent-playground/`.
- **Run commands** assume the venv at `.agent-playground/` is active. Use `source .agent-playground/bin/activate` once at the start of a session.
- **Trailing co-author line** on every commit:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

---

## File structure (locked in)

```
agent-playground/
├── .env.example                         # committed; .env gitignored
├── .gitignore                           # already exists; will be expanded
├── .streamlit/config.toml               # base theme
├── pyproject.toml                       # project + deps
├── README.md                            # added in Phase 12
├── providers.toml                       # provider catalog
├── mcp.json                             # MCP server config
│
├── app.py                               # Streamlit Home page
├── pages/
│   └── 1_Basic_Chat.py                  # MVP page
│
├── playground/
│   ├── __init__.py
│   ├── branding.py                      # CSS, wordmark, theme toggle
│   ├── chat_ui.py                       # message + tool-call rendering
│   ├── persistence.py                   # ConversationStore + Conversation
│   ├── smoke.py                         # python -m playground.smoke CLI
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                      # LLMClient protocol + types
│   │   ├── anthropic_client.py
│   │   ├── openai_client.py
│   │   ├── lmstudio_client.py
│   │   └── registry.py
│   ├── tools/
│   │   ├── __init__.py                  # @register_tool, get_local_tools()
│   │   └── examples/
│   │       ├── __init__.py              # imports all examples to register them
│   │       ├── echo.py
│   │       └── get_current_time.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── loader.py                    # list_prompts, load_prompt
│   │   └── library/
│   │       └── default.md
│   └── mcp/
│       ├── __init__.py
│       ├── config.py                    # MCPServerConfig, load_mcp_config
│       ├── client.py                    # MCPClientPool
│       └── bridge.py                    # MCP ↔ provider tool format
│
├── mcp_servers/
│   ├── README.md
│   └── notes/
│       ├── server.py
│       └── README.md
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # shared fixtures
│   ├── test_persistence.py
│   ├── test_providers.py
│   ├── test_tools_registry.py
│   ├── test_mcp_bridge.py
│   ├── test_mcp_config.py
│   ├── test_prompts_loader.py
│   └── fixtures/
│       └── conversation_v1.json         # canonical example for round-trip tests
│
├── conversations/                       # gitignored, auto-created at runtime
└── docs/
    └── superpowers/
        ├── specs/                       # already populated
        └── plans/                       # this plan lives here
```

The pre-existing empty `anthropic/` and `openai/` directories at the repo root are **removed in Task 1.0**.

---

## Phase 0 — Project initialization

### Task 0.0: Activate venv and remove stale empty dirs

**Files:**
- Delete: `anthropic/`, `openai/` (both empty)

- [ ] **Step 1: Activate the existing venv**

```bash
cd /Users/travislelle/agent-playground
source .agent-playground/bin/activate
python --version
```

Expected: `Python 3.14.4`.

- [ ] **Step 2: Remove the empty placeholder directories**

```bash
rmdir anthropic openai
git status
```

Expected: working tree clean (the dirs were untracked).

- [ ] **Step 3: Commit (skip — nothing tracked changed)**

No commit; proceed to Task 0.1.

---

### Task 0.1: Create `pyproject.toml` with all v1 dependencies

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "travisml-agent-playground"
version = "0.1.0"
description = "Branded multi-provider agent playground — chat, tools, MCP, dreaming, and more."
readme = "README.md"
requires-python = ">=3.12"
authors = [{ name = "Travis Lelle" }]
license = { text = "MIT" }

dependencies = [
  "streamlit>=1.40",
  "anthropic>=0.40",
  "openai>=1.55",
  "mcp>=0.9",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.24",
  "respx>=0.21",
  "ruff>=0.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["playground"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 2: Install the project in editable mode with dev extras**

```bash
pip install -e ".[dev]"
```

Expected: installs streamlit, anthropic, openai, mcp, python-dotenv, pytest, pytest-asyncio, respx, ruff, hatchling.

- [ ] **Step 3: Sanity-check imports**

```bash
python -c "import streamlit, anthropic, openai, mcp, dotenv, pytest, respx; print('all imports OK')"
```

Expected: `all imports OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore: initialize pyproject.toml with v1 dependencies

Streamlit + provider SDKs (anthropic, openai, mcp) + python-dotenv runtime;
pytest + pytest-asyncio + respx + ruff for dev. Hatchling build backend so
the project is editable-installable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.2: Expand `.gitignore` and add `.env.example`

**Files:**
- Modify: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Replace `.gitignore` with the v1 version**

```bash
cat > .gitignore <<'EOF'
# Brainstorm session files (visual companion mockups, server logs)
.superpowers/

# Claude Code local state
.claude/

# Secrets — never commit
.env

# Python venv
.agent-playground/
.venv/

# Python cache
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# Saved conversations (auto-saved per-run; remove this line to commit interesting fixtures)
conversations/

# Notes server data store
.travisml-playground/

# pytest
.pytest_cache/

# ruff
.ruff_cache/

# macOS
.DS_Store
EOF
```

- [ ] **Step 2: Create `.env.example`**

```bash
cat > .env.example <<'EOF'
# Copy to .env and fill in. .env is gitignored.

ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# LM Studio (or any OpenAI-compatible local server). Override if non-default.
LMSTUDIO_BASE_URL=http://localhost:1234/v1
EOF
```

- [ ] **Step 3: Verify `.env.example` parses cleanly with python-dotenv**

```bash
python -c "from dotenv import dotenv_values; print(dotenv_values('.env.example'))"
```

Expected: `{'ANTHROPIC_API_KEY': '', 'OPENAI_API_KEY': '', 'LMSTUDIO_BASE_URL': 'http://localhost:1234/v1'}`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore .env.example
git commit -m "$(cat <<'EOF'
chore: expand .gitignore and add .env.example

Adds Python build/cache artifacts, conversations/, notes data store, and
pytest/ruff caches. .env.example documents the three runtime env vars.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.3: Create the `tests/` scaffold

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create `tests/__init__.py`** (empty file)

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for the playground."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_conversations_root(tmp_path: Path) -> Path:
    """A fresh conversations/ root per test."""
    root = tmp_path / "conversations"
    root.mkdir()
    return root


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def canonical_conversation(fixtures_dir: Path) -> dict:
    """A v1-schema conversation used for round-trip tests."""
    with (fixtures_dir / "conversation_v1.json").open() as f:
        return json.load(f)
```

- [ ] **Step 3: Create the fixtures directory placeholder**

```bash
mkdir -p tests/fixtures
touch tests/fixtures/.gitkeep
```

- [ ] **Step 4: Verify pytest discovers no tests yet**

```bash
pytest -q
```

Expected: `no tests ran` exit 5 — that's fine, the harness is wired up.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: scaffold pytest harness with conftest fixtures

Adds tests/__init__.py, conftest.py with tmp_conversations_root and
canonical_conversation fixtures, and an empty fixtures/ directory. No tests
yet — those come with each phase that adds testable code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — Branding foundation + Home page

### Task 1.1: Create the `playground` package and base theme

**Files:**
- Create: `playground/__init__.py`
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Create empty package init**

```bash
mkdir -p playground
touch playground/__init__.py
```

- [ ] **Step 2: Create `.streamlit/config.toml`**

```bash
mkdir -p .streamlit
cat > .streamlit/config.toml <<'EOF'
[theme]
base = "light"
primaryColor = "#047A5E"
backgroundColor = "#F4F1EA"
secondaryBackgroundColor = "#EDE9DE"
textColor = "#0F1E16"
font = "sans serif"

[server]
headless = true
runOnSave = true

[browser]
gatherUsageStats = false
EOF
```

- [ ] **Step 3: Commit**

```bash
git add playground/ .streamlit/
git commit -m "$(cat <<'EOF'
feat: scaffold playground package and base Streamlit theme

Streamlit theme set to "Emerald on Porcelain" defaults. Server runs
headless and reloads on save; usage stats disabled.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Implement `playground/branding.py` (palettes, fonts, wordmark, theme toggle)

**Files:**
- Create: `playground/branding.py`

- [ ] **Step 1: Write `playground/branding.py`**

```python
"""Brand application — palettes, font loading, wordmark, theme toggle."""

from __future__ import annotations

from typing import TypedDict

import streamlit as st


class Palette(TypedDict):
    bg_void: str
    bg_deep: str
    bg_panel: str
    bg_elevated: str
    line: str
    line_strong: str
    text_100: str
    text_200: str
    text_300: str
    text_400: str
    accent: str
    accent_bright: str


LIGHT: Palette = {
    "bg_void": "#F4F1EA",
    "bg_deep": "#EDE9DE",
    "bg_panel": "#E6E1D2",
    "bg_elevated": "#DDD8C8",
    "line": "rgba(15,30,22,0.10)",
    "line_strong": "rgba(15,30,22,0.24)",
    "text_100": "#0F1E16",
    "text_200": "#2A3D32",
    "text_300": "#5A6D62",
    "text_400": "#8B9C92",
    "accent": "#047A5E",
    "accent_bright": "#0BA37A",
}

DARK: Palette = {
    "bg_void": "#0D1612",
    "bg_deep": "#131C17",
    "bg_panel": "#1A241D",
    "bg_elevated": "#222D25",
    "line": "rgba(244,241,234,0.08)",
    "line_strong": "rgba(244,241,234,0.20)",
    "text_100": "#F4F1EA",
    "text_200": "#D4D0C2",
    "text_300": "#9AA89E",
    "text_400": "#6E7C73",
    "accent": "#0BA37A",
    "accent_bright": "#14C490",
}

_FONTS_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&"
    "family=JetBrains+Mono:wght@400;500&"
    "family=Sora:wght@300;400;500&display=swap"
)


def get_theme() -> Palette:
    """Return the active palette; defaults to light."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    return DARK if st.session_state.theme == "dark" else LIGHT


def inject_brand_css() -> None:
    """Emit brand CSS scoped to the current theme. Call once per page."""
    t = get_theme()
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="{_FONTS_HREF}" rel="stylesheet">
        <style>
          :root {{
            --bg-void: {t['bg_void']};
            --bg-deep: {t['bg_deep']};
            --bg-panel: {t['bg_panel']};
            --bg-elevated: {t['bg_elevated']};
            --line: {t['line']};
            --line-strong: {t['line_strong']};
            --text-100: {t['text_100']};
            --text-200: {t['text_200']};
            --text-300: {t['text_300']};
            --text-400: {t['text_400']};
            --accent: {t['accent']};
            --accent-bright: {t['accent_bright']};
          }}

          html, body, [class*="st-"] {{
            font-family: 'Sora', sans-serif;
            font-weight: 300;
          }}

          h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--text-100);
          }}
          h1 em, h2 em, h3 em {{
            font-style: italic;
            color: var(--accent);
            font-feature-settings: "ss01";
          }}

          code, pre, .stCodeBlock {{
            font-family: 'JetBrains Mono', monospace;
          }}

          .stApp {{
            background: var(--bg-void);
            color: var(--text-200);
          }}
          [data-testid="stSidebar"] {{
            background: var(--bg-deep);
            border-right: 1px solid var(--line);
          }}

          .tml-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--text-300);
          }}
          .tml-label::before {{
            content: '';
            display: inline-block;
            width: 5px;
            height: 5px;
            background: var(--accent);
            margin-right: 8px;
            vertical-align: 2px;
          }}

          .block-container {{
            padding-top: 2rem;
            max-width: 1100px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_wordmark() -> None:
    """Render the 'TravisML / Playground' wordmark in the sidebar."""
    st.sidebar.markdown(
        """
        <div style="font-family:'Fraunces',serif;font-weight:600;font-size:22px;
                    line-height:1.05;color:var(--text-100);margin-bottom:6px;">
          TravisML<br>
          <em style="font-style:italic;font-weight:500;color:var(--accent);
                     font-feature-settings:'ss01';">Playground</em>
        </div>
        <div class="tml-label" style="margin-bottom:24px;">Agent harness · v0.1</div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """Sidebar widget — sticks to the bottom of the sidebar."""
    with st.sidebar:
        st.divider()
        current = st.session_state.get("theme", "light")
        is_dark = st.toggle("Dark mode", value=(current == "dark"), key="_theme_toggle")
        next_theme = "dark" if is_dark else "light"
        if next_theme != current:
            st.session_state.theme = next_theme
            st.rerun()
```

- [ ] **Step 2: Sanity-check the module imports cleanly**

```bash
python -c "from playground.branding import LIGHT, DARK, inject_brand_css, render_brand_wordmark, render_theme_toggle; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add playground/branding.py
git commit -m "$(cat <<'EOF'
feat(branding): light + dark palettes, font loading, wordmark, theme toggle

Three exports drive every page: inject_brand_css() emits the CSS variables
+ Fraunces/Sora/JetBrains Mono Google Fonts; render_brand_wordmark() puts
the editorial 'TravisML / Playground' mark in the sidebar; and
render_theme_toggle() flips st.session_state.theme between LIGHT and DARK.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Build the Home page (`app.py`)

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py`**

```python
"""TravisML Agent Playground — Home page."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from playground.branding import (
    inject_brand_css,
    render_brand_wordmark,
    render_theme_toggle,
)

load_dotenv()

st.set_page_config(
    page_title="TravisML Agent Playground",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_brand_css()
render_brand_wordmark()


# --- Hero ---------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex;gap:22px;flex-wrap:wrap;margin-bottom:36px;">
      <span class="tml-label">Edition / 001</span>
      <span class="tml-label">Local · v0.1</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="font-size:clamp(40px,7vw,82px);line-height:0.95;
               letter-spacing:-0.035em;max-width:14ch;margin-bottom:28px;">
      Build, test, debug <em>agents</em>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="border-top:1px solid var(--line);padding-top:24px;
                font-size:17px;color:var(--text-200);max-width:60ch;
                line-height:1.6;margin-bottom:48px;">
      <strong style="color:var(--text-100);font-weight:500;">
        TravisML Agent Playground
      </strong> is a branded harness for experimenting with agentic systems —
      chat, tools, memory, prompts, and MCP servers — across multiple model
      providers.
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Provider status grid ----------------------------------------------

st.markdown('<div class="tml-label">Providers</div>', unsafe_allow_html=True)


def _status_card(title: str, model_summary: str, connected: bool) -> str:
    accent = "var(--accent)" if connected else "#C97A2A"
    label = "Connected" if connected else "Awaiting setup"
    return f"""
    <div style="background:var(--bg-deep);border:1px solid var(--line);
                padding:22px;height:100%;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  letter-spacing:0.16em;text-transform:uppercase;
                  color:{accent};display:flex;align-items:center;gap:8px;
                  margin-bottom:8px;">
        <span style="display:inline-block;width:5px;height:5px;
                     background:{accent};"></span>{label}
      </div>
      <div style="font-family:'Fraunces',serif;font-weight:600;font-size:18px;
                  color:var(--text-100);letter-spacing:-0.01em;
                  margin-bottom:6px;">{title}</div>
      <div style="font-size:13px;color:var(--text-300);line-height:1.55;">
        {model_summary}
      </div>
    </div>
    """


anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
lmstudio_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

cols = st.columns(3, gap="medium")
with cols[0]:
    st.markdown(
        _status_card("Anthropic / Claude", "opus-4-7, sonnet-4-6, haiku-4-5", anthropic_ok),
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        _status_card("OpenAI / GPT", "gpt-4o, gpt-4o-mini, o1", openai_ok),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        _status_card("LM Studio / Local", lmstudio_url, False),
        unsafe_allow_html=True,
    )

st.write("")
st.write("")


# --- MCP servers list ---------------------------------------------------

mcp_path = Path("mcp.json")
st.markdown('<div class="tml-label">MCP servers</div>', unsafe_allow_html=True)

if mcp_path.exists():
    st.caption(f"Configured in `{mcp_path}`. Toggle them per-page in Basic Chat.")
else:
    st.info("No `mcp.json` yet — it'll be created in Phase 8 of the plan.")


render_theme_toggle()
```

- [ ] **Step 2: Smoke-check the page renders**

```bash
streamlit run app.py --server.port 8501 &
sleep 4
curl -fs http://localhost:8501/_stcore/health
echo
kill %1 2>/dev/null
```

Expected: `ok` from `/_stcore/health` and no Python tracebacks in the foreground output. (You can also open `http://localhost:8501` in a browser to eyeball the brand treatment.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "$(cat <<'EOF'
feat(home): brand-themed Home page with provider status and MCP placeholder

Hero with editorial Fraunces title and italic emerald accent. Three provider
status cards (Anthropic, OpenAI, LM Studio) with connection state derived
from env vars. Theme toggle in the sidebar. MCP section is a placeholder
until Phase 8 wires up mcp.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Configuration loaders (providers + MCP)

### Task 2.1: Create `providers.toml` and the loader

**Files:**
- Create: `providers.toml`
- Create: `playground/providers/__init__.py` (empty for now)
- Create: `tests/test_providers_config.py`

- [ ] **Step 1: Write `providers.toml`**

```toml
# Provider catalog. Edit this file to add/remove models or change defaults.
# Secrets (API keys, base URLs) live in .env.

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
# Models discovered at runtime from /v1/models — empty static list.
models = []
default_model = ""
default_max_tokens = 2048
default_temperature = 0.7
capabilities = ["tools", "streaming"]
```

- [ ] **Step 2: Create empty `playground/providers/__init__.py`**

```bash
mkdir -p playground/providers
touch playground/providers/__init__.py
```

- [ ] **Step 3: Write the failing test for the config loader**

`tests/test_providers_config.py`:

```python
"""Tests for the providers.toml loader."""

from pathlib import Path

import pytest

from playground.providers.config import (
    ProviderConfig,
    load_providers_config,
    UnknownProviderError,
)


def test_load_known_anthropic_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "providers.toml"
    cfg_path.write_text(
        """
        [anthropic]
        models = ["claude-sonnet-4-6"]
        default_model = "claude-sonnet-4-6"
        default_max_tokens = 4096
        default_temperature = 1.0
        capabilities = ["tools", "streaming"]
        """
    )
    cfg = load_providers_config(cfg_path)
    assert "anthropic" in cfg
    anthropic = cfg["anthropic"]
    assert isinstance(anthropic, ProviderConfig)
    assert anthropic.default_model == "claude-sonnet-4-6"
    assert anthropic.default_max_tokens == 4096
    assert "streaming" in anthropic.capabilities


def test_unknown_provider_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "providers.toml"
    cfg_path.write_text(
        """
        [made_up_provider]
        models = ["x"]
        default_model = "x"
        default_max_tokens = 1
        default_temperature = 1.0
        capabilities = []
        """
    )
    with pytest.raises(UnknownProviderError):
        load_providers_config(cfg_path)
```

- [ ] **Step 4: Run the test to confirm it fails**

```bash
pytest tests/test_providers_config.py -v
```

Expected: collection error or `ModuleNotFoundError` for `playground.providers.config`.

- [ ] **Step 5: Implement `playground/providers/config.py`**

```python
"""Loader for providers.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

KNOWN_PROVIDERS = {"anthropic", "openai", "lmstudio"}


class UnknownProviderError(ValueError):
    """Raised when providers.toml contains a section we don't recognize."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    models: list[str]
    default_model: str
    default_max_tokens: int
    default_temperature: float
    capabilities: list[str]


def load_providers_config(path: str | Path = "providers.toml") -> dict[str, ProviderConfig]:
    """Load and validate providers.toml. Returns {provider_name: ProviderConfig}."""
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)
    out: dict[str, ProviderConfig] = {}
    for name, section in data.items():
        if name not in KNOWN_PROVIDERS:
            raise UnknownProviderError(
                f"providers.toml has unknown provider {name!r}; "
                f"expected one of {sorted(KNOWN_PROVIDERS)}"
            )
        out[name] = ProviderConfig(
            name=name,
            models=list(section.get("models", [])),
            default_model=section["default_model"],
            default_max_tokens=int(section["default_max_tokens"]),
            default_temperature=float(section["default_temperature"]),
            capabilities=list(section.get("capabilities", [])),
        )
    return out
```

- [ ] **Step 6: Run the test to confirm it passes**

```bash
pytest tests/test_providers_config.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Sanity-check loading the real file**

```bash
python -c "from playground.providers.config import load_providers_config; \
import json; print(json.dumps({k: v.__dict__ for k,v in load_providers_config().items()}, indent=2))"
```

Expected: pretty-printed dict with all three providers.

- [ ] **Step 8: Commit**

```bash
git add providers.toml playground/providers/ tests/test_providers_config.py
git commit -m "$(cat <<'EOF'
feat(providers): config loader for providers.toml

ProviderConfig dataclass and load_providers_config() validate the catalog.
Unknown sections raise UnknownProviderError so typos in the toml fail loud.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.2: Create `mcp.json` and the loader

**Files:**
- Create: `mcp.json`
- Create: `playground/mcp/__init__.py` (empty)
- Create: `playground/mcp/config.py`
- Create: `tests/test_mcp_config.py`

- [ ] **Step 1: Write `mcp.json`** (notes server is bundled in Phase 8 — disabled until then)

```json
{
  "mcpServers": {
    "notes": {
      "command": "python",
      "args": ["mcp_servers/notes/server.py"],
      "description": "Bundled — agent scratch notes, persists to disk",
      "enabled": false
    }
  }
}
```

- [ ] **Step 2: Create empty `playground/mcp/__init__.py`**

```bash
mkdir -p playground/mcp
touch playground/mcp/__init__.py
```

- [ ] **Step 3: Write the failing test**

`tests/test_mcp_config.py`:

```python
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
```

- [ ] **Step 4: Run the test, confirm it fails**

```bash
pytest tests/test_mcp_config.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement `playground/mcp/config.py`**

```python
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
```

- [ ] **Step 6: Run the test, confirm it passes**

```bash
pytest tests/test_mcp_config.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add mcp.json playground/mcp/__init__.py playground/mcp/config.py tests/test_mcp_config.py
git commit -m "$(cat <<'EOF'
feat(mcp): mcp.json loader (Claude Desktop / Code compatible format)

MCPServerConfig dataclass and load_mcp_config() parse the standard format
plus our two extras (description, enabled). Missing file returns empty.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Provider abstraction

### Task 3.1: Define the `LLMClient` protocol and shared types

**Files:**
- Create: `playground/providers/base.py`

- [ ] **Step 1: Write `playground/providers/base.py`**

```python
"""LLMClient protocol and shared types.

The protocol is intentionally narrow: stream a chat response with optional
tools, yield typed events the UI can consume without branching on provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


# --- Message types (Anthropic-native shape) ----------------------------

@dataclass
class TextBlock:
    type: Literal["text"]
    text: str


@dataclass
class ToolUseBlock:
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    source: dict[str, str] = field(default_factory=dict)  # {"kind": "local"|"mcp"|"builtin", ...}


@dataclass
class ToolResultBlock:
    type: Literal["tool_result"]
    tool_use_id: str
    content: list[dict[str, Any]]   # always list of {"type": "text", "text": ...}
    is_error: bool = False
    duration_ms: int | None = None


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


# --- Tool definitions passed to providers -------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]   # JSON Schema


# --- Stream events the UI consumes -------------------------------------

@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    """Provider has started emitting a tool call. Multiple deltas may arrive
    before ToolCallComplete; UI accumulates them. For v1 we render once
    complete, so deltas can simply update an in-flight buffer."""
    id: str
    name: str
    partial_input_json: str   # may be empty until the final delta


@dataclass
class ToolCallComplete:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class MessageComplete:
    usage: Usage
    stop_reason: str = ""   # "end_turn" | "tool_use" | "max_tokens" | provider-specific


StreamEvent = TextDelta | ToolCallDelta | ToolCallComplete | MessageComplete


# --- The protocol -------------------------------------------------------

class LLMClient(Protocol):
    """All three provider implementations satisfy this protocol."""

    name: str
    model: str

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        """Yield events as the model streams a response."""
        ...
```

- [ ] **Step 2: Sanity-check the module imports**

```bash
python -c "from playground.providers.base import (
    LLMClient, ChatMessage, TextBlock, ToolUseBlock, ToolResultBlock,
    ToolDefinition, TextDelta, ToolCallComplete, MessageComplete, Usage
); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add playground/providers/base.py
git commit -m "$(cat <<'EOF'
feat(providers): LLMClient protocol + shared message and event types

Anthropic-native content blocks (text, tool_use, tool_result) plus a tagged
union of stream events (TextDelta, ToolCallDelta/Complete, MessageComplete
with Usage) that the UI consumes without branching on provider.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.2: Implement `AnthropicClient`

**Files:**
- Create: `playground/providers/anthropic_client.py`
- Create: `tests/test_providers.py` (Anthropic section first; OpenAI + LMStudio added in later tasks)
- Create: `tests/fixtures/anthropic_basic_response.jsonl`

- [ ] **Step 1: Capture a recorded Anthropic streaming response (one short turn, no tools)**

`tests/fixtures/anthropic_basic_response.jsonl` — these are the SSE events as they'd arrive from the API. (The implementer may instead record from a real call once and replace this file; the line shapes below are sufficient for the test.)

```jsonl
{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}
{"type":"content_block_stop","index":0}
{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}
{"type":"message_stop"}
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_providers.py` (file may not exist yet — create it):

```python
"""Tests for provider client implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.providers.base import (
    ChatMessage,
    MessageComplete,
    TextBlock,
    TextDelta,
)


# ---------------- Anthropic ----------------

def _replay_anthropic_stream(events_file: Path):
    """Yield event dicts as the anthropic SDK's stream iterator would yield
    raw events. The AnthropicClient adapter translates these into our
    StreamEvent types."""
    with events_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def test_anthropic_client_basic_stream(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers import anthropic_client as ac

    events = list(_replay_anthropic_stream(fixtures_dir / "anthropic_basic_response.jsonl"))

    class _FakeStream:
        def __init__(self, evs): self._evs = evs
        def __enter__(self): return iter(self._evs)
        def __exit__(self, *a): pass

    class _FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream(events)

    class _FakeAnthropic:
        def __init__(self, **kwargs): self.messages = _FakeMessages()

    monkeypatch.setattr(ac, "Anthropic", _FakeAnthropic)

    client = ac.AnthropicClient(model="claude-sonnet-4-6", api_key="test")
    out = list(
        client.stream_chat(
            messages=[ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=1.0,
        )
    )

    deltas = [e for e in out if isinstance(e, TextDelta)]
    completes = [e for e in out if isinstance(e, MessageComplete)]
    assert "".join(d.text for d in deltas) == "Hello there"
    assert len(completes) == 1
    assert completes[0].usage.output_tokens == 7
    assert completes[0].stop_reason == "end_turn"
```

- [ ] **Step 3: Run the test, confirm it fails**

```bash
pytest tests/test_providers.py -v
```

Expected: `ModuleNotFoundError: No module named 'playground.providers.anthropic_client'`.

- [ ] **Step 4: Implement `playground/providers/anthropic_client.py`**

```python
"""Anthropic provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic

from playground.providers.base import (
    ChatMessage,
    LLMClient,
    MessageComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        api_messages = [_to_anthropic_message(m) for m in messages]
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        usage = Usage(input_tokens=0, output_tokens=0)
        stop_reason = ""
        in_flight_tools: dict[int, dict[str, Any]] = {}  # block_index → {id, name, partial}

        with self._client.messages.stream(**kwargs) as stream:
            for ev in stream:
                etype = ev["type"] if isinstance(ev, dict) else getattr(ev, "type", None)
                ev_dict = ev if isinstance(ev, dict) else _to_dict(ev)

                if etype == "message_start":
                    u = ev_dict.get("message", {}).get("usage", {})
                    usage.input_tokens = u.get("input_tokens", 0)
                    usage.cache_read_tokens = u.get("cache_read_input_tokens", 0) or 0
                    usage.cache_creation_tokens = (
                        u.get("cache_creation_input_tokens", 0) or 0
                    )

                elif etype == "content_block_start":
                    block = ev_dict.get("content_block", {})
                    if block.get("type") == "tool_use":
                        in_flight_tools[ev_dict["index"]] = {
                            "id": block["id"],
                            "name": block["name"],
                            "partial": "",
                        }

                elif etype == "content_block_delta":
                    delta = ev_dict.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield TextDelta(text=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        idx = ev_dict["index"]
                        if idx in in_flight_tools:
                            in_flight_tools[idx]["partial"] += delta.get("partial_json", "")
                            yield ToolCallDelta(
                                id=in_flight_tools[idx]["id"],
                                name=in_flight_tools[idx]["name"],
                                partial_input_json=in_flight_tools[idx]["partial"],
                            )

                elif etype == "content_block_stop":
                    idx = ev_dict["index"]
                    if idx in in_flight_tools:
                        t = in_flight_tools.pop(idx)
                        try:
                            parsed = json.loads(t["partial"]) if t["partial"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallComplete(id=t["id"], name=t["name"], input=parsed)

                elif etype == "message_delta":
                    d = ev_dict.get("delta", {})
                    if d.get("stop_reason"):
                        stop_reason = d["stop_reason"]
                    u = ev_dict.get("usage", {})
                    if "output_tokens" in u:
                        usage.output_tokens = u["output_tokens"]

                elif etype == "message_stop":
                    pass

        yield MessageComplete(usage=usage, stop_reason=stop_reason)


def _to_anthropic_message(m: ChatMessage) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": [_block_to_dict(b) for b in m.content],
    }


def _block_to_dict(b: TextBlock | ToolUseBlock | ToolResultBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": b.is_error,
        }
    raise TypeError(f"Unknown block type: {type(b)}")


def _to_dict(ev: Any) -> dict[str, Any]:
    """Best-effort coerce a Pydantic-ish stream event to a plain dict."""
    if hasattr(ev, "model_dump"):
        return ev.model_dump()
    if hasattr(ev, "dict"):
        return ev.dict()
    return dict(ev) if hasattr(ev, "__iter__") else {}
```

- [ ] **Step 5: Run the test, confirm it passes**

```bash
pytest tests/test_providers.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/providers/anthropic_client.py tests/test_providers.py tests/fixtures/anthropic_basic_response.jsonl
git commit -m "$(cat <<'EOF'
feat(providers): AnthropicClient streaming implementation

Translates anthropic.messages.stream() raw events into our typed
StreamEvent union. Handles text deltas, tool-use blocks (start/json deltas/
stop), and final usage. Test replays a recorded SSE-style transcript.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.3: Implement `OpenAIClient`

**Files:**
- Create: `playground/providers/openai_client.py`
- Modify: `tests/test_providers.py` (add OpenAI tests)
- Create: `tests/fixtures/openai_basic_response.jsonl`

- [ ] **Step 1: Capture a recorded OpenAI streaming response**

`tests/fixtures/openai_basic_response.jsonl`:

```jsonl
{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}
{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":7,"total_tokens":17}}
```

- [ ] **Step 2: Write the failing test (append to `tests/test_providers.py`)**

```python
# ---------------- OpenAI ----------------

def test_openai_client_basic_stream(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers import openai_client as oc

    events = []
    with (fixtures_dir / "openai_basic_response.jsonl").open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    class _FakeStream:
        def __init__(self, evs):
            self._evs = [type("Chunk", (), {"model_dump": lambda self, e=e: e})() for e in evs]
        def __iter__(self): return iter(self._evs)

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeStream(events)

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs): self.chat = _FakeChat()

    monkeypatch.setattr(oc, "OpenAI", _FakeOpenAI)

    client = oc.OpenAIClient(model="gpt-4o", api_key="test")
    out = list(
        client.stream_chat(
            messages=[ChatMessage(role="user", content=[TextBlock(type="text", text="hi")])],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=1.0,
        )
    )

    deltas = [e for e in out if isinstance(e, TextDelta)]
    completes = [e for e in out if isinstance(e, MessageComplete)]
    assert "".join(d.text for d in deltas) == "Hello there"
    assert len(completes) == 1
    assert completes[0].usage.output_tokens == 7
    assert completes[0].stop_reason == "stop"
```

- [ ] **Step 3: Run, confirm fails**

```bash
pytest tests/test_providers.py::test_openai_client_basic_stream -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `playground/providers/openai_client.py`**

```python
"""OpenAI provider implementation. Also used by LMStudioClient via custom base_url."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from playground.providers.base import (
    ChatMessage,
    LLMClient,
    MessageComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[StreamEvent]:
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_to_openai_messages(messages))

        api_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            or None
        )

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=api_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = Usage(input_tokens=0, output_tokens=0)
        stop_reason = ""
        # tool_call_id → {"id":..., "name":..., "args":""}
        in_flight: dict[str, dict[str, str]] = {}

        for chunk in stream:
            ev = _maybe_dump(chunk)
            choices = ev.get("choices", [])
            if choices:
                ch = choices[0]
                delta = ch.get("delta", {}) or {}
                if (text := delta.get("content")):
                    yield TextDelta(text=text)
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    key = str(idx)
                    if key not in in_flight:
                        in_flight[key] = {"id": tc.get("id", key), "name": "", "args": ""}
                    fn = tc.get("function", {}) or {}
                    if fn.get("name"):
                        in_flight[key]["name"] = fn["name"]
                    if fn.get("arguments"):
                        in_flight[key]["args"] += fn["arguments"]
                        yield ToolCallDelta(
                            id=in_flight[key]["id"],
                            name=in_flight[key]["name"],
                            partial_input_json=in_flight[key]["args"],
                        )
                if ch.get("finish_reason"):
                    stop_reason = ch["finish_reason"]
                    for t in in_flight.values():
                        try:
                            parsed = json.loads(t["args"]) if t["args"] else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolCallComplete(id=t["id"], name=t["name"], input=parsed)
                    in_flight.clear()

            if (u := ev.get("usage")):
                usage.input_tokens = u.get("prompt_tokens", 0)
                usage.output_tokens = u.get("completion_tokens", 0)

        yield MessageComplete(usage=usage, stop_reason=stop_reason)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            text_parts = [b.text for b in m.content if isinstance(b, TextBlock)]
            tool_results = [b for b in m.content if isinstance(b, ToolResultBlock)]
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": "".join(
                            c.get("text", "") for c in tr.content if c.get("type") == "text"
                        ),
                    }
                )
        else:  # assistant
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in m.content
                if isinstance(b, ToolUseBlock)
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
    return out


def _maybe_dump(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if isinstance(chunk, dict):
        return chunk
    return {}
```

- [ ] **Step 5: Run, confirm passes**

```bash
pytest tests/test_providers.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/providers/openai_client.py tests/test_providers.py tests/fixtures/openai_basic_response.jsonl
git commit -m "$(cat <<'EOF'
feat(providers): OpenAIClient streaming implementation

Translates OpenAI's chat.completions delta events into our StreamEvent
union, including parallel tool-call accumulation across delta chunks.
Supports passing custom base_url so LMStudioClient can subclass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.4: Implement `LMStudioClient`

**Files:**
- Create: `playground/providers/lmstudio_client.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Write `playground/providers/lmstudio_client.py`**

```python
"""LM Studio provider — OpenAI-compatible local endpoint."""

from __future__ import annotations

import os

import httpx

from playground.providers.openai_client import OpenAIClient


class LMStudioClient(OpenAIClient):
    name = "lmstudio"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        # api_key required by SDK but ignored by LM Studio — placeholder is fine.
        super().__init__(model=model, api_key="lm-studio", base_url=url)


def discover_lmstudio_models(base_url: str | None = None, timeout: float = 1.0) -> list[str]:
    """Hit /v1/models to discover what's loaded. Returns [] if unreachable."""
    url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    try:
        resp = httpx.get(f"{url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
```

- [ ] **Step 2: Add tests for discovery + subclass behavior** (append to `tests/test_providers.py`)

```python
# ---------------- LM Studio ----------------

def test_lmstudio_client_subclasses_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://example.invalid:1234/v1")
    from playground.providers.lmstudio_client import LMStudioClient
    from playground.providers.openai_client import OpenAIClient

    c = LMStudioClient(model="local-model")
    assert isinstance(c, OpenAIClient)
    assert c.name == "lmstudio"


def test_discover_lmstudio_models_returns_empty_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers.lmstudio_client import discover_lmstudio_models

    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1/v1")  # unlikely to be live
    models = discover_lmstudio_models(timeout=0.1)
    assert models == []
```

- [ ] **Step 3: Run, confirm passes**

```bash
pytest tests/test_providers.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add playground/providers/lmstudio_client.py tests/test_providers.py
git commit -m "$(cat <<'EOF'
feat(providers): LMStudioClient (OpenAI-compatible) + model discovery

Subclasses OpenAIClient with the LM Studio base URL pinned. Adds
discover_lmstudio_models() that hits /v1/models with a tight timeout and
returns [] when the local server isn't running, so the page can detect
LMStudio liveness without raising.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.5: Provider registry + availability detection

**Files:**
- Create: `playground/providers/registry.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add the failing test (append to `tests/test_providers.py`)**

```python
# ---------------- Registry ----------------

def test_registry_lists_available_providers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers.registry import list_available_providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1/v1")  # unreachable

    avail = list_available_providers(check_lmstudio=False)
    assert "anthropic" in avail
    assert "openai" not in avail
    assert "lmstudio" in avail   # presence of env var, not reachability, when check disabled


def test_registry_get_client_returns_correct_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    from playground.providers.registry import get_client
    from playground.providers.anthropic_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    c = get_client("anthropic", "claude-sonnet-4-6")
    assert isinstance(c, AnthropicClient)
    assert c.model == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest tests/test_providers.py -k registry -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `playground/providers/registry.py`**

```python
"""Provider registry — instantiate clients and detect availability."""

from __future__ import annotations

import os

from playground.providers.anthropic_client import AnthropicClient
from playground.providers.base import LLMClient
from playground.providers.lmstudio_client import LMStudioClient, discover_lmstudio_models
from playground.providers.openai_client import OpenAIClient


def get_client(provider: str, model: str, **overrides) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, **overrides)
    if provider == "openai":
        return OpenAIClient(model=model, **overrides)
    if provider == "lmstudio":
        return LMStudioClient(model=model, **overrides)
    raise ValueError(f"Unknown provider: {provider!r}")


def list_available_providers(check_lmstudio: bool = True) -> list[str]:
    """Return providers that are likely usable right now."""
    out: list[str] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    if os.getenv("LMSTUDIO_BASE_URL"):
        if not check_lmstudio or discover_lmstudio_models(timeout=0.5):
            out.append("lmstudio")
    return out


def list_models(provider: str, static_models: list[str]) -> list[str]:
    """Return models for a provider. For lmstudio, queries /v1/models."""
    if provider == "lmstudio":
        return discover_lmstudio_models() or static_models
    return list(static_models)
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest tests/test_providers.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add playground/providers/registry.py tests/test_providers.py
git commit -m "$(cat <<'EOF'
feat(providers): registry with availability detection and dynamic LMStudio models

get_client() instantiates the right provider; list_available_providers()
filters by env-var presence (and LMStudio reachability when requested);
list_models() queries LM Studio's /v1/models at runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Persistence (conversation save/load)

### Task 4.1: Persistence types and `Conversation` save logic

**Files:**
- Create: `playground/persistence.py`
- Create: `tests/test_persistence.py`
- Create: `tests/fixtures/conversation_v1.json`

- [ ] **Step 1: Create the canonical fixture**

`tests/fixtures/conversation_v1.json`:

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
    "system_prompt": {"source": "library/default.md", "text": "You are a helpful assistant."},
    "tools": {"local": [], "mcp": [], "builtin": []},
    "mcp_servers_enabled": []
  },
  "messages": [
    {
      "role": "user",
      "ts": "2026-05-09T22:30:18Z",
      "content": [{"type": "text", "text": "Hello"}]
    },
    {
      "role": "assistant",
      "ts": "2026-05-09T22:30:21Z",
      "content": [{"type": "text", "text": "Hi there"}],
      "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_tokens": 0}
    }
  ],
  "events": []
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_persistence.py`:

```python
"""Tests for conversation persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from playground.persistence import (
    ConversationStore,
    ConversationSummary,
    SCHEMA_VERSION,
)


def _config(provider: str = "anthropic") -> dict:
    return {
        "provider": provider,
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "temperature": 1.0,
        "system_prompt": {"source": None, "text": "hi"},
        "tools": {"local": [], "mcp": [], "builtin": []},
        "mcp_servers_enabled": [],
    }


def test_new_conversation_writes_file_with_schema(tmp_conversations_root: Path) -> None:
    store = ConversationStore(tmp_conversations_root)
    conv = store.new("basic_chat", _config())
    conv.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                          "content": [{"type": "text", "text": "hi"}]})

    files = list((tmp_conversations_root / "basic_chat").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["page"] == "basic_chat"
    assert data["config"]["provider"] == "anthropic"
    assert len(data["messages"]) == 1


def test_round_trip_canonical_fixture(canonical_conversation: dict, tmp_conversations_root: Path) -> None:
    """Loading a v1 fixture and saving it produces semantically equal JSON."""
    page_dir = tmp_conversations_root / "basic_chat"
    page_dir.mkdir()
    fpath = page_dir / f"{canonical_conversation['id']}.json"
    fpath.write_text(json.dumps(canonical_conversation))

    store = ConversationStore(tmp_conversations_root)
    loaded = store.load(canonical_conversation["id"])
    assert loaded.data["messages"] == canonical_conversation["messages"]
    assert loaded.data["config"] == canonical_conversation["config"]


def test_list_returns_summaries_sorted_descending(tmp_conversations_root: Path) -> None:
    store = ConversationStore(tmp_conversations_root)
    a = store.new("basic_chat", _config())
    a.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                      "content": [{"type": "text", "text": "first"}]})
    b = store.new("basic_chat", _config(provider="openai"))
    b.append_message({"role": "user", "ts": "2026-01-02T00:00:00Z",
                      "content": [{"type": "text", "text": "second"}]})

    summaries = store.list("basic_chat")
    assert len(summaries) == 2
    assert summaries[0].started_at >= summaries[1].started_at
    assert all(isinstance(s, ConversationSummary) for s in summaries)
    assert any(s.first_user_message == "first" for s in summaries)
    assert any(s.first_user_message == "second" for s in summaries)


def test_atomic_write_does_not_leave_partial_file(tmp_conversations_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a write failing mid-flight: original file should be untouched
    if it existed, and no .tmp file should remain."""
    store = ConversationStore(tmp_conversations_root)
    conv = store.new("basic_chat", _config())
    conv.append_message({"role": "user", "ts": "2026-01-01T00:00:00Z",
                          "content": [{"type": "text", "text": "first"}]})
    page_dir = tmp_conversations_root / "basic_chat"

    # Force os.replace to fail
    import os
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        conv.append_message({"role": "user", "ts": "2026-01-01T00:00:01Z",
                              "content": [{"type": "text", "text": "second"}]})

    # Restore so cleanup works
    monkeypatch.setattr(os, "replace", real_replace)

    leftover_tmp = list(page_dir.glob("*.tmp"))
    assert leftover_tmp == [], f"Leftover tmp files: {leftover_tmp}"
```

- [ ] **Step 3: Run, confirm fails**

```bash
pytest tests/test_persistence.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `playground/persistence.py`**

```python
"""Conversation persistence — file-per-conversation JSON, atomic writes."""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
_TS_FILENAME_FMT = "%Y-%m-%dT%H-%M-%S"


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    page: str
    started_at: datetime
    ended_at: datetime | None
    provider: str
    model: str
    message_count: int
    first_user_message: str


class Conversation:
    """An open, in-memory conversation that auto-saves on append."""

    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self.data = data
        self._save()

    @property
    def id(self) -> str:
        return self.data["id"]

    def append_message(self, msg: dict) -> None:
        self.data["messages"].append(msg)
        self._save()

    def add_event(self, event: dict) -> None:
        self.data.setdefault("events", []).append(event)
        self._save()

    def end(self) -> None:
        self.data["ended_at"] = _now_iso()
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)


class ConversationStore:
    def __init__(self, root: str | Path = "conversations") -> None:
        self.root = Path(root)

    def new(self, page: str, config: dict) -> Conversation:
        page_dir = self.root / page
        page_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc)
        short_id = secrets.token_hex(2)
        conv_id = f"{ts.strftime(_TS_FILENAME_FMT)}-{short_id}"
        data = {
            "schema_version": SCHEMA_VERSION,
            "id": conv_id,
            "page": page,
            "started_at": ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "ended_at": None,
            "config": copy.deepcopy(config),
            "messages": [],
            "events": [],
        }
        return Conversation(page_dir / f"{conv_id}.json", data)

    def list(self, page: str | None = None) -> list[ConversationSummary]:
        roots: list[Path] = []
        if page is None:
            if not self.root.exists():
                return []
            roots = [p for p in self.root.iterdir() if p.is_dir()]
        else:
            page_dir = self.root / page
            if page_dir.exists():
                roots = [page_dir]

        out: list[ConversationSummary] = []
        for page_dir in roots:
            for jfile in page_dir.glob("*.json"):
                try:
                    data = json.loads(jfile.read_text())
                except Exception:
                    continue
                first_user = ""
                for m in data.get("messages", []):
                    if m.get("role") == "user":
                        for b in m.get("content", []):
                            if b.get("type") == "text":
                                first_user = b.get("text", "")[:80]
                                break
                        if first_user:
                            break
                started = _parse_iso(data["started_at"])
                ended = _parse_iso(data["ended_at"]) if data.get("ended_at") else None
                out.append(
                    ConversationSummary(
                        id=data["id"],
                        page=data["page"],
                        started_at=started,
                        ended_at=ended,
                        provider=data["config"]["provider"],
                        model=data["config"]["model"],
                        message_count=len(data.get("messages", [])),
                        first_user_message=first_user,
                    )
                )
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    def load(self, conv_id: str) -> Conversation:
        for page_dir in self.root.iterdir() if self.root.exists() else []:
            candidate = page_dir / f"{conv_id}.json"
            if candidate.exists():
                return Conversation(candidate, json.loads(candidate.read_text()))
        raise FileNotFoundError(f"No conversation with id {conv_id!r}")


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(Z|[+-]\d{2}:?\d{2})?$")


def _parse_iso(s: str) -> datetime:
    m = _ISO_RE.match(s)
    if not m:
        raise ValueError(f"Bad ISO timestamp: {s!r}")
    base = datetime.fromisoformat(m.group(1))
    return base.replace(tzinfo=timezone.utc)
```

- [ ] **Step 5: Run, confirm all 4 pass**

```bash
pytest tests/test_persistence.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/persistence.py tests/test_persistence.py tests/fixtures/conversation_v1.json
git commit -m "$(cat <<'EOF'
feat(persistence): Conversation + ConversationStore with atomic writes

Schema v1, ISO timestamp + 4-char hex id filenames, atomic temp-file
+ rename on every append. ConversationStore.list() returns lightweight
summaries sorted newest-first; load() resolves by id across pages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Basic Chat page (no tools yet)

### Task 5.1: Skeleton chat page with provider/model dropdowns and streaming

**Files:**
- Create: `pages/1_Basic_Chat.py`
- Create: `playground/chat_ui.py`

- [ ] **Step 1: Create `playground/chat_ui.py`** with rendering helpers (will grow in later tasks)

```python
"""Reusable Streamlit components for chat rendering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from playground.providers.base import (
    ChatMessage,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    ToolResultBlock,
)


def render_message(msg: ChatMessage) -> None:
    """Render a finalized assistant or user message in the transcript."""
    avatar = "🧑" if msg.role == "user" else "◐"
    with st.chat_message(msg.role, avatar=avatar):
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text:
                    st.markdown(block.text)
            elif isinstance(block, ToolUseBlock):
                # Phase 7 expands this to a collapsible block
                st.caption(f"→ tool call: `{block.name}`")
            elif isinstance(block, ToolResultBlock):
                st.caption(f"← tool result for `{block.tool_use_id}`")


def render_text_stream(events: Iterator[StreamEvent]) -> tuple[str, Any]:
    """Drive st.write_stream from a TextDelta iterator. Returns (full_text, last_event_or_none).

    The last_event_or_none lets callers inspect MessageComplete.usage etc.
    """
    last_non_text: Any = None
    text_buf: list[str] = []

    def _gen() -> Iterator[str]:
        nonlocal last_non_text
        for ev in events:
            if isinstance(ev, TextDelta):
                text_buf.append(ev.text)
                yield ev.text
            else:
                last_non_text = ev

    st.write_stream(_gen())
    return "".join(text_buf), last_non_text
```

- [ ] **Step 2: Create `pages/1_Basic_Chat.py`** (chat-only, no tools)

```python
"""Basic Chat — multi-provider streaming chat, no tools yet."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from playground.branding import (
    inject_brand_css,
    render_brand_wordmark,
    render_theme_toggle,
)
from playground.chat_ui import render_message, render_text_stream
from playground.persistence import ConversationStore
from playground.providers.base import ChatMessage, MessageComplete, TextBlock
from playground.providers.config import load_providers_config
from playground.providers.registry import (
    get_client,
    list_available_providers,
    list_models,
)

load_dotenv()

st.set_page_config(
    page_title="Basic Chat — TravisML Playground",
    page_icon="◐",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_brand_css()
render_brand_wordmark()


# ---------------- Sidebar config ----------------

st.sidebar.markdown('<div class="tml-label">Model</div>', unsafe_allow_html=True)

providers_cfg = load_providers_config()
available = list_available_providers(check_lmstudio=False)
if not available:
    st.error("No providers available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")
    st.stop()

provider = st.sidebar.selectbox("Provider", available, key="provider")
pcfg = providers_cfg[provider]
models = list_models(provider, pcfg.models) or [pcfg.default_model or ""]
model = st.sidebar.selectbox(
    "Model",
    models,
    index=models.index(pcfg.default_model) if pcfg.default_model in models else 0,
    key="model",
)

st.sidebar.markdown('<div class="tml-label">Sampling</div>', unsafe_allow_html=True)
max_tokens = st.sidebar.number_input(
    "max_tokens", min_value=1, max_value=128_000,
    value=pcfg.default_max_tokens, key="max_tokens",
)
temperature = st.sidebar.slider(
    "temperature", 0.0, 2.0, pcfg.default_temperature, 0.05, key="temperature",
)


# ---------------- Conversation state ----------------

store = ConversationStore()

if "conversation" not in st.session_state or st.session_state.get("conv_provider") != provider \
        or st.session_state.get("conv_model") != model:
    st.session_state.conversation = store.new(
        "basic_chat",
        config={
            "provider": provider,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system_prompt": {"source": None, "text": ""},
            "tools": {"local": [], "mcp": [], "builtin": []},
            "mcp_servers_enabled": [],
        },
    )
    st.session_state.messages = []
    st.session_state.conv_provider = provider
    st.session_state.conv_model = model

conv = st.session_state.conversation
messages: list[ChatMessage] = st.session_state.messages

# ---------------- Transcript ----------------

st.markdown(
    '<h1 style="font-size:36px;margin-bottom:8px;">Basic <em>chat</em></h1>',
    unsafe_allow_html=True,
)
st.caption(f"Conversation `{conv.id}` · provider: `{provider}/{model}`")
st.divider()

for m in messages:
    render_message(m)


# ---------------- Input + send ----------------

if prompt := st.chat_input("Ask anything..."):
    user_msg = ChatMessage(role="user", content=[TextBlock(type="text", text=prompt)])
    messages.append(user_msg)
    conv.append_message({
        "role": "user",
        "ts": _now_iso(),
        "content": [{"type": "text", "text": prompt}],
    })
    render_message(user_msg)

    with st.chat_message("assistant", avatar="◐"):
        client = get_client(provider, model)
        events = client.stream_chat(
            messages=messages,
            system=None,
            tools=[],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        full_text, last = render_text_stream(events)

    asst_msg = ChatMessage(role="assistant", content=[TextBlock(type="text", text=full_text)])
    messages.append(asst_msg)
    save_msg = {
        "role": "assistant",
        "ts": _now_iso(),
        "content": [{"type": "text", "text": full_text}],
    }
    if isinstance(last, MessageComplete):
        save_msg["usage"] = {
            "input_tokens": last.usage.input_tokens,
            "output_tokens": last.usage.output_tokens,
            "cache_read_tokens": last.usage.cache_read_tokens,
        }
    conv.append_message(save_msg)


render_theme_toggle()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

> **Note:** Python evaluates the `def _now_iso` at module load — Streamlit re-executes the file each interaction, so it'll be defined every run. The forward reference inside the chat handler works because `_now_iso` is at module scope. If a runtime ordering issue surfaces, hoist `_now_iso` above the chat handler.

- [ ] **Step 3: Hoist `_now_iso` to the top of the file** (defensive ordering)

Move the `_now_iso` definition to immediately after the imports section.

- [ ] **Step 4: Smoke check — run a real chat turn against Anthropic**

```bash
streamlit run app.py --server.port 8501 &
sleep 3
echo "Open http://localhost:8501/Basic_Chat — send 'Hello, who are you?'"
echo "Expected: streaming response, conversation file appears in conversations/basic_chat/"
read -r -p "Press Enter when done testing... "
kill %1 2>/dev/null
ls -la conversations/basic_chat/ 2>/dev/null | head -5
```

- [ ] **Step 5: Commit**

```bash
git add pages/ playground/chat_ui.py
git commit -m "$(cat <<'EOF'
feat(chat): Basic Chat page — streaming multi-provider chat with auto-save

Sidebar provider+model dropdowns gate by env-var availability. Sampling
sliders seeded from providers.toml defaults. Streaming via st.write_stream;
each turn persists to conversations/basic_chat/<id>.json with usage stats.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.2: Smoke-test CLI (`python -m playground.smoke`)

**Files:**
- Create: `playground/smoke.py`

- [ ] **Step 1: Write `playground/smoke.py`**

```python
"""Smoke-test CLI — run a single chat turn end-to-end without Streamlit.

Usage:
    python -m playground.smoke --provider anthropic --model claude-sonnet-4-6 \
        --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from playground.providers.base import ChatMessage, MessageComplete, TextBlock, TextDelta
from playground.providers.config import load_providers_config
from playground.providers.registry import get_client


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="playground.smoke")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "lmstudio"])
    parser.add_argument("--model", default=None, help="Defaults to providers.toml default_model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_providers_config()
    pcfg = cfg[args.provider]
    model = args.model or pcfg.default_model
    if not model:
        parser.error(f"--model required for {args.provider} (no default in providers.toml)")

    client = get_client(args.provider, model)
    events = client.stream_chat(
        messages=[ChatMessage(role="user", content=[TextBlock(type="text", text=args.prompt)])],
        system=args.system,
        tools=[],
        max_tokens=args.max_tokens or pcfg.default_max_tokens,
        temperature=args.temperature if args.temperature is not None else pcfg.default_temperature,
    )

    text_parts: list[str] = []
    final: MessageComplete | None = None
    for ev in events:
        if isinstance(ev, TextDelta):
            sys.stdout.write(ev.text)
            sys.stdout.flush()
            text_parts.append(ev.text)
        elif isinstance(ev, MessageComplete):
            final = ev
    sys.stdout.write("\n")

    if final:
        sys.stderr.write(
            f"[stop_reason={final.stop_reason} "
            f"in={final.usage.input_tokens} out={final.usage.output_tokens}]\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-check the CLI works** (requires `ANTHROPIC_API_KEY` in `.env`)

```bash
python -m playground.smoke --provider anthropic --prompt "Say hi in 5 words"
```

Expected: streaming text output, then a `[stop_reason=end_turn ...]` line on stderr.

- [ ] **Step 3: Commit**

```bash
git add playground/smoke.py
git commit -m "$(cat <<'EOF'
feat(smoke): python -m playground.smoke CLI for non-GUI verification

Runs a single chat turn against any provider, streams text to stdout, prints
usage on stderr. Useful for foundation checks without launching Streamlit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — System prompt editor + library

### Task 6.1: Prompt loader

**Files:**
- Create: `playground/prompts/__init__.py`
- Create: `playground/prompts/loader.py`
- Create: `playground/prompts/library/default.md`
- Create: `tests/test_prompts_loader.py`

- [ ] **Step 1: Create the package skeleton + the default prompt**

```bash
mkdir -p playground/prompts/library
touch playground/prompts/__init__.py
cat > playground/prompts/library/default.md <<'EOF'
You are a helpful, concise assistant running inside the TravisML Agent
Playground. When tools are available, use them when they make the answer
better; otherwise just answer directly.
EOF
```

- [ ] **Step 2: Write the failing test**

`tests/test_prompts_loader.py`:

```python
"""Tests for the prompt library loader."""

from pathlib import Path

import pytest

from playground.prompts.loader import (
    PromptNotFoundError,
    list_prompts,
    load_prompt,
)


def test_list_prompts_includes_default(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "default.md").write_text("hello")
    (lib / "other.md").write_text("world")
    names = list_prompts(library_dir=lib)
    assert sorted(names) == ["default", "other"]


def test_load_prompt_returns_text(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "default.md").write_text("system text\n")
    text = load_prompt("default", library_dir=lib)
    assert text == "system text"   # trimmed


def test_unknown_prompt_raises(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    with pytest.raises(PromptNotFoundError):
        load_prompt("nope", library_dir=lib)
```

- [ ] **Step 3: Run, confirm fails**

```bash
pytest tests/test_prompts_loader.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `playground/prompts/loader.py`**

```python
"""Loader for the on-disk prompt library."""

from __future__ import annotations

from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).parent / "library"


class PromptNotFoundError(KeyError):
    """Raised when load_prompt() can't find the named prompt."""


def list_prompts(library_dir: str | Path = DEFAULT_LIBRARY) -> list[str]:
    p = Path(library_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.md"))


def load_prompt(name: str, library_dir: str | Path = DEFAULT_LIBRARY) -> str:
    p = Path(library_dir) / f"{name}.md"
    if not p.exists():
        raise PromptNotFoundError(name)
    return p.read_text().strip()
```

- [ ] **Step 5: Run, confirm passes**

```bash
pytest tests/test_prompts_loader.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/prompts/ tests/test_prompts_loader.py
git commit -m "$(cat <<'EOF'
feat(prompts): library loader + default system prompt

Reads markdown files from playground/prompts/library/. PromptNotFoundError
keeps lookup failures explicit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6.2: Wire system prompt editor into Basic Chat

**Files:**
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Add system prompt sidebar + pass it to `stream_chat`**

In `pages/1_Basic_Chat.py`:

After the sampling sidebar, before "Conversation state":

```python
# ---------------- System prompt ----------------

from playground.prompts.loader import list_prompts, load_prompt

st.sidebar.markdown('<div class="tml-label">System prompt</div>', unsafe_allow_html=True)
prompts_available = ["(none)"] + list_prompts()
prompt_choice = st.sidebar.selectbox(
    "Load from library",
    prompts_available,
    key="prompt_choice",
)
default_text = (
    "" if prompt_choice == "(none)" else load_prompt(prompt_choice)
)
system_prompt = st.sidebar.text_area(
    "System prompt", value=default_text, height=180, key="system_prompt_text",
)
```

In the `stream_chat` call below, replace `system=None` with `system=system_prompt or None`.

In the `system_prompt` field of the conversation `config`, set `"text": system_prompt or "", "source": prompt_choice if prompt_choice != "(none)" else None`.

- [ ] **Step 2: Smoke check**

```bash
streamlit run app.py --server.port 8501 &
sleep 3
echo "Open Basic Chat. Pick 'default' from the System prompt dropdown."
echo "Edit the textarea. Send a message. Expected: response reflects the prompt."
read -r -p "Press Enter when done... "
kill %1 2>/dev/null
```

- [ ] **Step 3: Commit**

```bash
git add pages/1_Basic_Chat.py
git commit -m "$(cat <<'EOF'
feat(chat): system prompt editor with library dropdown

Sidebar shows '(none)' or any *.md prompt under playground/prompts/library/.
Selecting one seeds the textarea; edits flow into stream_chat() and the
saved conversation config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7 — Local tools + tool-call rendering

### Task 7.1: `@register_tool` decorator + schema inference

**Files:**
- Create: `playground/tools/__init__.py`
- Create: `playground/tools/examples/__init__.py`
- Create: `playground/tools/examples/echo.py`
- Create: `playground/tools/examples/get_current_time.py`
- Create: `tests/test_tools_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tools_registry.py`:

```python
"""Tests for the local tool registry."""

from __future__ import annotations

import pytest


def test_register_tool_infers_schema_from_signature_and_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
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
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest tests/test_tools_registry.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `playground/tools/__init__.py`**

```python
"""Local tool registry — @register_tool decorator + schema inference."""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

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
    if t is int: return "integer"
    if t is float: return "number"
    if t is bool: return "boolean"
    if t is list: return "array"
    if t is dict: return "object"
    return "string"
```

- [ ] **Step 4: Implement examples**

`playground/tools/examples/echo.py`:

```python
"""Echo example tool."""

from __future__ import annotations

from playground.tools import register_tool


@register_tool
def echo(text: str) -> str:
    """Echo the input text back unchanged."""
    return text
```

`playground/tools/examples/get_current_time.py`:

```python
"""Current-time example tool."""

from __future__ import annotations

from datetime import datetime, timezone

from playground.tools import register_tool


@register_tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current time as an ISO 8601 string. Timezone is informational only."""
    return datetime.now(timezone.utc).isoformat() + f" ({timezone_name})"
```

`playground/tools/examples/__init__.py`:

```python
"""Importing this package registers all example tools."""

from playground.tools.examples import echo, get_current_time   # noqa: F401
```

- [ ] **Step 5: Run, confirm passes**

```bash
pytest tests/test_tools_registry.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add playground/tools/ tests/test_tools_registry.py
git commit -m "$(cat <<'EOF'
feat(tools): @register_tool decorator with schema inference + 2 examples

Decorator inspects function signature and docstring to build a JSON Schema
suitable for any provider. Two example tools (echo, get_current_time)
ship in playground/tools/examples/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7.2: Tool-call rendering + tool-use loop

**Files:**
- Modify: `playground/chat_ui.py`
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Add `render_tool_call_block` and a streaming helper that handles tool calls in `playground/chat_ui.py`**

Append to `playground/chat_ui.py`:

```python
import json
import time
from collections.abc import Callable
from typing import Any

from playground.providers.base import (
    ToolCallComplete,
    MessageComplete,
)


def render_tool_call_block(
    *,
    name: str,
    source: dict[str, str],
    input: dict[str, Any],
    result_text: str | None,
    duration_ms: int | None,
    is_error: bool,
) -> None:
    """Collapsible block showing a tool call's name, source, input, output."""
    src_label = source.get("kind", "?")
    if src_label == "mcp":
        src_label = f"mcp/{source.get('server', '?')}"
    head = f"⚙ {name} · {src_label}"
    if duration_ms is not None:
        head += f" · {duration_ms}ms"
    if is_error:
        head = "⚠ " + head
    with st.expander(head, expanded=False):
        st.markdown("**Input**")
        st.code(json.dumps(input, indent=2, ensure_ascii=False), language="json")
        if result_text is not None:
            st.markdown("**Result**")
            st.code(result_text, language="json")


def stream_assistant_turn(
    client_stream: Callable[[], Any],
    *,
    on_text: Callable[[str], None],
) -> tuple[str, list[ToolCallComplete], MessageComplete | None]:
    """Drive a single assistant streaming turn.

    Returns (final_text, tool_calls, message_complete).
    """
    buf: list[str] = []
    tool_calls: list[ToolCallComplete] = []
    final: MessageComplete | None = None
    for ev in client_stream():
        from playground.providers.base import TextDelta
        if isinstance(ev, TextDelta):
            buf.append(ev.text)
            on_text(ev.text)
        elif isinstance(ev, ToolCallComplete):
            tool_calls.append(ev)
        elif isinstance(ev, MessageComplete):
            final = ev
    return "".join(buf), tool_calls, final
```

- [ ] **Step 2: Modify `pages/1_Basic_Chat.py` to load examples and run the tool-use loop**

Near the top of the file, after imports, add:

```python
import playground.tools.examples  # registers echo, get_current_time  # noqa: F401
from playground.tools import call_local_tool, get_local_tools
```

In the sidebar, after the system prompt block, add:

```python
# ---------------- Local tools ----------------

st.sidebar.markdown('<div class="tml-label">Local tools</div>', unsafe_allow_html=True)
local_tool_defs = get_local_tools()
enabled_local: list[str] = st.sidebar.multiselect(
    "Enabled",
    [t.name for t in local_tool_defs],
    default=[t.name for t in local_tool_defs],
    key="enabled_local_tools",
)
active_tools = [t for t in local_tool_defs if t.name in enabled_local]
```

Replace the chat handler with the tool-use loop:

```python
if prompt := st.chat_input("Ask anything..."):
    user_msg = ChatMessage(role="user", content=[TextBlock(type="text", text=prompt)])
    messages.append(user_msg)
    conv.append_message({
        "role": "user",
        "ts": _now_iso(),
        "content": [{"type": "text", "text": prompt}],
    })
    render_message(user_msg)

    MAX_ITERS = 10
    for _ in range(MAX_ITERS):
        with st.chat_message("assistant", avatar="◐"):
            text_box = st.empty()
            text_buf: list[str] = []

            def _on_text(t: str) -> None:
                text_buf.append(t)
                text_box.markdown("".join(text_buf))

            client = get_client(provider, model)
            full_text, tool_calls, final = stream_assistant_turn(
                lambda: client.stream_chat(
                    messages=messages,
                    system=system_prompt or None,
                    tools=active_tools,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                ),
                on_text=_on_text,
            )

            content_blocks: list = []
            if full_text:
                content_blocks.append(TextBlock(type="text", text=full_text))
            for tc in tool_calls:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use", id=tc.id, name=tc.name, input=tc.input,
                        source={"kind": "local"},
                    )
                )
            asst_msg = ChatMessage(role="assistant", content=content_blocks)
            messages.append(asst_msg)
            save_msg = {
                "role": "assistant",
                "ts": _now_iso(),
                "content": [
                    ({"type": "text", "text": b.text} if isinstance(b, TextBlock)
                     else {"type": "tool_use", "id": b.id, "name": b.name,
                           "input": b.input, "source": b.source})
                    for b in content_blocks
                ],
            }
            if final:
                save_msg["usage"] = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cache_read_tokens": final.usage.cache_read_tokens,
                }
            conv.append_message(save_msg)

            if not tool_calls:
                break

            tool_result_blocks = []
            for tc in tool_calls:
                t0 = time.time()
                is_err = False
                try:
                    out = call_local_tool(tc.name, tc.input)
                    out_text = out if isinstance(out, str) else json.dumps(out)
                except Exception as e:
                    out_text = f"{type(e).__name__}: {e}"
                    is_err = True
                duration_ms = int((time.time() - t0) * 1000)
                render_tool_call_block(
                    name=tc.name, source={"kind": "local"}, input=tc.input,
                    result_text=out_text, duration_ms=duration_ms, is_error=is_err,
                )
                tool_result_blocks.append(
                    ToolResultBlock(
                        type="tool_result", tool_use_id=tc.id,
                        content=[{"type": "text", "text": out_text}],
                        is_error=is_err, duration_ms=duration_ms,
                    )
                )

            tr_msg = ChatMessage(role="user", content=tool_result_blocks)
            messages.append(tr_msg)
            conv.append_message({
                "role": "user",
                "ts": _now_iso(),
                "content": [
                    {"type": "tool_result", "tool_use_id": b.tool_use_id,
                     "content": b.content, "is_error": b.is_error,
                     "duration_ms": b.duration_ms}
                    for b in tool_result_blocks
                ],
            })
```

Also update the imports at the top of the file to include the new types/helpers:

```python
import json
import time
from playground.chat_ui import (
    render_message, render_text_stream, render_tool_call_block, stream_assistant_turn,
)
from playground.providers.base import (
    ChatMessage, MessageComplete, TextBlock, ToolUseBlock, ToolResultBlock,
)
```

And update the conversation `config.tools.local` field to capture `[t.name for t in active_tools]` when the conversation is created (re-create on config change, as the page already does).

- [ ] **Step 3: Smoke check — local tool call**

```bash
streamlit run app.py --server.port 8501 &
sleep 3
echo "Open Basic Chat. Send: 'What time is it right now?'"
echo "Expected: assistant calls get_current_time, expander shows JSON input/output, then a final answer."
read -r -p "Press Enter when done... "
kill %1 2>/dev/null
```

- [ ] **Step 4: Commit**

```bash
git add playground/chat_ui.py pages/1_Basic_Chat.py
git commit -m "$(cat <<'EOF'
feat(chat): tool-use loop + collapsible tool-call rendering

Iterates up to 10 turns; each tool_use block dispatches to the local
registry, captures result + duration, renders an inline expander with
JSON input/output, and feeds tool_result back to the model. All blocks
are persisted to the conversation file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 8 — MCP integration (tools)

### Task 8.1: Bundled `notes` MCP server

**Files:**
- Create: `mcp_servers/__init__.py` (or just the `notes/` subdir; no package init needed if not imported)
- Create: `mcp_servers/notes/server.py`
- Create: `mcp_servers/notes/README.md`
- Create: `mcp_servers/README.md`

- [ ] **Step 1: Write `mcp_servers/notes/server.py`**

```python
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
```

- [ ] **Step 2: Write `mcp_servers/notes/README.md`**

```markdown
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
```

- [ ] **Step 3: Write `mcp_servers/README.md`**

```markdown
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
```

- [ ] **Step 4: Smoke-check the server starts and lists tools**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_servers/notes/server.py 2>/dev/null | head -1
```

(The exact stdio-protocol smoke depends on `mcp` SDK behaviour — at minimum, ensure `python mcp_servers/notes/server.py` doesn't crash. You can also run the manual UI smoke at the end of Phase 8 instead.)

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/
git commit -m "$(cat <<'EOF'
feat(mcp): bundled notes MCP server (list/save/delete) + READMEs

Persists to ~/.travisml-playground/notes.json. Doubles as a template for
writing your own — see mcp_servers/README.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8.2: MCP bridge — schema translation

**Files:**
- Create: `playground/mcp/bridge.py`
- Create: `tests/test_mcp_bridge.py`

- [ ] **Step 1: Write the failing test**

`tests/test_mcp_bridge.py`:

```python
"""Tests for the MCP ↔ provider tool-format bridge."""

from playground.mcp.bridge import mcp_tool_to_provider_format
from playground.providers.base import ToolDefinition


def test_anthropic_format_passthrough():
    td = ToolDefinition(
        name="save_note",
        description="Save a note.",
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    out = mcp_tool_to_provider_format(td, provider="anthropic")
    assert out == {
        "name": "save_note",
        "description": "Save a note.",
        "input_schema": td.input_schema,
    }


def test_openai_format_wraps_in_function_envelope():
    td = ToolDefinition(
        name="save_note",
        description="Save a note.",
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    out = mcp_tool_to_provider_format(td, provider="openai")
    assert out == {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note.",
            "parameters": td.input_schema,
        },
    }
```

- [ ] **Step 2: Run, confirm fails**

```bash
pytest tests/test_mcp_bridge.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `playground/mcp/bridge.py`**

```python
"""MCP tool schema ↔ provider tool format conversion."""

from __future__ import annotations

from typing import Any

from playground.providers.base import ToolDefinition


def mcp_tool_to_provider_format(td: ToolDefinition, provider: str) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "name": td.name,
            "description": td.description,
            "input_schema": td.input_schema,
        }
    if provider in ("openai", "lmstudio"):
        return {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.input_schema,
            },
        }
    raise ValueError(f"Unsupported provider for tool conversion: {provider!r}")


def mcp_tools_to_provider_format(tds: list[ToolDefinition], provider: str) -> list[dict[str, Any]]:
    return [mcp_tool_to_provider_format(t, provider) for t in tds]
```

- [ ] **Step 4: Run, confirm passes**

```bash
pytest tests/test_mcp_bridge.py -v
```

Expected: 2 passed.

> **Note:** The provider clients in Phase 3 already build their own provider-format from `ToolDefinition`. The bridge module exists so we have one canonical place when MCP tools and local tools both need a unified path; the providers can call into it later if desirable.

- [ ] **Step 5: Commit**

```bash
git add playground/mcp/bridge.py tests/test_mcp_bridge.py
git commit -m "$(cat <<'EOF'
feat(mcp): provider-format bridge for tool schemas

Anthropic uses the bare {name, description, input_schema} shape; OpenAI/
LMStudio wrap in {type:function, function:{name, parameters}}. Centralizing
the translation keeps provider clients simple.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8.3: `MCPClientPool` — manage MCP server connections

**Files:**
- Create: `playground/mcp/client.py`

> **Implementation note:** The `mcp` Python SDK uses asyncio for stdio
> transport. Streamlit's execution model is synchronous. To bridge: the
> pool runs a dedicated asyncio event loop in a background thread; sync
> wrappers submit coroutines and `await` them via `asyncio.run_coroutine_threadsafe`.

- [ ] **Step 1: Write `playground/mcp/client.py`**

```python
"""Manage connections to MCP servers — sync façade over async stdio clients."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from playground.mcp.config import MCPServerConfig
from playground.providers.base import ToolDefinition


@dataclass
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


@dataclass
class MCPPrompt:
    server: str
    name: str
    description: str
    arguments: list[dict[str, Any]]


@dataclass
class MCPResource:
    server: str
    uri: str
    name: str
    description: str
    mime_type: str


class MCPClientPool:
    """One async loop in a background thread; sync façade for Streamlit."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._stack: AsyncExitStack | None = None
        self._configs: dict[str, MCPServerConfig] = {}

    # ---------- lifecycle ----------

    def start(self, servers: dict[str, MCPServerConfig]) -> None:
        if self._loop is not None:
            return
        self._configs = {n: c for n, c in servers.items() if c.enabled}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-pool",
        )
        self._thread.start()
        self._submit(self._open_all()).result(timeout=30)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        try:
            self._submit(self._close_all()).result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._sessions.clear()

    async def _open_all(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cfg in self._configs.items():
            params = StdioServerParameters(
                command=cfg.command, args=list(cfg.args), env={**cfg.env} or None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session

    async def _close_all(self) -> None:
        if self._stack:
            await self._stack.__aexit__(None, None, None)
            self._stack = None

    # ---------- queries ----------

    def list_tools(self, servers: list[str]) -> list[MCPTool]:
        async def _go() -> list[MCPTool]:
            out: list[MCPTool] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                resp = await self._sessions[name].list_tools()
                for t in resp.tools:
                    out.append(
                        MCPTool(
                            server=name, name=t.name,
                            description=t.description or "",
                            input_schema=t.inputSchema,
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    def list_prompts(self, servers: list[str]) -> list[MCPPrompt]:
        async def _go() -> list[MCPPrompt]:
            out: list[MCPPrompt] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                try:
                    resp = await self._sessions[name].list_prompts()
                except Exception:
                    continue
                for p in resp.prompts:
                    out.append(
                        MCPPrompt(
                            server=name, name=p.name,
                            description=p.description or "",
                            arguments=[a.model_dump() if hasattr(a, "model_dump") else dict(a)
                                       for a in (p.arguments or [])],
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    def list_resources(self, servers: list[str]) -> list[MCPResource]:
        async def _go() -> list[MCPResource]:
            out: list[MCPResource] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                try:
                    resp = await self._sessions[name].list_resources()
                except Exception:
                    continue
                for r in resp.resources:
                    out.append(
                        MCPResource(
                            server=name, uri=str(r.uri),
                            name=r.name or str(r.uri),
                            description=r.description or "",
                            mime_type=r.mimeType or "",
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    # ---------- actions ----------

    def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        async def _go() -> str:
            resp = await self._sessions[server].call_tool(tool, args)
            parts: list[str] = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return self._submit(_go()).result(timeout=60)

    def get_prompt(self, server: str, prompt: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        async def _go() -> list[dict[str, Any]]:
            resp = await self._sessions[server].get_prompt(prompt, args)
            out: list[dict[str, Any]] = []
            for m in resp.messages:
                role = m.role
                content = m.content
                text = content.text if hasattr(content, "text") else ""
                out.append({"role": role, "content": [{"type": "text", "text": text}]})
            return out
        return self._submit(_go()).result(timeout=30)

    def read_resource(self, server: str, uri: str) -> str:
        async def _go() -> str:
            resp = await self._sessions[server].read_resource(uri)
            parts: list[str] = []
            for c in resp.contents:
                if hasattr(c, "text"):
                    parts.append(c.text or "")
                elif hasattr(c, "blob"):
                    parts.append(f"[binary content, {len(c.blob)} bytes]")
            return "\n".join(parts)
        return self._submit(_go()).result(timeout=30)

    # ---------- helpers ----------

    def _submit(self, coro) -> Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
```

- [ ] **Step 2: Sanity-check the module imports**

```bash
python -c "from playground.mcp.client import MCPClientPool, MCPTool, MCPPrompt, MCPResource; print('ok')"
```

Expected: `ok`.

> **Note:** Direct unit-testing of `MCPClientPool` against real subprocesses is brittle; we rely on the manual smoke check at the end of Phase 8 to validate end-to-end. If the implementation hits a snag against the real `mcp` SDK, adjust accordingly — the SDK's API is stable but field names (`tools`, `prompts`, `resources`, `inputSchema`, etc.) can vary by version.

- [ ] **Step 3: Commit**

```bash
git add playground/mcp/client.py
git commit -m "$(cat <<'EOF'
feat(mcp): MCPClientPool — sync façade over async stdio MCP clients

One asyncio loop in a background thread; pool exposes sync list/call
methods that submit coros via run_coroutine_threadsafe. Used by Streamlit
pages without forcing them to be async.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8.4: Wire MCP tools into Basic Chat

**Files:**
- Modify: `mcp.json` (set `notes.enabled = true`)
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Enable the bundled `notes` server in `mcp.json`**

Set `"enabled": true` for `notes` (was `false` in Task 2.2).

- [ ] **Step 2: Modify `pages/1_Basic_Chat.py` to start the pool, list servers, and route tool calls**

Add to imports:

```python
from playground.mcp.client import MCPClientPool, MCPTool
from playground.mcp.config import load_mcp_config
```

After the existing sidebar sections (Local tools), add:

```python
# ---------------- MCP servers ----------------

mcp_servers = load_mcp_config()
if mcp_servers:
    if "mcp_pool" not in st.session_state:
        pool = MCPClientPool()
        try:
            pool.start(mcp_servers)
            st.session_state.mcp_pool = pool
        except Exception as e:
            st.session_state.mcp_pool = None
            st.sidebar.error(f"MCP pool failed to start: {e}")
    pool = st.session_state.get("mcp_pool")

    if pool:
        st.sidebar.markdown('<div class="tml-label">MCP servers</div>', unsafe_allow_html=True)
        enabled_servers: list[str] = []
        for name, cfg in mcp_servers.items():
            label = f"{name} — {cfg.description}" if cfg.description else name
            if st.sidebar.checkbox(label, value=cfg.enabled, key=f"_mcp_{name}"):
                enabled_servers.append(name)

        if st.sidebar.button("Reload mcp.json"):
            pool.shutdown()
            st.session_state.pop("mcp_pool", None)
            st.rerun()

        mcp_tools_meta: list[MCPTool] = pool.list_tools(enabled_servers)
        mcp_tool_defs = [t.to_tool_definition() for t in mcp_tools_meta]
        # Map name → server for dispatch
        st.session_state._mcp_tool_to_server = {t.name: t.server for t in mcp_tools_meta}
    else:
        mcp_tool_defs = []
        enabled_servers = []
else:
    mcp_tool_defs = []
    enabled_servers = []
```

In the `active_tools` computation, append: `active_tools = active_tools + mcp_tool_defs`.

In the tool-call dispatch loop in the chat handler, replace the local-only `call_local_tool(...)` call with:

```python
tool_to_server: dict[str, str] = st.session_state.get("_mcp_tool_to_server", {})
local_names = {t.name for t in get_local_tools()}

for tc in tool_calls:
    t0 = time.time()
    is_err = False
    source = {"kind": "local"}
    try:
        if tc.name in local_names:
            out = call_local_tool(tc.name, tc.input)
            out_text = out if isinstance(out, str) else json.dumps(out)
        elif tc.name in tool_to_server:
            server = tool_to_server[tc.name]
            source = {"kind": "mcp", "server": server}
            out_text = pool.call_tool(server, tc.name, tc.input)
        else:
            out_text = f"Unknown tool: {tc.name}"
            is_err = True
    except Exception as e:
        out_text = f"{type(e).__name__}: {e}"
        is_err = True
    duration_ms = int((time.time() - t0) * 1000)
    render_tool_call_block(
        name=tc.name, source=source, input=tc.input,
        result_text=out_text, duration_ms=duration_ms, is_error=is_err,
    )
    tool_result_blocks.append(
        ToolResultBlock(
            type="tool_result", tool_use_id=tc.id,
            content=[{"type": "text", "text": out_text}],
            is_error=is_err, duration_ms=duration_ms,
        )
    )
```

Also update the `ToolUseBlock` construction in the assistant message: set `source` to `{"kind":"local"}` if local, `{"kind":"mcp","server":server}` if MCP. (Equivalent to `tool_to_server.get(tc.name, "")` — set when building the block.)

Update the conversation `config["tools"]["mcp"]` to capture `[{"server": s, "tools": [t.name for t in mcp_tools_meta if t.server == s]} for s in enabled_servers]` and `config["mcp_servers_enabled"]` accordingly.

- [ ] **Step 3: Smoke check — agent uses notes via MCP**

```bash
streamlit run app.py --server.port 8501 &
sleep 4
echo "Open Basic Chat. Send: 'Save a note titled \"hello\" with content \"world\". Then list notes.'"
echo "Expected: two MCP tool calls (save_note, list_notes), expanders show inputs/outputs,"
echo "the assistant confirms the save and reads back the list."
read -r -p "Press Enter when done... "
kill %1 2>/dev/null
ls -la ~/.travisml-playground/notes.json 2>/dev/null
```

- [ ] **Step 4: Commit**

```bash
git add mcp.json pages/1_Basic_Chat.py
git commit -m "$(cat <<'EOF'
feat(mcp): wire MCPClientPool into Basic Chat — server toggles + dispatch

Sidebar checkboxes per configured server; tool-call loop routes by name
to local registry or pool.call_tool(). Conversation config captures the
enabled servers and their exposed tools at run-time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 9 — MCP prompts

### Task 9.1: Sidebar widget + insertion logic

**Files:**
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: After the MCP-servers block in the sidebar, add the prompts widget**

```python
# ---------------- MCP prompts ----------------

if mcp_servers and pool and enabled_servers:
    mcp_prompts = pool.list_prompts(enabled_servers)
    if mcp_prompts:
        st.sidebar.markdown('<div class="tml-label">MCP prompts</div>', unsafe_allow_html=True)
        prompt_options = {f"{p.server}/{p.name}": p for p in mcp_prompts}
        sel = st.sidebar.selectbox(
            "Prompt", list(prompt_options.keys()), key="_mcp_prompt_sel",
        )
        chosen = prompt_options[sel]
        arg_values: dict = {}
        for arg in chosen.arguments:
            arg_values[arg["name"]] = st.sidebar.text_input(
                f"  arg: {arg['name']}",
                value=arg.get("default", ""),
                help=arg.get("description"),
                key=f"_mcp_prompt_arg_{arg['name']}",
            )

        col1, col2 = st.sidebar.columns(2)
        if col1.button("Use as user message"):
            try:
                msgs = pool.get_prompt(chosen.server, chosen.name, arg_values)
            except Exception as e:
                st.sidebar.error(f"get_prompt failed: {e}")
                msgs = []
            for m in msgs:
                st.session_state.messages.append(
                    ChatMessage(
                        role="user" if m["role"] == "user" else "assistant",
                        content=[TextBlock(type="text", text=m["content"][0]["text"])],
                    )
                )
                conv.append_message({**m, "ts": _now_iso()})
                conv.add_event({
                    "ts": _now_iso(),
                    "type": "prompt_inserted",
                    "server": chosen.server,
                    "prompt": chosen.name,
                    "args": arg_values,
                })
            st.rerun()
        if col2.button("Use as system prompt"):
            try:
                msgs = pool.get_prompt(chosen.server, chosen.name, arg_values)
            except Exception as e:
                st.sidebar.error(f"get_prompt failed: {e}")
                msgs = []
            new_text = "\n\n".join(m["content"][0]["text"] for m in msgs)
            st.session_state.system_prompt_text = new_text
            conv.add_event({
                "ts": _now_iso(),
                "type": "system_prompt_replaced_by_mcp",
                "server": chosen.server, "prompt": chosen.name, "args": arg_values,
            })
            st.rerun()
```

- [ ] **Step 2: Smoke check** — only meaningful if you have an MCP server with prompts. The bundled `notes` server doesn't ship with prompts, so this is best validated by adding a temporary one in development or skipping until you wire up an external server. Manually verify:

```
Sidebar shows "MCP prompts" only when an MCP server with prompts is selected.
Clicking "Use as user message" inserts the rendered prompt into the chat.
```

- [ ] **Step 3: Commit**

```bash
git add pages/1_Basic_Chat.py
git commit -m "$(cat <<'EOF'
feat(mcp): MCP prompts sidebar — insert as user message or system prompt

Lists prompts from all enabled servers, renders argument inputs from the
prompt's argument schema, calls prompts/get and either appends to the
conversation or replaces the system prompt textarea.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 10 — MCP resources

### Task 10.1: Sidebar list + attach + builtin `read_mcp_resource`

**Files:**
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Add a resource sidebar widget after the prompts block**

```python
# ---------------- MCP resources ----------------

attached_resources: list[dict[str, str]] = []  # {"server", "uri", "mime_type"}
if mcp_servers and pool and enabled_servers:
    resources = pool.list_resources(enabled_servers)
    if resources:
        st.sidebar.markdown('<div class="tml-label">MCP resources</div>', unsafe_allow_html=True)
        for r in resources:
            label = f"{r.server}/{r.uri.split('/')[-1] or r.uri}"
            if st.sidebar.checkbox(label, key=f"_mcp_res_{r.server}_{r.uri}"):
                attached_resources.append({"server": r.server, "uri": r.uri, "mime_type": r.mime_type})
        if st.sidebar.button("Refresh resources"):
            st.rerun()
```

- [ ] **Step 2: When sending a user message, prepend resource contents**

In the chat handler, immediately after `if prompt := st.chat_input(...)`:

```python
preamble_blocks: list = []
for ar in attached_resources:
    try:
        content_text = pool.read_resource(ar["server"], ar["uri"])
    except Exception as e:
        content_text = f"[failed to read {ar['uri']}: {e}]"
    preamble_blocks.append(
        TextBlock(
            type="text",
            text=f'<resource uri="{ar["uri"]}" mimeType="{ar["mime_type"]}">\n{content_text}\n</resource>',
        )
    )
    conv.add_event({
        "ts": _now_iso(),
        "type": "resource_attached",
        "server": ar["server"], "uri": ar["uri"],
    })

user_msg = ChatMessage(
    role="user",
    content=preamble_blocks + [TextBlock(type="text", text=prompt)],
)
```

(Adjust the existing user_msg construction to use this pattern.)

- [ ] **Step 3: Add the builtin `read_mcp_resource` tool**

In `pages/1_Basic_Chat.py`, after computing `mcp_tool_defs`, also build a builtin tool when any MCP server is connected:

```python
builtin_tools: list[ToolDefinition] = []
if mcp_servers and pool and enabled_servers:
    builtin_tools.append(
        ToolDefinition(
            name="read_mcp_resource",
            description="Read an MCP resource by URI (use when you see a uri the user mentioned or one returned by another tool).",
            input_schema={
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name"},
                    "uri": {"type": "string", "description": "Resource URI"},
                },
                "required": ["server", "uri"],
            },
        )
    )

active_tools = active_tools + mcp_tool_defs + builtin_tools
```

In the tool dispatch block, add a new branch:

```python
elif tc.name == "read_mcp_resource":
    server = tc.input.get("server", "")
    uri = tc.input.get("uri", "")
    source = {"kind": "builtin"}
    try:
        out_text = pool.read_resource(server, uri)
    except Exception as e:
        out_text = f"{type(e).__name__}: {e}"
        is_err = True
```

Update `config["tools"]["builtin"]` capture to include `"read_mcp_resource"` when MCP is active.

- [ ] **Step 4: Smoke check** — set `notes.enabled=true` and ensure that when a resource is attached, its contents appear in the next prompt. Easier to test against an external MCP server with resources (e.g., the `@modelcontextprotocol/server-filesystem` from npm).

- [ ] **Step 5: Commit**

```bash
git add pages/1_Basic_Chat.py
git commit -m "$(cat <<'EOF'
feat(mcp): MCP resources sidebar + read_mcp_resource builtin tool

Checkbox-attached resources prepend as <resource> blocks before the next
user message. The synthesized builtin lets the LLM pull additional
resources by URI mid-conversation. Both modes get logged as events.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 11 — History sidebar

### Task 11.1: List past conversations + open read-only + fork

**Files:**
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Render history in the sidebar above the theme toggle**

Just before `render_theme_toggle()`, add:

```python
# ---------------- History ----------------

st.sidebar.markdown('<div class="tml-label">History</div>', unsafe_allow_html=True)

if st.sidebar.button("New conversation"):
    st.session_state.pop("conversation", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("loaded_conv_id", None)
    st.session_state.pop("read_only", None)
    st.rerun()

summaries = store.list("basic_chat")
for s in summaries[:20]:
    label = f"{s.started_at:%H:%M}  {s.first_user_message[:36] or '(empty)'}"
    if st.sidebar.button(label, key=f"_hist_{s.id}", use_container_width=True):
        loaded = store.load(s.id)
        st.session_state.messages = _data_to_chat_messages(loaded.data)
        st.session_state.loaded_conv_id = s.id
        st.session_state.read_only = True
        st.rerun()

if st.session_state.get("read_only"):
    st.sidebar.warning("Viewing past conversation (read-only)")
    if st.sidebar.button("Fork from here"):
        forked = store.new(
            "basic_chat",
            config=loaded.data["config"],
        )
        for m in loaded.data["messages"]:
            forked.append_message(m)
        st.session_state.conversation = forked
        st.session_state.read_only = False
        st.session_state.loaded_conv_id = None
        st.rerun()
```

Where `_data_to_chat_messages` is a small helper added to `playground/chat_ui.py`:

```python
def _data_to_chat_messages(conv_data: dict) -> list:
    from playground.providers.base import (
        ChatMessage, TextBlock, ToolUseBlock, ToolResultBlock,
    )
    out = []
    for m in conv_data.get("messages", []):
        blocks: list = []
        for b in m.get("content", []):
            if b.get("type") == "text":
                blocks.append(TextBlock(type="text", text=b["text"]))
            elif b.get("type") == "tool_use":
                blocks.append(ToolUseBlock(
                    type="tool_use", id=b["id"], name=b["name"],
                    input=b.get("input", {}), source=b.get("source", {}),
                ))
            elif b.get("type") == "tool_result":
                blocks.append(ToolResultBlock(
                    type="tool_result", tool_use_id=b["tool_use_id"],
                    content=b.get("content", []), is_error=b.get("is_error", False),
                    duration_ms=b.get("duration_ms"),
                ))
        out.append(ChatMessage(role=m["role"], content=blocks))
    return out
```

In the chat handler, if `st.session_state.get("read_only")`, skip processing of `st.chat_input`.

- [ ] **Step 2: Smoke check**

```bash
streamlit run app.py --server.port 8501 &
sleep 3
echo "1. Send a couple of messages to create some history."
echo "2. Click 'New conversation' — transcript clears."
echo "3. Click a past entry in History — read-only view loads."
echo "4. Click 'Fork from here' — read-only badge clears, you can keep chatting."
read -r -p "Press Enter when done... "
kill %1 2>/dev/null
```

- [ ] **Step 3: Commit**

```bash
git add pages/1_Basic_Chat.py playground/chat_ui.py
git commit -m "$(cat <<'EOF'
feat(chat): History sidebar — load past conversations, fork from any point

ConversationStore.list() drives a clickable list of past runs. Selecting one
loads it in read-only mode; "Fork from here" creates a new conversation
seeded with the loaded transcript.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 12 — Polish

### Task 12.1: README + verify full test suite

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# TravisML Agent Playground

Branded multi-provider agent harness — chat, tools, MCP servers, and
auto-saved transcripts — for experimenting with agentic systems against
Anthropic, OpenAI, and locally-hosted (LM Studio) models.

## Setup

```bash
# 1. Activate the project venv (Python 3.14)
source .agent-playground/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
$EDITOR .env   # set ANTHROPIC_API_KEY / OPENAI_API_KEY / LMSTUDIO_BASE_URL
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

## Spec

Full v1 design: [docs/superpowers/specs/2026-05-09-travisml-agent-playground-design.md](docs/superpowers/specs/2026-05-09-travisml-agent-playground-design.md).
```

- [ ] **Step 2: Run the full test suite green**

```bash
pytest -v
```

Expected: all tests pass (rough count: ~20 tests across 6 test files).

- [ ] **Step 3: Run `ruff` formatting check**

```bash
ruff check .
```

Fix any complaints inline (most likely import ordering and unused imports).

- [ ] **Step 4: Final smoke run**

```bash
streamlit run app.py
# Click through Home → Basic Chat → toggle theme → run a tool-using turn
# → fork from history. Verify saved JSON looks complete.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README with setup, run, smoke-test, and extension instructions

Walks new contributors through venv activation, env config, running the
GUI, the smoke CLI, the test suite, and how to add a local tool or
external MCP server. Links to the v1 spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12.2: Push and confirm

- [ ] **Step 1: Push to origin**

```bash
git push
```

- [ ] **Step 2: Confirm CI-readiness (no CI in v1, but suite passes)**

```bash
pytest && ruff check . && echo "v1 ready"
```

Expected: `v1 ready`.

---

## Self-review

**Spec coverage** — checked each spec section against tasks:

| Spec section | Plan task(s) |
|--------------|--------------|
| §1 Overview / goals | implicit across all phases |
| §2 Tech stack | Task 0.1 (`pyproject.toml`) |
| §3 Folder layout | Task 0.0 (cleanup), 0.3, 1.1, every later task adds files |
| §4 Configuration (`.env`, `providers.toml`, runtime, conversation) | 0.2, 2.1, 4.1, 5.1, 6.2 |
| §5 Branding (light/dark, fonts, wordmark, toggle) | 1.1, 1.2, 1.3 |
| §6 Provider abstraction (protocol + 3 clients + registry) | 3.1, 3.2, 3.3, 3.4, 3.5 |
| §7 Tools + MCP (local, MCP, builtin, bundled server, `mcp.json`) | 2.2, 7.1, 7.2, 8.1, 8.2, 8.3, 8.4, 10.1 |
| §8 Persistence (schema, store, atomic) | 4.1 |
| §9 Pages (Home, Basic Chat) | 1.3, 5.1, 6.2, 7.2, 8.4, 9.1, 10.1, 11.1 |
| §10 Testing (pytest + smoke CLI + manual) | 0.3, 2.1, 2.2, 3.x, 4.1, 5.2, 6.1, 7.1, 8.2, manual smoke checks throughout, 12.1 |
| §11 Build order (9 phases) | Mirrored in plan phases 0–12 (0 = skeleton additions, 12 = polish) |
| §12 Out of scope | Not implemented (correct) |
| §13 Open questions | Not implemented (correct) |

**Placeholder scan** — no "TBD", "TODO", "implement later", etc. Every code step shows the actual code; every command step shows expected output. The two implementation notes (about MCP SDK field names and `mcp.run()` smoke-check) are flagged as such, not as work the engineer needs to invent.

**Type consistency** — verified across phases:

- `LLMClient` protocol signature used identically in 3.1 (definition), 3.2/3.3/3.4 (implementations), 3.5 (registry), 5.1/5.2/7.2 (consumers).
- `ChatMessage` / `TextBlock` / `ToolUseBlock` / `ToolResultBlock` shapes consistent everywhere.
- `Conversation.append_message()` and `add_event()` signatures match between 4.1 and all callers.
- `ConversationSummary` fields (`started_at`, `first_user_message`, etc.) match between 4.1 (definition) and 11.1 (consumer).
- `MCPClientPool.list_tools/list_prompts/list_resources/call_tool/get_prompt/read_resource` signatures match between 8.3 (definition) and 8.4/9.1/10.1 (consumers).
- Conversation JSON `config.tools` shape is `{"local": [...], "mcp": [...], "builtin": [...]}` in 4.1 fixture, 5.1 page, 7.2 page, 8.4 page, 10.1 page — consistent.

**Build-step ordering** — each task's tests and code are independently runnable and committable.

No issues found that need fixing inline.

---

## Execution Handoff

Plan complete and saved to [docs/superpowers/plans/2026-05-10-travisml-agent-playground-v1.md](docs/superpowers/plans/2026-05-10-travisml-agent-playground-v1.md). Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each task lands in this conversation as a clean diff for review before the next task starts. Best for plans this big — keeps the main context lean.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Heavier on context but lets you intervene mid-task.

**Which approach?**
