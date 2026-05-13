# Memory + Dreaming MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bundled MCP server (`mcp_servers/memory/`) and a background "dreamer" daemon that give the TravisML Agent Playground a single persistent agent identity with cross-conversation memory.

**Architecture:** Three processes (Streamlit playground, memory-mcp stdio server, memory-dreamer daemon) share one SQLite database with the `sqlite-vec` extension. Hot path is append-only; the dreamer runs a six-stage cycle that produces a bi-temporal knowledge graph plus speculative hypotheses. Retrieval combines vector top-K with personalized PageRank over a typed link graph (HippoRAG-style).

**Tech Stack:** Python 3.14, `FastMCP` (mcp SDK), SQLite + WAL + `sqlite-vec`, `sentence-transformers` (nomic-embed-text-v1.5), `networkx` (PageRank), `scikit-learn` (clustering), `python-ulid`, existing `playground.providers` registry (vLLM via the `lmstudio` provider routed at `LMSTUDIO_BASE_URL`).

**Spec:** `docs/superpowers/specs/2026-05-11-memory-dreaming-mcp-design.md` (commit `80ecdd9`).

---

## File structure

```text
mcp_servers/memory/
├── __init__.py
├── README.md
├── server.py                          # FastMCP entry: stdio server
├── dreamer.py                         # CLI: python -m mcp_servers.memory.dreamer serve
├── ids.py                             # ULID-prefixed id generators
├── models.py                          # dataclasses (Episode, Fact, Entity, …)
├── config.py                          # dreamer_config helpers (read/write key/json)
├── db/
│   ├── __init__.py
│   ├── connection.py                  # WAL setup, sqlite-vec loading, advisory lock
│   ├── migrations.py                  # migration runner
│   └── migrations/
│       └── 001_initial.sql
├── repo/
│   ├── __init__.py
│   ├── raw_turns.py
│   ├── episodes.py
│   ├── entities.py
│   ├── facts.py                       # bi-temporal supersession lives here
│   ├── reflections.py
│   ├── hypotheses.py
│   ├── links.py
│   └── dream_runs.py
├── embeddings/
│   ├── __init__.py
│   ├── base.py                        # EmbeddingProvider protocol
│   ├── sentence_transformers_provider.py
│   └── openai_compatible_provider.py  # vLLM /v1/embeddings
├── extractor/
│   ├── __init__.py
│   ├── prompts.py
│   └── worker.py                      # thread inside memory-mcp
├── retrieval/
│   ├── __init__.py
│   ├── vector_search.py
│   ├── pagerank.py
│   ├── recall.py
│   └── background_pack.py
├── dreamer_runner/
│   ├── __init__.py
│   ├── lifecycle.py                   # process lifecycle + PID lock + heartbeat
│   ├── triggers.py                    # idle / queue / scheduled rules
│   ├── runner.py                      # orchestrates stages
│   └── stages/
│       ├── __init__.py
│       ├── stage_1_cluster.py
│       ├── stage_2_consolidate.py
│       ├── stage_3_extract.py
│       ├── stage_4_reflect.py
│       ├── stage_5_recombine.py
│       └── stage_6_decay_reindex.py
└── prompts_lib/
    ├── extractor.md
    ├── consolidate.md
    ├── extract_facts.md
    ├── reflect.md
    └── recombine.md

pages/
└── 2_Dreaming.py                      # NEW operator console

tests/
├── memory/
│   ├── __init__.py
│   ├── conftest.py                    # in-memory DB fixture, mocked LLM
│   ├── test_ids.py
│   ├── test_migrations.py
│   ├── test_repo_raw_turns.py
│   ├── test_repo_episodes.py
│   ├── test_repo_facts.py             # bi-temporal invariants
│   ├── test_repo_entities.py
│   ├── test_repo_reflections.py
│   ├── test_repo_hypotheses.py
│   ├── test_repo_links.py
│   ├── test_embeddings.py
│   ├── test_extractor_worker.py
│   ├── test_vector_search.py
│   ├── test_pagerank.py
│   ├── test_recall.py
│   ├── test_background_pack.py
│   ├── test_mcp_server.py             # end-to-end MCP tool / prompt / resource
│   ├── test_dreamer_lifecycle.py
│   ├── test_stage_1_cluster.py
│   ├── test_stage_2_consolidate.py
│   ├── test_stage_3_extract.py
│   ├── test_stage_4_reflect.py
│   ├── test_stage_5_recombine.py
│   ├── test_stage_6_decay_reindex.py
│   └── fixtures/
│       └── seeded_conversation.json
└── eval/
    └── memory/
        ├── conftest.py
        ├── scenarios/
        │   └── 01_user_preferences/
        │       ├── conversations/
        │       ├── questions.yaml
        │       └── expected.yaml
        └── runner.py
```

**Touched in the existing codebase:**
- `mcp.json` — register `memory` server (one line).
- `pages/1_Basic_Chat.py` — call `record_turn` after each turn append (small hook).
- `pyproject.toml` — add dependencies.
- `playground/providers/lmstudio_client.py` — comment note explaining "lmstudio" is the OpenAI-compatible local-server provider and the URL can point at vLLM. No behavior change.

---

## Phase index

| Phase | Title | Tasks |
|---|---|---|
| 0 | Foundation: deps, package skeleton, schema | 0.1–0.6 |
| 1 | Hot path: `record_turn` + raw_turn_refs | 1.1–1.3 |
| 2 | Embeddings layer | 2.1–2.3 |
| 3 | Extractor worker (LLM → atomic episodes) | 3.1–3.4 |
| 4 | Bi-temporal facts data layer | 4.1–4.4 |
| 5 | Entities, reflections, hypotheses, links repos | 5.1–5.4 |
| 6 | MCP server stub (FastMCP wiring + basic tools) | 6.1–6.4 |
| 7 | Dreamer skeleton + advisory lock | 7.1–7.3 |
| 8 | Stage 1: ingest + cluster | 8.1–8.2 |
| 9 | Stage 2: consolidate | 9.1–9.2 |
| 10 | Stage 3: extract → bi-temporal facts | 10.1–10.3 |
| 11 | Stage 4: reflect (recursive) | 11.1–11.2 |
| 12 | Stage 5: recombine (REM-like) | 12.1–12.2 |
| 13 | Stage 6: decay + reindex (PageRank) | 13.1–13.3 |
| 14 | HippoRAG retrieval (`recall`) | 14.1–14.3 |
| 15 | Background pack (MCP prompt) | 15.1–15.2 |
| 16 | Dreaming page (Streamlit operator UI) | 16.1–16.4 |
| 17 | Eval harness | 17.1–17.3 |
| 18 | Wire-up + smoke run | 18.1–18.2 |

---

## Phase 0 — Foundation

### Task 0.1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Append the new dependencies block**

Update the `dependencies` list to add (alphabetical placement):

```toml
dependencies = [
  "streamlit>=1.40",
  "anthropic>=0.40",
  "openai>=1.55",
  "mcp>=0.9",
  "python-dotenv>=1.0",
  "sqlite-vec>=0.1.6",
  "sentence-transformers>=3.0",
  "scikit-learn>=1.5",
  "networkx>=3.3",
  "python-ulid>=3.0",
  "pyyaml>=6.0",
]
```

- [ ] **Step 2: Install**

Run: `.agent-playground/bin/pip install -e ".[dev]"`
Expected: completes with no errors; `pip list` shows the new packages.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add memory subsystem deps (sqlite-vec, sentence-transformers, networkx, scikit-learn, python-ulid, pyyaml)"
```

---

### Task 0.2: Create package skeleton

**Files:**
- Create: `mcp_servers/memory/__init__.py` (empty `"""Memory + Dreaming MCP server."""` docstring)
- Create: `mcp_servers/memory/README.md`
- Create: `mcp_servers/memory/db/__init__.py` (empty)
- Create: `mcp_servers/memory/repo/__init__.py` (empty)
- Create: `mcp_servers/memory/embeddings/__init__.py` (empty)
- Create: `mcp_servers/memory/extractor/__init__.py` (empty)
- Create: `mcp_servers/memory/retrieval/__init__.py` (empty)
- Create: `mcp_servers/memory/dreamer_runner/__init__.py` (empty)
- Create: `mcp_servers/memory/dreamer_runner/stages/__init__.py` (empty)
- Create: `mcp_servers/memory/prompts_lib/` (directory; no `__init__.py` since this is asset content)
- Create: `tests/memory/__init__.py` (empty)
- Create: `tests/memory/fixtures/` (directory)

- [ ] **Step 1: Make directories and stub files**

```bash
mkdir -p mcp_servers/memory/db/migrations \
         mcp_servers/memory/repo \
         mcp_servers/memory/embeddings \
         mcp_servers/memory/extractor \
         mcp_servers/memory/retrieval \
         mcp_servers/memory/dreamer_runner/stages \
         mcp_servers/memory/prompts_lib \
         tests/memory/fixtures
touch mcp_servers/memory/__init__.py \
      mcp_servers/memory/db/__init__.py \
      mcp_servers/memory/repo/__init__.py \
      mcp_servers/memory/embeddings/__init__.py \
      mcp_servers/memory/extractor/__init__.py \
      mcp_servers/memory/retrieval/__init__.py \
      mcp_servers/memory/dreamer_runner/__init__.py \
      mcp_servers/memory/dreamer_runner/stages/__init__.py \
      tests/memory/__init__.py
```

- [ ] **Step 2: Write the package docstring**

Replace `mcp_servers/memory/__init__.py` with:

```python
"""Memory + Dreaming MCP server.

A bundled MCP server plus a background dreamer daemon that gives the
TravisML Agent Playground a persistent cross-conversation memory with
offline consolidation. See docs/superpowers/specs/2026-05-11-memory-
dreaming-mcp-design.md for the design.
"""
```

- [ ] **Step 3: Write the README pointer**

Create `mcp_servers/memory/README.md`:

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add mcp_servers/memory/ tests/memory/
git commit -m "feat(memory): package skeleton + README pointer"
```

---

### Task 0.3: ULID-prefixed ID generators

**Files:**
- Create: `mcp_servers/memory/ids.py`
- Create: `tests/memory/test_ids.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_ids.py
import re

from mcp_servers.memory.ids import (
    new_dream_run_id,
    new_entity_id,
    new_episode_id,
    new_fact_id,
    new_hypothesis_id,
    new_raw_turn_id,
    new_reflection_id,
)


_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _assert_prefixed_ulid(s: str, prefix: str) -> None:
    assert s.startswith(prefix + "_"), s
    _, rest = s.split("_", 1)
    assert _ULID_RE.match(rest), f"not a ULID after prefix: {rest}"


def test_each_generator_has_distinct_prefix_and_yields_ulids() -> None:
    pairs = [
        (new_raw_turn_id(), "rt"),
        (new_episode_id(), "ep"),
        (new_entity_id(), "en"),
        (new_fact_id(), "fa"),
        (new_reflection_id(), "re"),
        (new_hypothesis_id(), "hy"),
        (new_dream_run_id(), "dr"),
    ]
    for value, prefix in pairs:
        _assert_prefixed_ulid(value, prefix)


def test_consecutive_ids_are_unique() -> None:
    assert new_episode_id() != new_episode_id()
```

- [ ] **Step 2: Run the test (expect FAIL)**

```bash
.agent-playground/bin/pytest tests/memory/test_ids.py -v
```

Expected: `ModuleNotFoundError: No module named 'mcp_servers.memory.ids'`.

- [ ] **Step 3: Write the implementation**

```python
# mcp_servers/memory/ids.py
"""ULID-based id generators with kind prefixes."""

from __future__ import annotations

from ulid import ULID


def _new(prefix: str) -> str:
    return f"{prefix}_{ULID()}"


def new_raw_turn_id() -> str:
    return _new("rt")


def new_episode_id() -> str:
    return _new("ep")


def new_entity_id() -> str:
    return _new("en")


def new_fact_id() -> str:
    return _new("fa")


def new_reflection_id() -> str:
    return _new("re")


def new_hypothesis_id() -> str:
    return _new("hy")


def new_dream_run_id() -> str:
    return _new("dr")
```

- [ ] **Step 4: Run tests (expect PASS)**

```bash
.agent-playground/bin/pytest tests/memory/test_ids.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/memory/ids.py tests/memory/test_ids.py
git commit -m "feat(memory): ULID-prefixed id generators"
```

---

### Task 0.4: SQLite connection helper with WAL + sqlite-vec

**Files:**
- Create: `mcp_servers/memory/db/connection.py`
- Create: `tests/memory/conftest.py`
- Create: `tests/memory/test_connection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_connection.py
from pathlib import Path

import pytest

from mcp_servers.memory.db.connection import open_connection


def test_open_connection_enables_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    conn = open_connection(db_path)
    cur = conn.execute("PRAGMA journal_mode")
    assert cur.fetchone()[0] == "wal"
    conn.close()


def test_open_connection_loads_sqlite_vec(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    cur = conn.execute("SELECT vec_version()")
    version = cur.fetchone()[0]
    assert isinstance(version, str) and len(version) > 0
    conn.close()


def test_open_connection_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "memory.db"
    conn = open_connection(nested)
    assert nested.parent.is_dir()
    conn.close()
```

- [ ] **Step 2: Run (expect FAIL)**

```bash
.agent-playground/bin/pytest tests/memory/test_connection.py -v
```

Expected: import error.

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/db/connection.py
"""SQLite connection setup with WAL and sqlite-vec extension."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def open_connection(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
```

- [ ] **Step 4: Run (expect PASS)**

```bash
.agent-playground/bin/pytest tests/memory/test_connection.py -v
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/memory/db/connection.py tests/memory/test_connection.py
git commit -m "feat(memory): sqlite connection with WAL + sqlite-vec extension"
```

---

### Task 0.5: Initial migration with the full schema

**Files:**
- Create: `mcp_servers/memory/db/migrations/001_initial.sql`
- Create: `mcp_servers/memory/db/migrations.py`
- Create: `tests/memory/test_migrations.py`

- [ ] **Step 1: Write the migration file**

```sql
-- mcp_servers/memory/db/migrations/001_initial.sql
CREATE TABLE schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE raw_turn_refs (
  id                TEXT PRIMARY KEY,
  conversation_id   TEXT NOT NULL,
  turn_index        INTEGER NOT NULL,
  role              TEXT NOT NULL,
  occurred_at       TEXT NOT NULL,
  recorded_at       TEXT NOT NULL,
  extraction_status TEXT NOT NULL DEFAULT 'pending',
  retry_count       INTEGER NOT NULL DEFAULT 0,
  last_error        TEXT,
  UNIQUE (conversation_id, turn_index)
);
CREATE INDEX idx_raw_turn_refs_extraction_status ON raw_turn_refs(extraction_status);

CREATE TABLE entities (
  id             TEXT PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  kind           TEXT NOT NULL,
  aliases        TEXT NOT NULL DEFAULT '[]',
  summary        TEXT,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  importance     REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX idx_entities_kind ON entities(kind);

CREATE TABLE episodes (
  id              TEXT PRIMARY KEY,
  actor           TEXT NOT NULL,
  predicate       TEXT NOT NULL,
  subject_entity  TEXT REFERENCES entities(id),
  object_entity   TEXT REFERENCES entities(id),
  object_value    TEXT,
  summary         TEXT NOT NULL,
  importance      REAL NOT NULL DEFAULT 0.5,
  occurred_at     TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'fresh',
  source_refs     TEXT NOT NULL
);
CREATE INDEX idx_episodes_status ON episodes(status);
CREATE INDEX idx_episodes_occurred_at ON episodes(occurred_at);

CREATE TABLE facts (
  id                   TEXT PRIMARY KEY,
  subject_entity       TEXT NOT NULL REFERENCES entities(id),
  predicate            TEXT NOT NULL,
  object_entity        TEXT REFERENCES entities(id),
  object_value         TEXT,
  valid_from           TEXT NOT NULL,
  valid_to             TEXT,
  learned_at           TEXT NOT NULL,
  invalidated_at       TEXT,
  source_episode_ids   TEXT NOT NULL,
  confidence           REAL NOT NULL DEFAULT 0.7,
  supersedes           TEXT REFERENCES facts(id),
  superseded_by        TEXT REFERENCES facts(id),
  created_in_dream_run TEXT NOT NULL,
  CHECK (object_entity IS NOT NULL OR object_value IS NOT NULL)
);
CREATE INDEX idx_facts_subject_predicate ON facts(subject_entity, predicate);
CREATE INDEX idx_facts_valid_to ON facts(valid_to);
CREATE INDEX idx_facts_invalidated_at ON facts(invalidated_at);

CREATE TABLE reflections (
  id                   TEXT PRIMARY KEY,
  summary              TEXT NOT NULL,
  importance           REAL NOT NULL,
  level                INTEGER NOT NULL,
  source_kind          TEXT NOT NULL,
  source_ids           TEXT NOT NULL,
  created_at           TEXT NOT NULL,
  created_in_dream_run TEXT NOT NULL
);
CREATE INDEX idx_reflections_level ON reflections(level);

CREATE TABLE hypotheses (
  id                   TEXT PRIMARY KEY,
  statement            TEXT NOT NULL,
  source_node_ids      TEXT NOT NULL,
  confidence           REAL NOT NULL,
  status               TEXT NOT NULL DEFAULT 'open',
  resolved_at          TEXT,
  resolved_by          TEXT,
  resolution_note      TEXT,
  created_at           TEXT NOT NULL,
  created_in_dream_run TEXT NOT NULL
);
CREATE INDEX idx_hypotheses_status ON hypotheses(status);

CREATE TABLE links (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  src_kind             TEXT NOT NULL,
  src_id               TEXT NOT NULL,
  dst_kind             TEXT NOT NULL,
  dst_id               TEXT NOT NULL,
  link_type            TEXT NOT NULL,
  weight               REAL NOT NULL DEFAULT 1.0,
  created_in_dream_run TEXT,
  UNIQUE (src_kind, src_id, dst_kind, dst_id, link_type)
);
CREATE INDEX idx_links_src ON links(src_kind, src_id);
CREATE INDEX idx_links_dst ON links(dst_kind, dst_id);

CREATE VIRTUAL TABLE embeddings USING vec0(
  node_kind TEXT,
  node_id   TEXT,
  embedding FLOAT[768]
);

CREATE TABLE pagerank_scores (
  node_kind             TEXT NOT NULL,
  node_id               TEXT NOT NULL,
  score                 REAL NOT NULL,
  computed_in_dream_run TEXT NOT NULL,
  PRIMARY KEY (node_kind, node_id)
);

CREATE TABLE dream_runs (
  id             TEXT PRIMARY KEY,
  started_at     TEXT NOT NULL,
  ended_at       TEXT,
  cycle_mode     TEXT NOT NULL,
  trigger_reason TEXT NOT NULL,
  stages         TEXT NOT NULL DEFAULT '{}',
  model_used     TEXT NOT NULL,
  status         TEXT NOT NULL,
  error          TEXT
);
CREATE INDEX idx_dream_runs_status ON dream_runs(status);

CREATE TABLE dreamer_config (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE dreamer_lock (
  id          INTEGER PRIMARY KEY CHECK (id = 1),
  pid         INTEGER NOT NULL,
  acquired_at TEXT NOT NULL,
  heartbeat   TEXT NOT NULL
);
```

- [ ] **Step 2: Write the failing test**

```python
# tests/memory/test_migrations.py
from pathlib import Path

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations, current_version


def _tables(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return {r[0] for r in rows}


def test_apply_migrations_creates_full_schema(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    apply_migrations(conn)
    tables = _tables(conn)
    for expected in (
        "schema_version", "raw_turn_refs", "entities", "episodes", "facts",
        "reflections", "hypotheses", "links", "embeddings",
        "pagerank_scores", "dream_runs", "dreamer_config", "dreamer_lock",
    ):
        assert expected in tables, expected
    assert current_version(conn) == 1
    conn.close()


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    conn = open_connection(tmp_path / "memory.db")
    apply_migrations(conn)
    apply_migrations(conn)
    assert current_version(conn) == 1
    conn.close()
```

- [ ] **Step 3: Run (expect FAIL)**

```bash
.agent-playground/bin/pytest tests/memory/test_migrations.py -v
```

- [ ] **Step 4: Implement the runner**

```python
# mcp_servers/memory/db/migrations.py
"""Forward-only SQL migration runner. Migrations live in ./migrations/NNN_*.sql."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(row[0])


def apply_migrations(conn: sqlite3.Connection) -> None:
    current = current_version(conn)
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(path.name.split("_", 1)[0])
        if version <= current:
            continue
        sql = path.read_text()
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )
```

- [ ] **Step 5: Run (expect PASS)**

```bash
.agent-playground/bin/pytest tests/memory/test_migrations.py -v
```

- [ ] **Step 6: Commit**

```bash
git add mcp_servers/memory/db/migrations.py mcp_servers/memory/db/migrations/001_initial.sql tests/memory/test_migrations.py
git commit -m "feat(memory): initial schema migration (12 tables + sqlite-vec index)"
```

---

### Task 0.6: Shared test fixtures (in-memory DB, mocked LLM)

**Files:**
- Create: `tests/memory/conftest.py`

- [ ] **Step 1: Write the fixtures module**

```python
# tests/memory/conftest.py
"""Shared fixtures for memory subsystem tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    c = open_connection(db_path)
    apply_migrations(c)
    yield c
    c.close()


@pytest.fixture
def fake_llm() -> MagicMock:
    """A MagicMock standing in for an `LLMClient` instance.

    Tests configure `.stream_chat.return_value = [TextDelta(...), MessageComplete(...)]`
    or override per-call.
    """
    return MagicMock()


@pytest.fixture
def fixed_embedder() -> Any:
    """Returns a deterministic 768-dim embedding for any input string."""

    class _Embedder:
        dim = 768

        def embed(self, text: str) -> list[float]:
            # Cheap, deterministic, distinct per input — not semantic.
            h = abs(hash(text))
            return [((h >> (i % 30)) & 1) - 0.5 for i in range(self.dim)]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(t) for t in texts]

    return _Embedder()
```

- [ ] **Step 2: Sanity-check it imports**

```bash
.agent-playground/bin/pytest tests/memory/ -q
```

Expected: all prior tests still pass; no fixture import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/memory/conftest.py
git commit -m "test(memory): shared fixtures (in-memory DB, fake LLM, deterministic embedder)"
```

---

## Phase 1 — Hot path: `record_turn` + raw_turn_refs

### Task 1.1: `raw_turns` repo with idempotent insert

**Files:**
- Create: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/raw_turns.py`
- Create: `tests/memory/test_repo_raw_turns.py`

- [ ] **Step 1: Write the models module (initial slice)**

```python
# mcp_servers/memory/models.py
"""Dataclasses for memory subsystem rows. Storage layer maps to/from these."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTurnRef:
    id: str
    conversation_id: str
    turn_index: int
    role: str               # 'user' | 'assistant' | 'tool'
    occurred_at: str        # ISO-8601 Z
    recorded_at: str
    extraction_status: str  # 'pending' | 'done' | 'failed' | 'poison'
    retry_count: int = 0
    last_error: str | None = None
```

- [ ] **Step 2: Write the failing test**

```python
# tests/memory/test_repo_raw_turns.py
import sqlite3

import pytest

from mcp_servers.memory.repo.raw_turns import (
    list_pending,
    mark_extraction_status,
    record_turn,
)


def test_record_turn_inserts_pending_row(conn: sqlite3.Connection) -> None:
    rt = record_turn(
        conn,
        conversation_id="2026-05-12T15-00-00-aaaa",
        turn_index=0,
        role="user",
        occurred_at="2026-05-12T15:00:01Z",
    )
    assert rt.extraction_status == "pending"
    assert rt.retry_count == 0
    row = conn.execute(
        "SELECT id, role, extraction_status FROM raw_turn_refs"
    ).fetchone()
    assert row["id"] == rt.id
    assert row["role"] == "user"
    assert row["extraction_status"] == "pending"


def test_record_turn_is_idempotent_on_dup_key(conn: sqlite3.Connection) -> None:
    a = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    b = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    assert a.id == b.id


def test_list_pending_returns_only_pending(conn: sqlite3.Connection) -> None:
    a = record_turn(conn, conversation_id="c1", turn_index=0, role="user",
                    occurred_at="2026-05-12T15:00:01Z")
    b = record_turn(conn, conversation_id="c1", turn_index=1, role="assistant",
                    occurred_at="2026-05-12T15:00:02Z")
    mark_extraction_status(conn, b.id, "done")
    pending = list_pending(conn)
    assert [p.id for p in pending] == [a.id]
```

- [ ] **Step 3: Run (expect FAIL)**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_raw_turns.py -v
```

- [ ] **Step 4: Implement**

```python
# mcp_servers/memory/repo/raw_turns.py
"""Repo for raw_turn_refs — pointer rows into existing conversations/*.json."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_raw_turn_id
from mcp_servers.memory.models import RawTurnRef


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_turn(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_index: int,
    role: str,
    occurred_at: str,
) -> RawTurnRef:
    existing = conn.execute(
        "SELECT * FROM raw_turn_refs WHERE conversation_id = ? AND turn_index = ?",
        (conversation_id, turn_index),
    ).fetchone()
    if existing is not None:
        return _row_to_model(existing)
    rt_id = new_raw_turn_id()
    recorded_at = _now()
    conn.execute(
        """
        INSERT INTO raw_turn_refs
            (id, conversation_id, turn_index, role, occurred_at, recorded_at, extraction_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (rt_id, conversation_id, turn_index, role, occurred_at, recorded_at),
    )
    return RawTurnRef(
        id=rt_id,
        conversation_id=conversation_id,
        turn_index=turn_index,
        role=role,
        occurred_at=occurred_at,
        recorded_at=recorded_at,
        extraction_status="pending",
    )


def mark_extraction_status(
    conn: sqlite3.Connection,
    raw_turn_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if status == "failed":
        conn.execute(
            "UPDATE raw_turn_refs SET extraction_status = ?, retry_count = retry_count + 1, last_error = ? WHERE id = ?",
            (status, error, raw_turn_id),
        )
    else:
        conn.execute(
            "UPDATE raw_turn_refs SET extraction_status = ?, last_error = ? WHERE id = ?",
            (status, error, raw_turn_id),
        )


def list_pending(conn: sqlite3.Connection, *, limit: int = 100) -> list[RawTurnRef]:
    rows = conn.execute(
        "SELECT * FROM raw_turn_refs WHERE extraction_status = 'pending' ORDER BY recorded_at LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_model(r) for r in rows]


def _row_to_model(row: sqlite3.Row) -> RawTurnRef:
    return RawTurnRef(
        id=row["id"],
        conversation_id=row["conversation_id"],
        turn_index=row["turn_index"],
        role=row["role"],
        occurred_at=row["occurred_at"],
        recorded_at=row["recorded_at"],
        extraction_status=row["extraction_status"],
        retry_count=row["retry_count"],
        last_error=row["last_error"],
    )
```

- [ ] **Step 5: Run (expect PASS)** and **commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_raw_turns.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/raw_turns.py tests/memory/test_repo_raw_turns.py
git commit -m "feat(memory): raw_turn_refs repo (idempotent record_turn + status updates)"
```

---

### Task 1.2: Read raw turn content from `conversations/*.json`

**Files:**
- Create: `mcp_servers/memory/repo/raw_turn_content.py`
- Create: `tests/memory/test_repo_raw_turn_content.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_repo_raw_turn_content.py
import json
from pathlib import Path

from mcp_servers.memory.repo.raw_turn_content import read_turn_text


def _seed_conv(root: Path, conv_id: str, turns: list[dict]) -> None:
    page_dir = root / "basic_chat"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{conv_id}.json").write_text(json.dumps({
        "schema_version": 1, "id": conv_id, "page": "basic_chat",
        "messages": turns,
    }))


def test_read_turn_text_joins_text_blocks(tmp_path: Path) -> None:
    _seed_conv(tmp_path, "c1", [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]},
    ])
    assert read_turn_text(tmp_path, "c1", 0) == "hello\nworld"


def test_read_turn_text_skips_non_text_blocks(tmp_path: Path) -> None:
    _seed_conv(tmp_path, "c1", [
        {"role": "assistant", "content": [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
        ]},
    ])
    assert read_turn_text(tmp_path, "c1", 0) == "thinking..."
```

- [ ] **Step 2: Run (expect FAIL)** then **implement**

```python
# mcp_servers/memory/repo/raw_turn_content.py
"""Resolve raw turn text by reading the playground's conversations/*.json files."""

from __future__ import annotations

import json
from pathlib import Path


def read_turn_text(
    conversations_root: str | Path,
    conversation_id: str,
    turn_index: int,
    *,
    page: str = "basic_chat",
) -> str:
    path = Path(conversations_root) / page / f"{conversation_id}.json"
    data = json.loads(path.read_text())
    msg = data["messages"][turn_index]
    parts: list[str] = []
    for block in msg.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
```

- [ ] **Step 3: Run (expect PASS) and commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_raw_turn_content.py -v
git add mcp_servers/memory/repo/raw_turn_content.py tests/memory/test_repo_raw_turn_content.py
git commit -m "feat(memory): read raw turn text from conversations/*.json"
```

---

### Task 1.3: Hook `record_turn` into Basic Chat

**Files:**
- Modify: `pages/1_Basic_Chat.py`
- Create: `tests/memory/test_basic_chat_hook.py` (skip if not feasible — see below)

This task wires the existing chat page to call `record_turn` after every message append. Because the chat page is heavy on Streamlit state, the test is a focused unit test on a small helper we extract; we don't try to mount the page.

- [ ] **Step 1: Extract a small helper into the memory package and test it**

Create `mcp_servers/memory/hot_path.py`:

```python
# mcp_servers/memory/hot_path.py
"""Helpers the playground uses to record turns into memory."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.raw_turns import record_turn


def on_turn_appended(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_index: int,
    role: str,
    occurred_at: str,
) -> None:
    """Single entry point the playground calls. Wraps record_turn in a try
    so that memory failures never break the chat flow."""
    try:
        record_turn(
            conn,
            conversation_id=conversation_id,
            turn_index=turn_index,
            role=role,
            occurred_at=occurred_at,
        )
    except Exception:
        pass  # memory degrades gracefully
```

- [ ] **Step 2: Test the helper**

```python
# tests/memory/test_basic_chat_hook.py
import sqlite3

from mcp_servers.memory.hot_path import on_turn_appended


def test_on_turn_appended_writes_row(conn: sqlite3.Connection) -> None:
    on_turn_appended(conn, conversation_id="c1", turn_index=0,
                     role="user", occurred_at="2026-05-12T15:00:01Z")
    row = conn.execute("SELECT id FROM raw_turn_refs").fetchone()
    assert row is not None


def test_on_turn_appended_swallows_errors() -> None:
    # closed connection should NOT raise out of the helper
    bad = sqlite3.connect(":memory:")
    bad.close()
    on_turn_appended(bad, conversation_id="c1", turn_index=0,
                     role="user", occurred_at="2026-05-12T15:00:01Z")
```

Run:

```bash
.agent-playground/bin/pytest tests/memory/test_basic_chat_hook.py -v
```

- [ ] **Step 3: Wire the hook into `pages/1_Basic_Chat.py`**

In `pages/1_Basic_Chat.py`, find the two `conv.append_message(...)` call sites (one after the user message, one after each assistant message). Add a memory-DB connection set up once at module load, and call `on_turn_appended` right after each append.

Add near the top imports:

```python
from mcp_servers.memory.db.connection import open_connection as _mem_open
from mcp_servers.memory.db.migrations import apply_migrations as _mem_migrate
from mcp_servers.memory.hot_path import on_turn_appended as _mem_record
```

Add right after the existing `load_dotenv()`:

```python
@st.cache_resource(show_spinner=False)
def _memory_conn():
    p = (st.session_state.get("MEMORY_DB_PATH")
         or (__import__("pathlib").Path.home() / ".travisml-playground" / "memory.db"))
    c = _mem_open(p)
    _mem_migrate(c)
    return c
```

After the user-message `conv.append_message({...})` block, append:

```python
_mem_record(
    _memory_conn(),
    conversation_id=conv.id,
    turn_index=len(conv.data["messages"]) - 1,
    role="user",
    occurred_at=_now_iso(),
)
```

After the assistant `conv.append_message(save_msg)` block, append:

```python
_mem_record(
    _memory_conn(),
    conversation_id=conv.id,
    turn_index=len(conv.data["messages"]) - 1,
    role="assistant",
    occurred_at=_now_iso(),
)
```

- [ ] **Step 4: Smoke-test by running streamlit and sending one message**

```bash
streamlit run app.py
# in the browser: send "hello" in Basic Chat
# then in another shell:
sqlite3 ~/.travisml-playground/memory.db "SELECT id, role, extraction_status FROM raw_turn_refs"
```

Expected: at least two rows (one user, one assistant), both `extraction_status='pending'`.

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/memory/hot_path.py tests/memory/test_basic_chat_hook.py pages/1_Basic_Chat.py
git commit -m "feat(memory): record raw turns from Basic Chat (graceful-degrade hook)"
```

---

## Phase 2 — Embeddings layer

### Task 2.1: `EmbeddingProvider` protocol

**Files:**
- Create: `mcp_servers/memory/embeddings/base.py`
- Create: `tests/memory/test_embeddings_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_embeddings_base.py
from mcp_servers.memory.embeddings.base import EmbeddingProvider


def test_protocol_has_required_attrs() -> None:
    # All implementations must expose `dim` and `model_id`, plus
    # `embed(text)` and `embed_many(texts)`.
    assert hasattr(EmbeddingProvider, "dim")
    assert hasattr(EmbeddingProvider, "model_id")
    assert hasattr(EmbeddingProvider, "embed")
    assert hasattr(EmbeddingProvider, "embed_many")
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/embeddings/base.py
"""EmbeddingProvider protocol."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    dim: int
    model_id: str

    def embed(self, text: str) -> list[float]: ...
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_embeddings_base.py -v
git add mcp_servers/memory/embeddings/base.py tests/memory/test_embeddings_base.py
git commit -m "feat(memory): EmbeddingProvider protocol"
```

---

### Task 2.2: `sentence-transformers` default provider

**Files:**
- Create: `mcp_servers/memory/embeddings/sentence_transformers_provider.py`
- Create: `tests/memory/test_embeddings_st.py`

- [ ] **Step 1: Write the test (model loads lazily)**

```python
# tests/memory/test_embeddings_st.py
import pytest

from mcp_servers.memory.embeddings.sentence_transformers_provider import (
    SentenceTransformersProvider,
)


@pytest.mark.slow
def test_embed_returns_768_dim_vector() -> None:
    p = SentenceTransformersProvider(model_id="nomic-ai/nomic-embed-text-v1.5")
    vec = p.embed("hello world")
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.slow
def test_embed_many_matches_embed_singletons() -> None:
    p = SentenceTransformersProvider(model_id="nomic-ai/nomic-embed-text-v1.5")
    a = p.embed("foo")
    b = p.embed("bar")
    both = p.embed_many(["foo", "bar"])
    assert both[0] == pytest.approx(a, rel=1e-4)
    assert both[1] == pytest.approx(b, rel=1e-4)
```

Mark `@pytest.mark.slow` so default `pytest` runs skip them; add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["slow: tests that download models or are otherwise slow"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/embeddings/sentence_transformers_provider.py
"""Local, in-process embeddings via sentence-transformers."""

from __future__ import annotations


class SentenceTransformersProvider:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self._model = SentenceTransformer(model_id, trust_remote_code=True)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in arr]
```

- [ ] **Step 3: Run slow tests once locally**

```bash
.agent-playground/bin/pytest -m slow tests/memory/test_embeddings_st.py -v
```

- [ ] **Step 4: Commit**

```bash
git add mcp_servers/memory/embeddings/sentence_transformers_provider.py tests/memory/test_embeddings_st.py pyproject.toml
git commit -m "feat(memory): sentence-transformers embedding provider (nomic-embed-text-v1.5 default)"
```

---

### Task 2.3: OpenAI-compatible embedding provider (vLLM)

**Files:**
- Create: `mcp_servers/memory/embeddings/openai_compatible_provider.py`
- Create: `tests/memory/test_embeddings_openai_compat.py`

- [ ] **Step 1: Write the test using `respx`-style HTTP mocking**

```python
# tests/memory/test_embeddings_openai_compat.py
import respx
from httpx import Response

from mcp_servers.memory.embeddings.openai_compatible_provider import (
    OpenAICompatibleEmbeddingProvider,
)


@respx.mock
def test_embed_calls_v1_embeddings_endpoint() -> None:
    respx.post("http://localhost:8000/v1/embeddings").mock(
        return_value=Response(200, json={
            "data": [{"embedding": [0.1] * 768, "index": 0}],
            "model": "BAAI/bge-base-en-v1.5",
        })
    )
    p = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8000/v1",
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
    )
    vec = p.embed("hello")
    assert vec == [0.1] * 768


@respx.mock
def test_embed_many_batches_inputs() -> None:
    respx.post("http://localhost:8000/v1/embeddings").mock(
        return_value=Response(200, json={
            "data": [
                {"embedding": [0.1] * 768, "index": 0},
                {"embedding": [0.2] * 768, "index": 1},
            ],
            "model": "BAAI/bge-base-en-v1.5",
        })
    )
    p = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8000/v1",
        model_id="BAAI/bge-base-en-v1.5", dim=768,
    )
    out = p.embed_many(["a", "b"])
    assert out == [[0.1] * 768, [0.2] * 768]
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/embeddings/openai_compatible_provider.py
"""Embedding provider for any OpenAI-compatible /v1/embeddings endpoint (vLLM)."""

from __future__ import annotations

import os

import httpx


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_id: str,
        dim: int,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LMSTUDIO_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url required (or set LMSTUDIO_BASE_URL)")
        self.model_id = model_id
        self.dim = dim
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model_id, "input": texts},
            headers=self._headers(),
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        return [list(d["embedding"]) for d in data]
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_embeddings_openai_compat.py -v
git add mcp_servers/memory/embeddings/openai_compatible_provider.py tests/memory/test_embeddings_openai_compat.py
git commit -m "feat(memory): OpenAI-compatible embedding provider (vLLM /v1/embeddings)"
```

---

## Phase 3 — Extractor worker (atomic episodes)

### Task 3.1: Extractor prompt template

**Files:**
- Create: `mcp_servers/memory/prompts_lib/extractor.md`

- [ ] **Step 1: Write the prompt**

```markdown
You extract atomic episodic events from a single conversation turn.

Return a JSON object with one key "episodes" whose value is a list of 0
to 6 atomic events that are clearly stated or strongly implied by the
turn. Do NOT speculate beyond what the text says.

Each event must include:
- actor:      "user" | "agent" | "tool:<name>"
- predicate:  a normalized lowercase verb phrase using snake_case
              (e.g. "reported_problem", "expressed_preference",
              "diagnosed", "decided", "asked_question", "confirmed")
- subject:    short canonical-cased noun phrase OR null
- object:     short canonical-cased noun phrase OR null
- summary:    one sentence describing what happened (≤ 30 words)
- importance: 0.0 (trivial) to 1.0 (highly significant), float

If the turn contains nothing memorable, return {"episodes": []}.

Output ONLY the JSON object. No prose, no fences.

---
Conversation context (last few turns, oldest first):
{{context}}

---
Turn to extract from (role={{role}}, occurred_at={{occurred_at}}):
{{turn_text}}
```

- [ ] **Step 2: Commit**

```bash
git add mcp_servers/memory/prompts_lib/extractor.md
git commit -m "feat(memory): extractor LLM prompt template"
```

---

### Task 3.2: Episode repo

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/episodes.py`
- Create: `tests/memory/test_repo_episodes.py`

- [ ] **Step 1: Extend models.py**

Append to `mcp_servers/memory/models.py`:

```python
@dataclass(frozen=True)
class Episode:
    id: str
    actor: str
    predicate: str
    subject_entity: str | None
    object_entity: str | None
    object_value: str | None
    summary: str
    importance: float
    occurred_at: str
    created_at: str
    status: str               # 'fresh' | 'consolidated' | 'archived'
    source_refs: list[dict]   # list of {"raw_turn_ref_id": "rt_..."}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/memory/test_repo_episodes.py
import sqlite3

from mcp_servers.memory.repo.episodes import (
    insert_episode,
    list_by_status,
    set_status,
)


def test_insert_and_list_fresh(conn: sqlite3.Connection) -> None:
    ep = insert_episode(
        conn,
        actor="user",
        predicate="reported_problem",
        subject_entity=None,
        object_entity=None,
        object_value="mcp pool eventloop death",
        summary="user reports the MCP pool keeps dying",
        importance=0.7,
        occurred_at="2026-05-12T15:00:01Z",
        source_refs=[{"raw_turn_ref_id": "rt_abc"}],
    )
    assert ep.status == "fresh"
    out = list_by_status(conn, "fresh")
    assert [o.id for o in out] == [ep.id]


def test_set_status_consolidates(conn: sqlite3.Connection) -> None:
    ep = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="foo", summary="s",
        importance=0.1, occurred_at="2026-05-12T15:00:01Z",
        source_refs=[],
    )
    set_status(conn, ep.id, "consolidated")
    assert list_by_status(conn, "fresh") == []
    assert [e.id for e in list_by_status(conn, "consolidated")] == [ep.id]
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/repo/episodes.py
"""Repo for episodes — atomic episodic events extracted from raw turns."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_episode_id
from mcp_servers.memory.models import Episode


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_episode(
    conn: sqlite3.Connection,
    *,
    actor: str,
    predicate: str,
    subject_entity: str | None,
    object_entity: str | None,
    object_value: str | None,
    summary: str,
    importance: float,
    occurred_at: str,
    source_refs: list[dict],
) -> Episode:
    ep_id = new_episode_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO episodes
            (id, actor, predicate, subject_entity, object_entity, object_value,
             summary, importance, occurred_at, created_at, status, source_refs)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fresh', ?)
        """,
        (ep_id, actor, predicate, subject_entity, object_entity, object_value,
         summary, importance, occurred_at, created_at, json.dumps(source_refs)),
    )
    return Episode(
        id=ep_id, actor=actor, predicate=predicate,
        subject_entity=subject_entity, object_entity=object_entity,
        object_value=object_value, summary=summary, importance=importance,
        occurred_at=occurred_at, created_at=created_at,
        status="fresh", source_refs=source_refs,
    )


def set_status(conn: sqlite3.Connection, episode_id: str, status: str) -> None:
    conn.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id))


def list_by_status(
    conn: sqlite3.Connection, status: str, *, limit: int = 1000
) -> list[Episode]:
    rows = conn.execute(
        "SELECT * FROM episodes WHERE status = ? ORDER BY occurred_at LIMIT ?",
        (status, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Episode:
    return Episode(
        id=r["id"], actor=r["actor"], predicate=r["predicate"],
        subject_entity=r["subject_entity"], object_entity=r["object_entity"],
        object_value=r["object_value"], summary=r["summary"],
        importance=r["importance"], occurred_at=r["occurred_at"],
        created_at=r["created_at"], status=r["status"],
        source_refs=json.loads(r["source_refs"]),
    )
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_episodes.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/episodes.py tests/memory/test_repo_episodes.py
git commit -m "feat(memory): episodes repo (insert, status updates, list-by-status)"
```

---

### Task 3.3: Extractor that calls the LLM and writes episodes

**Files:**
- Create: `mcp_servers/memory/extractor/worker.py`
- Create: `tests/memory/test_extractor_worker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_extractor_worker.py
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

from mcp_servers.memory.extractor.worker import extract_for_turn
from mcp_servers.memory.providers.base import (  # noqa: re-exported in stub below
    MessageComplete, TextDelta, Usage,
)
from mcp_servers.memory.repo.episodes import list_by_status
from mcp_servers.memory.repo.raw_turns import record_turn


def _make_fake_stream(payload: dict) -> Iterator:
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=10, output_tokens=10), stop_reason="end_turn")


def test_extract_for_turn_writes_episodes(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # seed conversation file
    page = tmp_path / "basic_chat"
    page.mkdir(parents=True)
    (page / "c1.json").write_text(json.dumps({
        "id": "c1", "page": "basic_chat",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "MCP pool keeps dying"}]},
        ],
    }))
    rt = record_turn(
        conn, conversation_id="c1", turn_index=0, role="user",
        occurred_at="2026-05-12T15:00:01Z",
    )

    fake = MagicMock()
    fake.stream_chat.return_value = _make_fake_stream({
        "episodes": [{
            "actor": "user", "predicate": "reported_problem",
            "subject": None, "object": "MCP pool eventloop death",
            "summary": "user reports MCP pool keeps dying",
            "importance": 0.8,
        }],
    })

    extract_for_turn(
        conn=conn,
        llm=fake,
        conversations_root=tmp_path,
        raw_turn_id=rt.id,
    )

    eps = list_by_status(conn, "fresh")
    assert len(eps) == 1
    assert eps[0].summary == "user reports MCP pool keeps dying"
    row = conn.execute(
        "SELECT extraction_status FROM raw_turn_refs WHERE id = ?", (rt.id,)
    ).fetchone()
    assert row["extraction_status"] == "done"
```

- [ ] **Step 2: Add an import shim for the LLM types**

Create `mcp_servers/memory/providers/__init__.py` (empty) and `mcp_servers/memory/providers/base.py`:

```python
# mcp_servers/memory/providers/base.py
"""Re-export of LLM event types from the playground so tests do not import
the playground's package path inside memory tests directly."""

from __future__ import annotations

from playground.providers.base import (  # noqa: F401
    ChatMessage, LLMClient, MessageComplete, TextBlock, TextDelta,
    ToolCallComplete, ToolCallDelta, ToolDefinition, Usage,
)
```

- [ ] **Step 3: Implement the worker**

```python
# mcp_servers/memory/extractor/worker.py
"""Extract atomic episodes from a raw turn by calling the configured LLM."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mcp_servers.memory.providers.base import (
    ChatMessage, LLMClient, MessageComplete, TextBlock, TextDelta,
)
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.raw_turn_content import read_turn_text
from mcp_servers.memory.repo.raw_turns import mark_extraction_status

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts_lib" / "extractor.md"


def _build_prompt(*, turn_text: str, role: str, occurred_at: str, context: str = "") -> str:
    tpl = _PROMPT_PATH.read_text()
    return (
        tpl.replace("{{context}}", context or "(none)")
           .replace("{{role}}", role)
           .replace("{{occurred_at}}", occurred_at)
           .replace("{{turn_text}}", turn_text)
    )


def _collect_text(events) -> str:
    out: list[str] = []
    for ev in events:
        if isinstance(ev, TextDelta):
            out.append(ev.text)
        elif isinstance(ev, MessageComplete):
            break
    return "".join(out)


def _parse_payload(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    return list(data.get("episodes", []))


def extract_for_turn(
    *,
    conn: sqlite3.Connection,
    llm: LLMClient,
    conversations_root: str | Path,
    raw_turn_id: str,
    max_tokens: int = 1024,
) -> int:
    row = conn.execute(
        "SELECT * FROM raw_turn_refs WHERE id = ?", (raw_turn_id,)
    ).fetchone()
    if row is None:
        return 0
    turn_text = read_turn_text(
        conversations_root, row["conversation_id"], row["turn_index"],
    )
    prompt = _build_prompt(
        turn_text=turn_text, role=row["role"], occurred_at=row["occurred_at"],
    )

    try:
        events = llm.stream_chat(
            messages=[ChatMessage(role="user", content=[TextBlock(type="text", text=prompt)])],
            system="You are an information extractor. Return strict JSON only.",
            tools=[], max_tokens=max_tokens, temperature=0.0,
        )
        payload = _collect_text(events)
        episodes = _parse_payload(payload)
    except Exception as e:
        mark_extraction_status(conn, raw_turn_id, "failed", error=str(e))
        return 0

    count = 0
    for ep in episodes:
        insert_episode(
            conn,
            actor=ep["actor"],
            predicate=ep["predicate"],
            subject_entity=None,
            object_entity=None,
            object_value=ep.get("object") or ep.get("subject"),
            summary=ep["summary"],
            importance=float(ep.get("importance", 0.5)),
            occurred_at=row["occurred_at"],
            source_refs=[{"raw_turn_ref_id": raw_turn_id}],
        )
        count += 1
    mark_extraction_status(conn, raw_turn_id, "done")
    return count
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_extractor_worker.py -v
git add mcp_servers/memory/providers/__init__.py mcp_servers/memory/providers/base.py mcp_servers/memory/extractor/worker.py tests/memory/test_extractor_worker.py
git commit -m "feat(memory): extractor worker (LLM-driven atomic episode extraction)"
```

---

### Task 3.4: Extractor pump (process the pending queue)

**Files:**
- Create: `mcp_servers/memory/extractor/pump.py`
- Create: `tests/memory/test_extractor_pump.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_extractor_pump.py
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from mcp_servers.memory.extractor.pump import pump_once
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.raw_turns import record_turn


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_pump_once_processes_all_pending(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    page = tmp_path / "basic_chat"; page.mkdir(parents=True)
    (page / "c.json").write_text(json.dumps({
        "id": "c", "page": "basic_chat",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        ],
    }))
    record_turn(conn, conversation_id="c", turn_index=0, role="user",
                occurred_at="2026-05-12T15:00:01Z")
    record_turn(conn, conversation_id="c", turn_index=1, role="assistant",
                occurred_at="2026-05-12T15:00:02Z")

    llm = MagicMock()
    llm.stream_chat.side_effect = lambda **kw: _stream({"episodes": []})

    processed = pump_once(conn=conn, llm=llm, conversations_root=tmp_path)

    assert processed == 2
    rows = conn.execute(
        "SELECT extraction_status FROM raw_turn_refs"
    ).fetchall()
    assert all(r["extraction_status"] == "done" for r in rows)
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/extractor/pump.py
"""Pump that drains the pending raw_turn_refs queue, calling extract_for_turn."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp_servers.memory.extractor.worker import extract_for_turn
from mcp_servers.memory.providers.base import LLMClient
from mcp_servers.memory.repo.raw_turns import list_pending


def pump_once(
    *,
    conn: sqlite3.Connection,
    llm: LLMClient,
    conversations_root: str | Path,
    max_batch: int = 50,
) -> int:
    processed = 0
    for rt in list_pending(conn, limit=max_batch):
        extract_for_turn(
            conn=conn, llm=llm,
            conversations_root=conversations_root,
            raw_turn_id=rt.id,
        )
        processed += 1
    return processed
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_extractor_pump.py -v
git add mcp_servers/memory/extractor/pump.py tests/memory/test_extractor_pump.py
git commit -m "feat(memory): extractor pump (drain pending raw turns)"
```

---

## Phase 4 — Bi-temporal facts data layer

This phase implements the most distinctive piece of the schema: facts with `valid_from / valid_to / learned_at / invalidated_at` and explicit supersession lineage. We get this right and tested *before* the dream cycle uses it.

### Task 4.1: Entities repo (small dependency of facts)

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/entities.py`
- Create: `tests/memory/test_repo_entities.py`

- [ ] **Step 1: Extend models**

Append to `mcp_servers/memory/models.py`:

```python
@dataclass(frozen=True)
class Entity:
    id: str
    canonical_name: str
    kind: str          # 'person'|'project'|'concept'|'tool'|'file'|'place'|'other'
    aliases: list[str]
    summary: str | None
    first_seen: str
    last_seen: str
    importance: float
```

- [ ] **Step 2: Write the failing test**

```python
# tests/memory/test_repo_entities.py
import sqlite3

from mcp_servers.memory.repo.entities import (
    get_by_canonical_name, get_or_create, list_top_importance, touch_seen,
)


def test_get_or_create_inserts_then_returns_existing(conn: sqlite3.Connection) -> None:
    e1 = get_or_create(conn, canonical_name="MCP pool", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    e2 = get_or_create(conn, canonical_name="MCP pool", kind="concept",
                       seen_at="2026-05-12T15:05:00Z")
    assert e1.id == e2.id
    # last_seen should advance
    e3 = get_by_canonical_name(conn, "MCP pool")
    assert e3.last_seen == "2026-05-12T15:05:00Z"


def test_list_top_importance_orders_desc(conn: sqlite3.Connection) -> None:
    a = get_or_create(conn, canonical_name="A", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    b = get_or_create(conn, canonical_name="B", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    conn.execute("UPDATE entities SET importance = 0.9 WHERE id = ?", (a.id,))
    conn.execute("UPDATE entities SET importance = 0.1 WHERE id = ?", (b.id,))
    top = list_top_importance(conn, limit=10)
    assert [t.id for t in top[:2]] == [a.id, b.id]


def test_touch_seen_updates_last_seen(conn: sqlite3.Connection) -> None:
    e = get_or_create(conn, canonical_name="X", kind="concept",
                      seen_at="2026-05-12T15:00:00Z")
    touch_seen(conn, e.id, "2026-05-12T16:00:00Z")
    refreshed = get_by_canonical_name(conn, "X")
    assert refreshed.last_seen == "2026-05-12T16:00:00Z"
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/repo/entities.py
"""Repo for entities (people / projects / concepts / tools / files / ...)."""

from __future__ import annotations

import json
import sqlite3

from mcp_servers.memory.ids import new_entity_id
from mcp_servers.memory.models import Entity


def get_or_create(
    conn: sqlite3.Connection, *, canonical_name: str, kind: str, seen_at: str,
) -> Entity:
    row = conn.execute(
        "SELECT * FROM entities WHERE canonical_name = ?", (canonical_name,)
    ).fetchone()
    if row is not None:
        conn.execute(
            "UPDATE entities SET last_seen = ? WHERE id = ?", (seen_at, row["id"]),
        )
        return _row(_refresh(conn, row["id"]))
    e_id = new_entity_id()
    conn.execute(
        """
        INSERT INTO entities
            (id, canonical_name, kind, aliases, summary, first_seen, last_seen, importance)
        VALUES (?, ?, ?, '[]', NULL, ?, ?, 0.5)
        """,
        (e_id, canonical_name, kind, seen_at, seen_at),
    )
    return _row(_refresh(conn, e_id))


def get_by_canonical_name(conn: sqlite3.Connection, name: str) -> Entity | None:
    row = conn.execute(
        "SELECT * FROM entities WHERE canonical_name = ?", (name,)
    ).fetchone()
    return _row(row) if row else None


def get_by_id(conn: sqlite3.Connection, entity_id: str) -> Entity | None:
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return _row(row) if row else None


def touch_seen(conn: sqlite3.Connection, entity_id: str, seen_at: str) -> None:
    conn.execute(
        "UPDATE entities SET last_seen = ? WHERE id = ?", (seen_at, entity_id),
    )


def list_top_importance(
    conn: sqlite3.Connection, *, limit: int = 50, kind: str | None = None,
) -> list[Entity]:
    if kind is None:
        rows = conn.execute(
            "SELECT * FROM entities ORDER BY importance DESC, last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entities WHERE kind = ? ORDER BY importance DESC, last_seen DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
    return [_row(r) for r in rows]


def _refresh(conn: sqlite3.Connection, entity_id: str) -> sqlite3.Row:
    return conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()


def _row(r: sqlite3.Row) -> Entity:
    return Entity(
        id=r["id"], canonical_name=r["canonical_name"], kind=r["kind"],
        aliases=json.loads(r["aliases"]),
        summary=r["summary"], first_seen=r["first_seen"], last_seen=r["last_seen"],
        importance=r["importance"],
    )
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_entities.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/entities.py tests/memory/test_repo_entities.py
git commit -m "feat(memory): entities repo (canonical-name dedupe, get-or-create, last_seen touch)"
```

---

### Task 4.2: Fact insertion (new fact, no contradictions)

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/facts.py`
- Create: `tests/memory/test_repo_facts.py`

- [ ] **Step 1: Extend models**

```python
# append to mcp_servers/memory/models.py

@dataclass(frozen=True)
class Fact:
    id: str
    subject_entity: str
    predicate: str
    object_entity: str | None
    object_value: str | None
    valid_from: str
    valid_to: str | None
    learned_at: str
    invalidated_at: str | None
    source_episode_ids: list[str]
    confidence: float
    supersedes: str | None
    superseded_by: str | None
    created_in_dream_run: str
```

- [ ] **Step 2: Write the failing test (insertion only)**

```python
# tests/memory/test_repo_facts.py
import sqlite3

from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.facts import (
    current_belief, insert_new_fact,
)


def _ent(conn: sqlite3.Connection, name: str) -> str:
    return get_or_create(
        conn, canonical_name=name, kind="concept",
        seen_at="2026-05-12T15:00:00Z",
    ).id


def test_insert_new_fact_creates_current_belief(conn: sqlite3.Connection) -> None:
    user = _ent(conn, "Travis")
    python = _ent(conn, "Python")
    f = insert_new_fact(
        conn,
        subject_entity=user,
        predicate="uses",
        object_entity=python,
        object_value=None,
        valid_from="2026-05-12T15:00:00Z",
        learned_at="2026-05-12T15:01:00Z",
        source_episode_ids=["ep_a"],
        confidence=0.9,
        created_in_dream_run="dr_test",
    )
    assert f.valid_to is None and f.invalidated_at is None
    assert f.supersedes is None and f.superseded_by is None
    found = current_belief(conn, subject_entity=user, predicate="uses")
    assert found and found.id == f.id
```

- [ ] **Step 3: Implement (insertion + current-belief only; supersession in 4.3)**

```python
# mcp_servers/memory/repo/facts.py
"""Bi-temporal facts repo. Supersession + time-travel queries live here."""

from __future__ import annotations

import json
import sqlite3

from mcp_servers.memory.ids import new_fact_id
from mcp_servers.memory.models import Fact


def insert_new_fact(
    conn: sqlite3.Connection,
    *,
    subject_entity: str,
    predicate: str,
    object_entity: str | None,
    object_value: str | None,
    valid_from: str,
    learned_at: str,
    source_episode_ids: list[str],
    confidence: float,
    created_in_dream_run: str,
    supersedes: str | None = None,
) -> Fact:
    if object_entity is None and object_value is None:
        raise ValueError("must supply object_entity or object_value")
    f_id = new_fact_id()
    conn.execute(
        """
        INSERT INTO facts
            (id, subject_entity, predicate, object_entity, object_value,
             valid_from, valid_to, learned_at, invalidated_at,
             source_episode_ids, confidence, supersedes, superseded_by,
             created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, NULL, ?)
        """,
        (f_id, subject_entity, predicate, object_entity, object_value,
         valid_from, learned_at, json.dumps(source_episode_ids),
         confidence, supersedes, created_in_dream_run),
    )
    return get_by_id(conn, f_id)  # type: ignore[return-value]


def current_belief(
    conn: sqlite3.Connection, *, subject_entity: str, predicate: str,
) -> Fact | None:
    row = conn.execute(
        """
        SELECT * FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        ORDER BY learned_at DESC LIMIT 1
        """,
        (subject_entity, predicate),
    ).fetchone()
    return _row(row) if row else None


def get_by_id(conn: sqlite3.Connection, fact_id: str) -> Fact | None:
    row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return _row(row) if row else None


def _row(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"], subject_entity=r["subject_entity"], predicate=r["predicate"],
        object_entity=r["object_entity"], object_value=r["object_value"],
        valid_from=r["valid_from"], valid_to=r["valid_to"],
        learned_at=r["learned_at"], invalidated_at=r["invalidated_at"],
        source_episode_ids=json.loads(r["source_episode_ids"]),
        confidence=r["confidence"], supersedes=r["supersedes"],
        superseded_by=r["superseded_by"],
        created_in_dream_run=r["created_in_dream_run"],
    )
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_facts.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/facts.py tests/memory/test_repo_facts.py
git commit -m "feat(memory): facts repo — insert + current_belief"
```

---

### Task 4.3: Fact supersession (the bi-temporal core)

**Files:**
- Extend: `mcp_servers/memory/repo/facts.py`
- Extend: `tests/memory/test_repo_facts.py`

- [ ] **Step 1: Append failing tests for supersession**

```python
# append to tests/memory/test_repo_facts.py
import pytest

from mcp_servers.memory.repo.facts import (
    list_facts_for_subject_predicate, supersede_fact,
)


def test_supersede_closes_old_and_creates_new(conn: sqlite3.Connection) -> None:
    user = _ent(conn, "Travis")
    py3_13 = _ent(conn, "Python 3.13")
    py3_14 = _ent(conn, "Python 3.14")
    old = insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=py3_13, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=["ep_1"], confidence=0.9,
        created_in_dream_run="dr_1",
    )

    new = supersede_fact(
        conn, old_fact_id=old.id,
        new_object_entity=py3_14, new_object_value=None,
        change_time="2026-05-12T15:00:00Z",
        source_episode_ids=["ep_2"], confidence=0.95,
        created_in_dream_run="dr_2",
    )

    old_refreshed = get_by_id_via_repo(conn, old.id)
    assert old_refreshed.valid_to == "2026-05-12T15:00:00Z"
    assert old_refreshed.invalidated_at == "2026-05-12T15:00:00Z"
    assert old_refreshed.superseded_by == new.id

    assert new.valid_from == "2026-05-12T15:00:00Z"
    assert new.supersedes == old.id
    assert new.valid_to is None and new.invalidated_at is None

    cb = current_belief(conn, subject_entity=user, predicate="uses")
    assert cb and cb.id == new.id


def get_by_id_via_repo(conn, fact_id):
    from mcp_servers.memory.repo.facts import get_by_id
    return get_by_id(conn, fact_id)


def test_at_most_one_current_belief_per_subject_predicate(
    conn: sqlite3.Connection,
) -> None:
    user = _ent(conn, "Travis")
    a = _ent(conn, "A")
    b = _ent(conn, "B")
    f1 = insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=a, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_1",
    )
    supersede_fact(
        conn, old_fact_id=f1.id,
        new_object_entity=b, new_object_value=None,
        change_time="2026-05-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_2",
    )

    cur = list_facts_for_subject_predicate(
        conn, subject_entity=user, predicate="uses", currently_believed=True,
    )
    assert len(cur) == 1
```

- [ ] **Step 2: Run (expect FAIL)**

- [ ] **Step 3: Implement supersession**

```python
# append to mcp_servers/memory/repo/facts.py

def supersede_fact(
    conn: sqlite3.Connection,
    *,
    old_fact_id: str,
    new_object_entity: str | None,
    new_object_value: str | None,
    change_time: str,
    source_episode_ids: list[str],
    confidence: float,
    created_in_dream_run: str,
) -> Fact:
    old = get_by_id(conn, old_fact_id)
    if old is None:
        raise KeyError(old_fact_id)
    if old.superseded_by is not None:
        raise ValueError(f"fact {old_fact_id} already superseded")

    new = insert_new_fact(
        conn,
        subject_entity=old.subject_entity,
        predicate=old.predicate,
        object_entity=new_object_entity,
        object_value=new_object_value,
        valid_from=change_time,
        learned_at=change_time,
        source_episode_ids=source_episode_ids,
        confidence=confidence,
        created_in_dream_run=created_in_dream_run,
        supersedes=old_fact_id,
    )
    conn.execute(
        """
        UPDATE facts
        SET valid_to = ?, invalidated_at = ?, superseded_by = ?
        WHERE id = ?
        """,
        (change_time, change_time, new.id, old_fact_id),
    )
    return new


def list_facts_for_subject_predicate(
    conn: sqlite3.Connection, *,
    subject_entity: str, predicate: str, currently_believed: bool = False,
) -> list[Fact]:
    if currently_believed:
        rows = conn.execute(
            """
            SELECT * FROM facts
            WHERE subject_entity = ? AND predicate = ?
              AND valid_to IS NULL AND invalidated_at IS NULL
            ORDER BY learned_at DESC
            """,
            (subject_entity, predicate),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM facts
            WHERE subject_entity = ? AND predicate = ?
            ORDER BY learned_at
            """,
            (subject_entity, predicate),
        ).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run (expect PASS) + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_facts.py -v
git add mcp_servers/memory/repo/facts.py tests/memory/test_repo_facts.py
git commit -m "feat(memory): bi-temporal supersession (close old, link supersedes/superseded_by)"
```

---

### Task 4.4: Time-travel queries (`as_of`)

**Files:**
- Extend: `mcp_servers/memory/repo/facts.py`
- Extend: `tests/memory/test_repo_facts.py`

- [ ] **Step 1: Append failing tests**

```python
# append to tests/memory/test_repo_facts.py
from mcp_servers.memory.repo.facts import current_belief_as_of


def test_current_belief_as_of_returns_fact_valid_at_that_time(
    conn: sqlite3.Connection,
) -> None:
    user = _ent(conn, "Travis")
    a = _ent(conn, "Python 3.13")
    b = _ent(conn, "Python 3.14")
    f_old = insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=a, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_1",
    )
    supersede_fact(
        conn, old_fact_id=f_old.id,
        new_object_entity=b, new_object_value=None,
        change_time="2026-05-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_2",
    )

    earlier = current_belief_as_of(
        conn, subject_entity=user, predicate="uses",
        as_of="2026-04-15T00:00:00Z",
    )
    later = current_belief_as_of(
        conn, subject_entity=user, predicate="uses",
        as_of="2026-05-15T00:00:00Z",
    )
    assert earlier and earlier.object_entity == a
    assert later and later.object_entity == b
```

- [ ] **Step 2: Implement**

```python
# append to mcp_servers/memory/repo/facts.py

def current_belief_as_of(
    conn: sqlite3.Connection,
    *,
    subject_entity: str,
    predicate: str,
    as_of: str,
) -> Fact | None:
    row = conn.execute(
        """
        SELECT * FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_from <= ?
          AND (valid_to IS NULL OR valid_to > ?)
          AND learned_at <= ?
          AND (invalidated_at IS NULL OR invalidated_at > ?)
        ORDER BY learned_at DESC LIMIT 1
        """,
        (subject_entity, predicate, as_of, as_of, as_of, as_of),
    ).fetchone()
    return _row(row) if row else None
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_facts.py -v
git add mcp_servers/memory/repo/facts.py tests/memory/test_repo_facts.py
git commit -m "feat(memory): time-travel current_belief_as_of (4-dim bi-temporal query)"
```

---

## Phase 5 — Other repos (reflections, hypotheses, links, dream_runs)

These are flat CRUD; we batch them into one phase. Each task is small.

### Task 5.1: Reflections repo

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/reflections.py`
- Create: `tests/memory/test_repo_reflections.py`

- [ ] **Step 1: Extend models**

```python
# append to mcp_servers/memory/models.py

@dataclass(frozen=True)
class Reflection:
    id: str
    summary: str
    importance: float
    level: int
    source_kind: str       # 'episode_cluster' | 'reflection_cluster'
    source_ids: list[str]
    created_at: str
    created_in_dream_run: str
```

- [ ] **Step 2: Test then implement**

Test:

```python
# tests/memory/test_repo_reflections.py
import sqlite3

from mcp_servers.memory.repo.reflections import (
    insert_reflection, list_by_level,
)


def test_insert_and_list_by_level(conn: sqlite3.Connection) -> None:
    r1 = insert_reflection(
        conn, summary="user prefers terse output",
        importance=0.8, level=1, source_kind="episode_cluster",
        source_ids=["ep_a", "ep_b"], created_in_dream_run="dr_1",
    )
    r2 = insert_reflection(
        conn, summary="prefers brevity in commit messages too",
        importance=0.7, level=2, source_kind="reflection_cluster",
        source_ids=[r1.id], created_in_dream_run="dr_1",
    )
    assert [r.id for r in list_by_level(conn, level=1)] == [r1.id]
    assert [r.id for r in list_by_level(conn, level=2)] == [r2.id]
```

Implementation:

```python
# mcp_servers/memory/repo/reflections.py
"""Reflections repo — recursive synthesized insights."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_reflection_id
from mcp_servers.memory.models import Reflection


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_reflection(
    conn: sqlite3.Connection,
    *,
    summary: str,
    importance: float,
    level: int,
    source_kind: str,
    source_ids: list[str],
    created_in_dream_run: str,
) -> Reflection:
    r_id = new_reflection_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO reflections
            (id, summary, importance, level, source_kind, source_ids,
             created_at, created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (r_id, summary, importance, level, source_kind,
         json.dumps(source_ids), created_at, created_in_dream_run),
    )
    return Reflection(
        id=r_id, summary=summary, importance=importance, level=level,
        source_kind=source_kind, source_ids=source_ids,
        created_at=created_at, created_in_dream_run=created_in_dream_run,
    )


def list_by_level(
    conn: sqlite3.Connection, *, level: int, limit: int = 100,
) -> list[Reflection]:
    rows = conn.execute(
        "SELECT * FROM reflections WHERE level = ? ORDER BY created_at DESC LIMIT ?",
        (level, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def list_recent(
    conn: sqlite3.Connection, *, min_level: int = 1, limit: int = 20,
) -> list[Reflection]:
    rows = conn.execute(
        "SELECT * FROM reflections WHERE level >= ? ORDER BY created_at DESC LIMIT ?",
        (min_level, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Reflection:
    return Reflection(
        id=r["id"], summary=r["summary"], importance=r["importance"],
        level=r["level"], source_kind=r["source_kind"],
        source_ids=json.loads(r["source_ids"]),
        created_at=r["created_at"], created_in_dream_run=r["created_in_dream_run"],
    )
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_reflections.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/reflections.py tests/memory/test_repo_reflections.py
git commit -m "feat(memory): reflections repo (insert, list-by-level, list-recent)"
```

---

### Task 5.2: Hypotheses repo

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/hypotheses.py`
- Create: `tests/memory/test_repo_hypotheses.py`

- [ ] **Step 1: Extend models**

```python
# append to mcp_servers/memory/models.py

@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    source_node_ids: list[str]
    confidence: float
    status: str                   # 'open'|'corroborated'|'refuted'|'set_aside'
    resolved_at: str | None
    resolved_by: str | None
    resolution_note: str | None
    created_at: str
    created_in_dream_run: str
```

- [ ] **Step 2: Test**

```python
# tests/memory/test_repo_hypotheses.py
import sqlite3

from mcp_servers.memory.repo.hypotheses import (
    insert_hypothesis, list_by_status, resolve,
)


def test_insert_lists_under_open_status(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="X relates to Y",
        source_node_ids=["ep_1", "ep_2"], confidence=0.42,
        created_in_dream_run="dr_1",
    )
    assert h.status == "open"
    assert [x.id for x in list_by_status(conn, "open")] == [h.id]


def test_resolve_corroborated(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="A causes B", source_node_ids=[],
        confidence=0.5, created_in_dream_run="dr_1",
    )
    resolve(conn, h.id, status="corroborated",
            resolved_by="operator", note="confirmed in conversation")
    out = list_by_status(conn, "corroborated")
    assert [o.id for o in out] == [h.id]
    assert out[0].resolution_note == "confirmed in conversation"
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/repo/hypotheses.py
"""Hypotheses repo — first-class speculations from the recombine stage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_hypothesis_id
from mcp_servers.memory.models import Hypothesis


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_hypothesis(
    conn: sqlite3.Connection, *,
    statement: str, source_node_ids: list[str],
    confidence: float, created_in_dream_run: str,
) -> Hypothesis:
    h_id = new_hypothesis_id()
    created_at = _now()
    conn.execute(
        """
        INSERT INTO hypotheses
            (id, statement, source_node_ids, confidence, status,
             created_at, created_in_dream_run)
        VALUES (?, ?, ?, ?, 'open', ?, ?)
        """,
        (h_id, statement, json.dumps(source_node_ids), confidence,
         created_at, created_in_dream_run),
    )
    return Hypothesis(
        id=h_id, statement=statement, source_node_ids=source_node_ids,
        confidence=confidence, status="open",
        resolved_at=None, resolved_by=None, resolution_note=None,
        created_at=created_at, created_in_dream_run=created_in_dream_run,
    )


def resolve(
    conn: sqlite3.Connection,
    hypothesis_id: str,
    *,
    status: str,
    resolved_by: str,
    note: str | None = None,
) -> None:
    if status not in ("corroborated", "refuted", "set_aside"):
        raise ValueError(f"invalid status: {status}")
    conn.execute(
        """
        UPDATE hypotheses
        SET status = ?, resolved_at = ?, resolved_by = ?, resolution_note = ?
        WHERE id = ?
        """,
        (status, _now(), resolved_by, note, hypothesis_id),
    )


def list_by_status(
    conn: sqlite3.Connection, status: str, *, limit: int = 100,
) -> list[Hypothesis]:
    rows = conn.execute(
        "SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at DESC LIMIT ?",
        (status, limit),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> Hypothesis:
    return Hypothesis(
        id=r["id"], statement=r["statement"],
        source_node_ids=json.loads(r["source_node_ids"]),
        confidence=r["confidence"], status=r["status"],
        resolved_at=r["resolved_at"], resolved_by=r["resolved_by"],
        resolution_note=r["resolution_note"],
        created_at=r["created_at"], created_in_dream_run=r["created_in_dream_run"],
    )
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_hypotheses.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/hypotheses.py tests/memory/test_repo_hypotheses.py
git commit -m "feat(memory): hypotheses repo (open → corroborated|refuted|set_aside)"
```

---

### Task 5.3: Links repo

**Files:**
- Create: `mcp_servers/memory/repo/links.py`
- Create: `tests/memory/test_repo_links.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_repo_links.py
import sqlite3

from mcp_servers.memory.repo.links import (
    add_link, list_links_from, list_links_to,
)


def test_add_link_idempotent(conn: sqlite3.Connection) -> None:
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="entity", dst_id="en_x",
             link_type="about", weight=1.0, dream_run="dr_1")
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="entity", dst_id="en_x",
             link_type="about", weight=2.0, dream_run="dr_2")
    rows = list_links_from(conn, src_kind="episode", src_id="ep_1")
    assert len(rows) == 1
    # later weight wins on update path: but our UNIQUE prevents duplicates;
    # weight update is a separate operation if needed.


def test_list_links_to_filters_destination(conn: sqlite3.Connection) -> None:
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="fact", dst_id="fa_a",
             link_type="extracted_from", weight=1.0)
    add_link(conn, src_kind="episode", src_id="ep_2",
             dst_kind="fact", dst_id="fa_a",
             link_type="extracted_from", weight=1.0)
    rows = list_links_to(conn, dst_kind="fact", dst_id="fa_a")
    assert {r["src_id"] for r in rows} == {"ep_1", "ep_2"}
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/repo/links.py
"""Typed weighted links — the glue layer for graph traversal."""

from __future__ import annotations

import sqlite3


def add_link(
    conn: sqlite3.Connection,
    *,
    src_kind: str, src_id: str,
    dst_kind: str, dst_id: str,
    link_type: str,
    weight: float = 1.0,
    dream_run: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO links
            (src_kind, src_id, dst_kind, dst_id, link_type, weight,
             created_in_dream_run)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (src_kind, src_id, dst_kind, dst_id, link_type, weight, dream_run),
    )


def list_links_from(
    conn: sqlite3.Connection, *, src_kind: str, src_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM links WHERE src_kind = ? AND src_id = ?",
        (src_kind, src_id),
    ).fetchall()


def list_links_to(
    conn: sqlite3.Connection, *, dst_kind: str, dst_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM links WHERE dst_kind = ? AND dst_id = ?",
        (dst_kind, dst_id),
    ).fetchall()


def all_links(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM links").fetchall()
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_links.py -v
git add mcp_servers/memory/repo/links.py tests/memory/test_repo_links.py
git commit -m "feat(memory): links repo (typed weighted edges, idempotent insert)"
```

---

### Task 5.4: Dream-runs audit repo

**Files:**
- Extend: `mcp_servers/memory/models.py`
- Create: `mcp_servers/memory/repo/dream_runs.py`
- Create: `tests/memory/test_repo_dream_runs.py`

- [ ] **Step 1: Extend models**

```python
# append to mcp_servers/memory/models.py

@dataclass(frozen=True)
class DreamRun:
    id: str
    started_at: str
    ended_at: str | None
    cycle_mode: str           # 'light'|'full'|'maintenance'
    trigger_reason: str       # 'idle_timeout'|'queue_depth'|'scheduled'|'manual'
    stages: dict              # per-stage metrics & timing
    model_used: str
    status: str               # 'running'|'completed'|'failed'|'aborted'
    error: str | None = None
```

- [ ] **Step 2: Test**

```python
# tests/memory/test_repo_dream_runs.py
import sqlite3

from mcp_servers.memory.repo.dream_runs import (
    start_run, finish_run, list_recent, record_stage,
)


def test_start_finish_records_lifecycle(conn: sqlite3.Connection) -> None:
    dr = start_run(conn, cycle_mode="full", trigger_reason="manual",
                   model_used="vllm/gemma-4-31b")
    record_stage(conn, dr.id, name="cluster",
                 metrics={"clusters": 4, "wall_ms": 120})
    finish_run(conn, dr.id, status="completed")

    rows = list_recent(conn, limit=10)
    assert len(rows) == 1
    assert rows[0].status == "completed"
    assert rows[0].stages.get("cluster") == {"clusters": 4, "wall_ms": 120}
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/repo/dream_runs.py
"""Dream-runs audit log + per-stage metrics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_dream_run_id
from mcp_servers.memory.models import DreamRun


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def start_run(
    conn: sqlite3.Connection, *,
    cycle_mode: str, trigger_reason: str, model_used: str,
) -> DreamRun:
    dr_id = new_dream_run_id()
    started_at = _now()
    conn.execute(
        """
        INSERT INTO dream_runs
            (id, started_at, ended_at, cycle_mode, trigger_reason,
             stages, model_used, status)
        VALUES (?, ?, NULL, ?, ?, '{}', ?, 'running')
        """,
        (dr_id, started_at, cycle_mode, trigger_reason, model_used),
    )
    return DreamRun(
        id=dr_id, started_at=started_at, ended_at=None,
        cycle_mode=cycle_mode, trigger_reason=trigger_reason,
        stages={}, model_used=model_used, status="running",
    )


def record_stage(
    conn: sqlite3.Connection, run_id: str, *, name: str, metrics: dict,
) -> None:
    row = conn.execute(
        "SELECT stages FROM dream_runs WHERE id = ?", (run_id,)
    ).fetchone()
    stages = json.loads(row["stages"] or "{}")
    stages[name] = metrics
    conn.execute(
        "UPDATE dream_runs SET stages = ? WHERE id = ?",
        (json.dumps(stages), run_id),
    )


def finish_run(
    conn: sqlite3.Connection, run_id: str, *,
    status: str, error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE dream_runs SET ended_at = ?, status = ?, error = ? WHERE id = ?",
        (_now(), status, error, run_id),
    )


def list_recent(conn: sqlite3.Connection, *, limit: int = 20) -> list[DreamRun]:
    rows = conn.execute(
        "SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> DreamRun:
    return DreamRun(
        id=r["id"], started_at=r["started_at"], ended_at=r["ended_at"],
        cycle_mode=r["cycle_mode"], trigger_reason=r["trigger_reason"],
        stages=json.loads(r["stages"] or "{}"),
        model_used=r["model_used"], status=r["status"], error=r["error"],
    )
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_repo_dream_runs.py -v
git add mcp_servers/memory/models.py mcp_servers/memory/repo/dream_runs.py tests/memory/test_repo_dream_runs.py
git commit -m "feat(memory): dream_runs audit repo (start/finish/per-stage metrics)"
```

---

## Phase 6 — MCP server stub (FastMCP wiring)

This phase stands the memory MCP server up enough that the playground can call `record_turn` from the chat page via MCP, and the agent can call a small set of read tools. Dream-only features (recall PageRank, background pack) come later — we expose simple SQL-backed versions here so the surface exists end-to-end.

### Task 6.1: FastMCP server module with `record_turn` and basic reads

**Files:**
- Create: `mcp_servers/memory/server.py`
- Create: `tests/memory/test_mcp_server.py`

- [ ] **Step 1: Write the failing test**

We test the underlying handler functions directly. FastMCP's transport layer doesn't need to be exercised — the handlers are the contract.

```python
# tests/memory/test_mcp_server.py
import json
import sqlite3
from pathlib import Path

from mcp_servers.memory.server import (
    handle_get_entity,
    handle_record_turn,
    handle_search_episodes,
    handle_search_facts,
)
from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.facts import insert_new_fact


def test_handle_record_turn_writes_row(conn: sqlite3.Connection) -> None:
    out = handle_record_turn(
        conn=conn, conversation_id="c1", turn_index=0,
        role="user", occurred_at="2026-05-12T15:00:01Z",
    )
    assert out["status"] == "ok"
    assert out["raw_turn_id"].startswith("rt_")


def test_handle_search_episodes_filters_by_actor(conn: sqlite3.Connection) -> None:
    insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                   object_entity=None, object_value="alpha", summary="a",
                   importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                   source_refs=[])
    insert_episode(conn, actor="agent", predicate="x", subject_entity=None,
                   object_entity=None, object_value="beta", summary="b",
                   importance=0.5, occurred_at="2026-05-12T15:00:01Z",
                   source_refs=[])
    out = handle_search_episodes(conn=conn, actor="user", limit=10)
    assert [e["summary"] for e in out["episodes"]] == ["a"]


def test_handle_search_facts_default_returns_only_current(
    conn: sqlite3.Connection,
) -> None:
    user = get_or_create(conn, canonical_name="U", kind="person",
                        seen_at="2026-05-12T15:00:00Z").id
    insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=None, object_value="python",
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_x",
    )
    out = handle_search_facts(conn=conn, subject_canonical_name="U")
    assert len(out["facts"]) == 1
    assert out["facts"][0]["object_value"] == "python"


def test_handle_get_entity_returns_dossier(conn: sqlite3.Connection) -> None:
    user = get_or_create(
        conn, canonical_name="MCP pool", kind="concept",
        seen_at="2026-05-12T15:00:00Z",
    )
    out = handle_get_entity(conn=conn, name="MCP pool")
    assert out["entity"]["id"] == user.id
    assert out["entity"]["canonical_name"] == "MCP pool"
    assert isinstance(out["recent_facts"], list)
    assert isinstance(out["recent_episodes"], list)
```

- [ ] **Step 2: Run (expect FAIL)**

- [ ] **Step 3: Implement the server module**

```python
# mcp_servers/memory/server.py
"""FastMCP server for the memory subsystem.

Run standalone (stdio): `python -m mcp_servers.memory.server`
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.repo.entities import (
    get_by_canonical_name, list_top_importance,
)
from mcp_servers.memory.repo.episodes import list_by_status
from mcp_servers.memory.repo.facts import current_belief, _row as _fact_row  # noqa: F401
from mcp_servers.memory.repo.raw_turns import record_turn

_DEFAULT_DB = Path.home() / ".travisml-playground" / "memory.db"


def _open() -> sqlite3.Connection:
    p = Path(os.getenv("TRAVISML_MEMORY_DB", str(_DEFAULT_DB)))
    conn = open_connection(p)
    apply_migrations(conn)
    return conn


# ----- pure handlers (no MCP plumbing — easy to unit-test) ----------------

def handle_record_turn(
    *, conn: sqlite3.Connection,
    conversation_id: str, turn_index: int, role: str, occurred_at: str,
) -> dict[str, Any]:
    rt = record_turn(
        conn, conversation_id=conversation_id, turn_index=turn_index,
        role=role, occurred_at=occurred_at,
    )
    return {"status": "ok", "raw_turn_id": rt.id}


def handle_search_episodes(
    *, conn: sqlite3.Connection,
    actor: str | None = None, since: str | None = None, until: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    clauses = []
    params: list[Any] = []
    if actor:
        clauses.append("actor = ?"); params.append(actor)
    if since:
        clauses.append("occurred_at >= ?"); params.append(since)
    if until:
        clauses.append("occurred_at <= ?"); params.append(until)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM episodes {where} ORDER BY occurred_at DESC LIMIT ?", params
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "actor": r["actor"], "predicate": r["predicate"],
            "summary": r["summary"], "importance": r["importance"],
            "occurred_at": r["occurred_at"],
        })
    return {"episodes": out}


def handle_search_facts(
    *, conn: sqlite3.Connection,
    subject_canonical_name: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_invalidated: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    subject_id = None
    if subject_canonical_name:
        e = get_by_canonical_name(conn, subject_canonical_name)
        if e is None:
            return {"facts": []}
        subject_id = e.id
    clauses = ["1 = 1"]
    params: list[Any] = []
    if subject_id:
        clauses.append("subject_entity = ?"); params.append(subject_id)
    if predicate:
        clauses.append("predicate = ?"); params.append(predicate)
    if as_of:
        clauses += [
            "valid_from <= ?",
            "(valid_to IS NULL OR valid_to > ?)",
            "learned_at <= ?",
            "(invalidated_at IS NULL OR invalidated_at > ?)",
        ]
        params += [as_of, as_of, as_of, as_of]
    elif not include_invalidated:
        clauses += ["valid_to IS NULL", "invalidated_at IS NULL"]
    params.append(limit)
    sql = f"SELECT * FROM facts WHERE {' AND '.join(clauses)} ORDER BY learned_at DESC LIMIT ?"
    rows = conn.execute(sql, params).fetchall()
    facts = [{
        "id": r["id"], "subject_entity": r["subject_entity"],
        "predicate": r["predicate"], "object_entity": r["object_entity"],
        "object_value": r["object_value"], "valid_from": r["valid_from"],
        "valid_to": r["valid_to"], "learned_at": r["learned_at"],
        "confidence": r["confidence"],
    } for r in rows]
    return {"facts": facts}


def handle_get_entity(
    *, conn: sqlite3.Connection, name: str,
) -> dict[str, Any]:
    ent = get_by_canonical_name(conn, name)
    if ent is None:
        return {"entity": None}
    facts = conn.execute(
        """
        SELECT * FROM facts WHERE subject_entity = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        ORDER BY learned_at DESC LIMIT 10
        """, (ent.id,)
    ).fetchall()
    eps = conn.execute(
        """
        SELECT * FROM episodes
        WHERE subject_entity = ? OR object_entity = ?
        ORDER BY occurred_at DESC LIMIT 10
        """, (ent.id, ent.id)
    ).fetchall()
    return {
        "entity": {
            "id": ent.id, "canonical_name": ent.canonical_name,
            "kind": ent.kind, "summary": ent.summary,
            "first_seen": ent.first_seen, "last_seen": ent.last_seen,
            "importance": ent.importance,
        },
        "recent_facts": [{
            "id": r["id"], "predicate": r["predicate"],
            "object_value": r["object_value"], "object_entity": r["object_entity"],
            "learned_at": r["learned_at"],
        } for r in facts],
        "recent_episodes": [{
            "id": r["id"], "actor": r["actor"], "summary": r["summary"],
            "occurred_at": r["occurred_at"],
        } for r in eps],
    }


# ----- FastMCP wiring -----------------------------------------------------

mcp = FastMCP("memory")


@mcp.tool()
def record_turn_tool(
    conversation_id: str, turn_index: int, role: str, occurred_at: str,
) -> dict:
    """Hot-path write: append a raw turn ref. Called by the playground after
    every chat-turn append. Returns the new raw_turn_id."""
    with _open() as c:
        return handle_record_turn(
            conn=c, conversation_id=conversation_id, turn_index=turn_index,
            role=role, occurred_at=occurred_at,
        )


@mcp.tool()
def search_episodes(
    actor: str | None = None, since: str | None = None,
    until: str | None = None, limit: int = 20,
) -> dict:
    """Search atomic episodes by actor and/or time range."""
    with _open() as c:
        return handle_search_episodes(
            conn=c, actor=actor, since=since, until=until, limit=limit,
        )


@mcp.tool()
def search_facts(
    subject_canonical_name: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_invalidated: bool = False,
    limit: int = 50,
) -> dict:
    """Search bi-temporal facts. Defaults to currently-believed facts; pass
    `as_of` for time-travel queries."""
    with _open() as c:
        return handle_search_facts(
            conn=c, subject_canonical_name=subject_canonical_name,
            predicate=predicate, as_of=as_of,
            include_invalidated=include_invalidated, limit=limit,
        )


@mcp.tool()
def get_entity(name: str) -> dict:
    """Look up an entity by canonical name; returns dossier with recent
    facts and recent episodes."""
    with _open() as c:
        return handle_get_entity(conn=c, name=name)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_mcp_server.py -v
git add mcp_servers/memory/server.py tests/memory/test_mcp_server.py
git commit -m "feat(memory): FastMCP server with record_turn + basic read tools"
```

---

### Task 6.2: Register `memory` server in `mcp.json`

**Files:**
- Modify: `mcp.json`

- [ ] **Step 1: Update mcp.json**

Replace `mcp.json` with:

```json
{
  "mcpServers": {
    "notes": {
      "command": "python",
      "args": ["mcp_servers/notes/server.py"],
      "description": "Bundled — agent scratch notes, persists to disk",
      "enabled": true
    },
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_servers.memory.server"],
      "description": "Bundled — persistent cross-conversation memory + dreaming",
      "enabled": true
    }
  }
}
```

- [ ] **Step 2: Smoke-test via Basic Chat**

```bash
streamlit run app.py
# Verify in the sidebar that the "memory" MCP server lists at least
# the tools: record_turn_tool, search_episodes, search_facts, get_entity
```

- [ ] **Step 3: Commit**

```bash
git add mcp.json
git commit -m "build(mcp): register bundled memory server in mcp.json"
```

---

### Task 6.3: Switch the Basic Chat hot-path hook to call the MCP tool

The hook from Task 1.3 currently writes through a direct SQLite connection inside the Streamlit process. Now that the MCP server exists, route through MCP so we have a single writer story. Keep the direct-connection hook as a fallback only for the test suite (the helper function still works).

**Files:**
- Modify: `pages/1_Basic_Chat.py`

- [ ] **Step 1: Replace the direct-DB record path with an MCP tool call**

In `pages/1_Basic_Chat.py`, remove the `_memory_conn()` cache resource and the direct `_mem_record(...)` calls. Replace with a call to the MCP pool's `call_tool` on the memory server. After the user-message append:

```python
if pool and "memory" in enabled_servers:
    try:
        pool.call_tool(
            "memory", "record_turn_tool",
            {
                "conversation_id": conv.id,
                "turn_index": len(conv.data["messages"]) - 1,
                "role": "user",
                "occurred_at": _now_iso(),
            },
        )
    except Exception:
        pass
```

Add the analogous block after the assistant append.

Delete the unused direct-DB imports (`_mem_open`, `_mem_migrate`, `_mem_record`, `@st.cache_resource`).

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
# Send "hello" in Basic Chat. Confirm memory server is enabled in sidebar.
sqlite3 ~/.travisml-playground/memory.db "SELECT id, role, extraction_status FROM raw_turn_refs"
```

Expected: rows present.

- [ ] **Step 3: Commit**

```bash
git add pages/1_Basic_Chat.py
git commit -m "feat(chat): route hot-path record_turn through MCP memory server"
```

---

### Task 6.4: `memory://status` resource

**Files:**
- Extend: `mcp_servers/memory/server.py`
- Extend: `tests/memory/test_mcp_server.py`

- [ ] **Step 1: Test (handler)**

```python
# append to tests/memory/test_mcp_server.py
from mcp_servers.memory.server import handle_status


def test_handle_status_reports_counts(conn: sqlite3.Connection) -> None:
    out = handle_status(conn=conn)
    assert out["counts"]["raw_turn_refs"] == 0
    assert out["counts"]["episodes"] == 0
    assert "last_dream_run" in out
    assert out["last_dream_run"] is None
```

- [ ] **Step 2: Implement + expose as MCP resource**

```python
# append to mcp_servers/memory/server.py

def handle_status(*, conn: sqlite3.Connection) -> dict:
    counts = {}
    for table in ("raw_turn_refs", "episodes", "entities", "facts",
                  "reflections", "hypotheses", "links"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
    last = conn.execute(
        "SELECT id, started_at, ended_at, cycle_mode, status "
        "FROM dream_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return {
        "counts": counts,
        "last_dream_run": dict(last) if last else None,
    }


@mcp.resource("memory://status")
def status_resource() -> str:
    with _open() as c:
        return json.dumps(handle_status(conn=c), indent=2)
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_mcp_server.py -v
git add mcp_servers/memory/server.py tests/memory/test_mcp_server.py
git commit -m "feat(memory): memory://status resource (table counts + last dream run)"
```

---

## Phase 7 — Dreamer skeleton + advisory lock

This phase stands up the dreamer daemon as a process — empty stages for now (we fill them in later). It owns the advisory lock and writes `dream_runs` rows. Tasks 7.1 covers the lock, 7.2 covers the lifecycle, 7.3 covers the runner orchestrator.

### Task 7.1: Advisory write-lock (PID + heartbeat)

**Files:**
- Create: `mcp_servers/memory/dreamer_runner/lifecycle.py` (initially just the lock)
- Create: `tests/memory/test_dreamer_lock.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_dreamer_lock.py
import os
import sqlite3

import pytest

from mcp_servers.memory.dreamer_runner.lifecycle import (
    LockHeld, acquire_lock, heartbeat, release_lock,
)


def test_acquire_then_release(conn: sqlite3.Connection) -> None:
    acquire_lock(conn, pid=os.getpid())
    row = conn.execute("SELECT pid FROM dreamer_lock WHERE id = 1").fetchone()
    assert row["pid"] == os.getpid()
    release_lock(conn, pid=os.getpid())
    row = conn.execute("SELECT * FROM dreamer_lock WHERE id = 1").fetchone()
    assert row is None


def test_acquire_blocks_when_already_held_by_live_pid(
    conn: sqlite3.Connection,
) -> None:
    acquire_lock(conn, pid=os.getpid())
    with pytest.raises(LockHeld):
        acquire_lock(conn, pid=os.getpid() + 99_999, allow_steal_stale=False)


def test_acquire_steals_stale_lock_with_dead_pid(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "INSERT INTO dreamer_lock (id, pid, acquired_at, heartbeat) "
        "VALUES (1, 999999999, '2026-05-12T15:00:00Z', '2026-05-12T15:00:00Z')"
    )
    acquire_lock(conn, pid=os.getpid(), allow_steal_stale=True)
    row = conn.execute("SELECT pid FROM dreamer_lock WHERE id = 1").fetchone()
    assert row["pid"] == os.getpid()


def test_heartbeat_updates_only_for_owner(conn: sqlite3.Connection) -> None:
    acquire_lock(conn, pid=os.getpid())
    heartbeat(conn, pid=os.getpid())
    with pytest.raises(LockHeld):
        heartbeat(conn, pid=os.getpid() + 1)
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/lifecycle.py
"""Dreamer lifecycle helpers: advisory write-lock with PID + heartbeat.

Only one dreamer holds the lock at a time. If a previous dreamer crashed
without releasing, the next dreamer detects the stale lock via PID check
and reclaims it.
"""

from __future__ import annotations

import errno
import os
import sqlite3
from datetime import UTC, datetime


class LockHeld(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
    except OSError as e:
        return e.errno == errno.EPERM


def acquire_lock(
    conn: sqlite3.Connection, *, pid: int, allow_steal_stale: bool = True,
) -> None:
    existing = conn.execute(
        "SELECT pid FROM dreamer_lock WHERE id = 1"
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO dreamer_lock (id, pid, acquired_at, heartbeat) "
            "VALUES (1, ?, ?, ?)",
            (pid, _now(), _now()),
        )
        return
    if existing["pid"] == pid:
        # already ours
        return
    if allow_steal_stale and not _pid_alive(existing["pid"]):
        conn.execute(
            "UPDATE dreamer_lock SET pid = ?, acquired_at = ?, heartbeat = ? "
            "WHERE id = 1",
            (pid, _now(), _now()),
        )
        return
    raise LockHeld(f"dreamer_lock held by pid={existing['pid']}")


def heartbeat(conn: sqlite3.Connection, *, pid: int) -> None:
    cur = conn.execute(
        "UPDATE dreamer_lock SET heartbeat = ? WHERE id = 1 AND pid = ?",
        (_now(), pid),
    )
    if cur.rowcount == 0:
        raise LockHeld(f"not the lock owner: pid={pid}")


def release_lock(conn: sqlite3.Connection, *, pid: int) -> None:
    conn.execute("DELETE FROM dreamer_lock WHERE id = 1 AND pid = ?", (pid,))
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_dreamer_lock.py -v
git add mcp_servers/memory/dreamer_runner/lifecycle.py tests/memory/test_dreamer_lock.py
git commit -m "feat(memory): dreamer advisory lock (PID + heartbeat, stale-reclaim)"
```

---

### Task 7.2: Dreamer process entry + main loop skeleton

**Files:**
- Create: `mcp_servers/memory/dreamer.py`
- Create: `mcp_servers/memory/dreamer_runner/runner.py`
- Create: `tests/memory/test_dreamer_runner.py`

The runner is a callable that does ONE dream cycle. The CLI `dreamer.py` is a thin loop that calls the runner on triggers. Stages are stubs that record their metrics into `dream_runs.stages` — we fill them in starting at Task 8.

- [ ] **Step 1: Test the runner orchestration (with stub stages)**

```python
# tests/memory/test_dreamer_runner.py
import os
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.runner import run_cycle


def test_run_cycle_records_dream_run_with_stages(
    conn: sqlite3.Connection,
) -> None:
    fake_stages = {
        "ingest_cluster":   MagicMock(return_value={"clusters": 0}),
        "consolidate":      MagicMock(return_value={"deduped": 0}),
        "extract":          MagicMock(return_value={"facts_added": 0}),
        "reflect":          MagicMock(return_value={"reflections_added": 0}),
        "recombine":        MagicMock(return_value={"hypotheses_added": 0}),
        "decay_reindex":    MagicMock(return_value={"archived": 0}),
    }
    dr = run_cycle(
        conn=conn, pid=os.getpid(),
        cycle_mode="full", trigger_reason="manual",
        model_used="vllm/test",
        stages=fake_stages,
    )
    assert dr.status == "completed"
    assert set(dr.stages.keys()) == set(fake_stages.keys())


def test_light_cycle_runs_only_subset(conn: sqlite3.Connection) -> None:
    calls = {n: MagicMock(return_value={}) for n in [
        "ingest_cluster", "consolidate", "extract",
        "reflect", "recombine", "decay_reindex",
    ]}
    run_cycle(
        conn=conn, pid=os.getpid(), cycle_mode="light",
        trigger_reason="manual", model_used="vllm/test", stages=calls,
    )
    # light = ingest_cluster, consolidate, extract, decay_reindex
    for name in ("ingest_cluster", "consolidate", "extract", "decay_reindex"):
        calls[name].assert_called_once()
    for name in ("reflect", "recombine"):
        calls[name].assert_not_called()
```

- [ ] **Step 2: Implement the runner**

```python
# mcp_servers/memory/dreamer_runner/runner.py
"""Orchestrates a single dream cycle. Each stage is a callable accepting
(conn, dream_run_id, **kwargs) and returning a metrics dict."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mcp_servers.memory.dreamer_runner.lifecycle import (
    acquire_lock, heartbeat, release_lock,
)
from mcp_servers.memory.models import DreamRun
from mcp_servers.memory.repo.dream_runs import (
    finish_run, record_stage, start_run,
)


_CYCLE_STAGES: dict[str, list[str]] = {
    "light":       ["ingest_cluster", "consolidate", "extract", "decay_reindex"],
    "full":        ["ingest_cluster", "consolidate", "extract",
                    "reflect", "recombine", "decay_reindex"],
    "maintenance": ["decay_reindex"],
}


StageFn = Callable[..., dict[str, Any]]


def run_cycle(
    *,
    conn: sqlite3.Connection,
    pid: int,
    cycle_mode: str,
    trigger_reason: str,
    model_used: str,
    stages: dict[str, StageFn],
) -> DreamRun:
    acquire_lock(conn, pid=pid)
    try:
        dr = start_run(
            conn, cycle_mode=cycle_mode, trigger_reason=trigger_reason,
            model_used=model_used,
        )
        try:
            for name in _CYCLE_STAGES[cycle_mode]:
                fn = stages[name]
                metrics = fn(conn=conn, dream_run_id=dr.id) or {}
                record_stage(conn, dr.id, name=name, metrics=metrics)
                heartbeat(conn, pid=pid)
            finish_run(conn, dr.id, status="completed")
        except Exception as e:
            finish_run(conn, dr.id, status="failed", error=str(e))
            raise
        # re-read for stages dict
        from mcp_servers.memory.repo.dream_runs import list_recent
        return list_recent(conn, limit=1)[0]
    finally:
        release_lock(conn, pid=pid)
```

- [ ] **Step 3: CLI entry**

```python
# mcp_servers/memory/dreamer.py
"""Dreamer daemon CLI.

Usage:
    python -m mcp_servers.memory.dreamer serve              # background loop
    python -m mcp_servers.memory.dreamer run --cycle full   # single cycle, exit
    python -m mcp_servers.memory.dreamer status             # print status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.repo.dream_runs import list_recent

_DEFAULT_DB = Path.home() / ".travisml-playground" / "memory.db"


def _open():
    p = Path(os.getenv("TRAVISML_MEMORY_DB", str(_DEFAULT_DB)))
    conn = open_connection(p)
    apply_migrations(conn)
    return conn


def cmd_run(args: argparse.Namespace) -> int:
    conn = _open()
    stages = all_stages()
    dr = run_cycle(
        conn=conn, pid=os.getpid(),
        cycle_mode=args.cycle,
        trigger_reason="manual",
        model_used=args.model,
        stages=stages,
    )
    print(json.dumps({"dream_run_id": dr.id, "status": dr.status}))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _open()
    recent = list_recent(conn, limit=5)
    print(json.dumps([{
        "id": r.id, "cycle_mode": r.cycle_mode, "status": r.status,
        "started_at": r.started_at, "ended_at": r.ended_at,
        "stages": list(r.stages.keys()),
    } for r in recent], indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Trigger loop. Polls for trigger conditions; runs cycles. Exits on
    SIGINT. v1 cadence is hard-coded; tuning lives in dreamer_config (Phase 16)."""
    conn = _open()
    stages = all_stages()
    while True:
        # Phase 7 stub: no trigger logic yet — just sleep. Phase 8+ adds it.
        time.sleep(60)
        # full cycle every minute is far too aggressive; this stub is
        # replaced by triggers.py in a later phase.
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory.dreamer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--cycle", choices=["light", "full", "maintenance"],
                       default="full")
    p_run.add_argument("--model", default=os.getenv("DREAMER_MODEL", "vllm/local"))
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve")
    p_serve.set_defaults(func=cmd_serve)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Stub stages module**

```python
# mcp_servers/memory/dreamer_runner/stages/__init__.py
"""Stage registry. Each stage is filled in by its own phase (8-13)."""

from __future__ import annotations


def all_stages() -> dict:
    # Real implementations get registered here as they land in Phases 8-13.
    # Until then the runner is callable with mock stages (tests inject them).
    from mcp_servers.memory.dreamer_runner.stages.stage_1_cluster import run as ingest_cluster
    from mcp_servers.memory.dreamer_runner.stages.stage_2_consolidate import run as consolidate
    from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import run as extract
    from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import run as reflect
    from mcp_servers.memory.dreamer_runner.stages.stage_5_recombine import run as recombine
    from mcp_servers.memory.dreamer_runner.stages.stage_6_decay_reindex import run as decay_reindex

    return {
        "ingest_cluster": ingest_cluster,
        "consolidate":    consolidate,
        "extract":        extract,
        "reflect":        reflect,
        "recombine":      recombine,
        "decay_reindex":  decay_reindex,
    }
```

For each of the six stage files (`stage_1_cluster.py` … `stage_6_decay_reindex.py`), create a stub that returns an empty dict:

```python
# mcp_servers/memory/dreamer_runner/stages/stage_1_cluster.py  (and analogous for 2..6)
"""Stage stub — implementation lands in its dedicated phase."""

from __future__ import annotations

import sqlite3


def run(*, conn: sqlite3.Connection, dream_run_id: str, **kwargs) -> dict:
    return {}
```

(Identical bodies for `stage_2_consolidate.py`, `stage_3_extract.py`, `stage_4_reflect.py`, `stage_5_recombine.py`, `stage_6_decay_reindex.py`.)

- [ ] **Step 5: Run all tests + smoke + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_dreamer_runner.py -v
.agent-playground/bin/python -m mcp_servers.memory.dreamer run --cycle full --model vllm/local
.agent-playground/bin/python -m mcp_servers.memory.dreamer status
git add mcp_servers/memory/dreamer.py mcp_servers/memory/dreamer_runner/ tests/memory/test_dreamer_runner.py
git commit -m "feat(memory): dreamer runner + CLI (skeleton; stage stubs)"
```

---

### Task 7.3: Manual `force_dream` MCP tool

**Files:**
- Extend: `mcp_servers/memory/server.py`
- Extend: `tests/memory/test_mcp_server.py`

- [ ] **Step 1: Test the handler**

```python
# append to tests/memory/test_mcp_server.py
from mcp_servers.memory.server import handle_force_dream


def test_handle_force_dream_returns_dream_run_id(conn: sqlite3.Connection) -> None:
    out = handle_force_dream(conn=conn, cycle="maintenance", model="vllm/local")
    assert out["status"] in ("completed", "failed")
    assert out["dream_run_id"].startswith("dr_")
```

- [ ] **Step 2: Implement**

```python
# append to mcp_servers/memory/server.py

import os as _os
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages


def handle_force_dream(
    *, conn: sqlite3.Connection,
    cycle: str = "full",
    model: str = "vllm/local",
) -> dict:
    try:
        dr = run_cycle(
            conn=conn, pid=_os.getpid(),
            cycle_mode=cycle, trigger_reason="manual",
            model_used=model, stages=all_stages(),
        )
        return {"dream_run_id": dr.id, "status": dr.status}
    except Exception as e:
        return {"dream_run_id": None, "status": "failed", "error": str(e)}


@mcp.tool()
def force_dream(cycle: str = "full", model: str = "vllm/local") -> dict:
    """Operator-only: manually trigger a dream cycle. Returns the run id."""
    with _open() as c:
        return handle_force_dream(conn=c, cycle=cycle, model=model)
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_mcp_server.py -v
git add mcp_servers/memory/server.py tests/memory/test_mcp_server.py
git commit -m "feat(memory): force_dream MCP tool (manual cycle trigger)"
```

---

## Phase 8 — Dream stage ①: ingest + cluster

The first stage pulls `fresh` episodes, embeds any missing embeddings, and clusters them. Clusters are ephemeral (not persisted as a table) — they exist within the dream run only.

### Task 8.1: Episode embedder + sqlite-vec writer

**Files:**
- Create: `mcp_servers/memory/retrieval/vector_search.py` (initial — just write/get; search comes in Task 14)
- Create: `tests/memory/test_vector_search.py`

- [ ] **Step 1: Test the writer**

```python
# tests/memory/test_vector_search.py
import sqlite3

from mcp_servers.memory.retrieval.vector_search import (
    has_embedding, upsert_embedding,
)


def test_upsert_then_check_returns_true(conn: sqlite3.Connection) -> None:
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.1] * 768)
    assert has_embedding(conn, "episode", "ep_1") is True
    assert has_embedding(conn, "episode", "ep_999") is False


def test_upsert_replaces_existing(conn: sqlite3.Connection) -> None:
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.1] * 768)
    upsert_embedding(conn, node_kind="episode", node_id="ep_1",
                     embedding=[0.2] * 768)
    # only one row
    rows = conn.execute(
        "SELECT COUNT(*) AS c FROM embeddings WHERE node_id = 'ep_1'"
    ).fetchone()
    assert rows["c"] == 1
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/retrieval/vector_search.py
"""sqlite-vec backed embedding storage + search."""

from __future__ import annotations

import sqlite3
import struct


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def upsert_embedding(
    conn: sqlite3.Connection, *,
    node_kind: str, node_id: str, embedding: list[float],
) -> None:
    conn.execute(
        "DELETE FROM embeddings WHERE node_kind = ? AND node_id = ?",
        (node_kind, node_id),
    )
    conn.execute(
        "INSERT INTO embeddings (node_kind, node_id, embedding) VALUES (?, ?, ?)",
        (node_kind, node_id, _pack(embedding)),
    )


def has_embedding(conn: sqlite3.Connection, node_kind: str, node_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM embeddings WHERE node_kind = ? AND node_id = ?",
        (node_kind, node_id),
    ).fetchone()
    return row is not None
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_vector_search.py -v
git add mcp_servers/memory/retrieval/vector_search.py tests/memory/test_vector_search.py
git commit -m "feat(memory): sqlite-vec embedding upsert + presence check"
```

---

### Task 8.2: Stage 1 — embed missing + agglomerative cluster

**Files:**
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_1_cluster.py`
- Create: `tests/memory/test_stage_1_cluster.py`

- [ ] **Step 1: Test the stage end-to-end with deterministic embedder**

```python
# tests/memory/test_stage_1_cluster.py
import sqlite3

from mcp_servers.memory.dreamer_runner.stages.stage_1_cluster import (
    cluster_episodes, run,
)
from mcp_servers.memory.repo.episodes import insert_episode


def _seed_episodes(conn: sqlite3.Connection, n: int = 6) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        e = insert_episode(
            conn, actor="user", predicate="x",
            subject_entity=None, object_entity=None,
            object_value=f"value-{i // 3}",  # 3 + 3 grouping
            summary=f"summary-{i}", importance=0.5,
            occurred_at=f"2026-05-12T15:00:{i:02d}Z",
            source_refs=[],
        )
        ids.append(e.id)
    return ids


def test_cluster_episodes_groups_similar(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    eps = _seed_episodes(conn, n=6)
    # the fake embedder gives identical vectors for identical strings, so
    # episodes with the same summary cluster together; here summaries are
    # distinct, so we cluster by hash distance — verify the function shape:
    clusters = cluster_episodes(
        episode_ids=eps, embeddings=[fixed_embedder.embed(s) for s in eps],
        distance_threshold=0.5,
    )
    flat = [eid for c in clusters for eid in c]
    assert sorted(flat) == sorted(eps)
    assert all(len(c) >= 1 for c in clusters)


def test_stage_run_writes_metrics_and_embeddings(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    _seed_episodes(conn, n=4)
    metrics = run(conn=conn, dream_run_id="dr_x", embedder=fixed_embedder)
    assert metrics["episodes_seen"] == 4
    assert metrics["clusters"] >= 1
    rows = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()
    assert rows["c"] == 4
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_1_cluster.py
"""Stage ① — embed any missing episode embeddings; cluster fresh episodes."""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.repo.episodes import list_by_status
from mcp_servers.memory.retrieval.vector_search import (
    has_embedding, upsert_embedding,
)


def cluster_episodes(
    *,
    episode_ids: list[str],
    embeddings: list[list[float]],
    distance_threshold: float = 0.5,
) -> list[list[str]]:
    if len(episode_ids) == 0:
        return []
    if len(episode_ids) == 1:
        return [list(episode_ids)]
    X = np.asarray(embeddings, dtype=np.float32)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine", linkage="average",
    )
    labels = model.fit_predict(X)
    groups: dict[int, list[str]] = {}
    for eid, lab in zip(episode_ids, labels):
        groups.setdefault(int(lab), []).append(eid)
    return list(groups.values())


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    embedder: EmbeddingProvider | None = None,
    distance_threshold: float = 0.5,
    **_: Any,
) -> dict[str, int | list[list[str]]]:
    if embedder is None:
        from mcp_servers.memory.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        embedder = SentenceTransformersProvider()

    eps = list_by_status(conn, "fresh")
    if not eps:
        return {"episodes_seen": 0, "clusters": 0, "cluster_ids": []}

    missing = [e for e in eps if not has_embedding(conn, "episode", e.id)]
    if missing:
        vecs = embedder.embed_many([e.summary for e in missing])
        for e, v in zip(missing, vecs):
            upsert_embedding(conn, node_kind="episode", node_id=e.id, embedding=v)

    # gather embeddings for ALL fresh episodes (including ones we just wrote)
    all_vecs: list[list[float]] = []
    summary_vecs = {e.id: v for e, v in zip(missing, vecs)} if missing else {}
    for e in eps:
        if e.id in summary_vecs:
            all_vecs.append(summary_vecs[e.id])
        else:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE node_kind='episode' AND node_id = ?",
                (e.id,),
            ).fetchone()
            import struct
            all_vecs.append(list(struct.unpack(f"{embedder.dim}f", row["embedding"])))

    clusters = cluster_episodes(
        episode_ids=[e.id for e in eps],
        embeddings=all_vecs,
        distance_threshold=distance_threshold,
    )
    return {
        "episodes_seen": len(eps),
        "clusters": len(clusters),
        "cluster_ids": clusters,
    }
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_1_cluster.py -v
git add mcp_servers/memory/dreamer_runner/stages/stage_1_cluster.py tests/memory/test_stage_1_cluster.py
git commit -m "feat(memory): dream stage 1 — embed missing + agglomerative cluster"
```

Note: the runner currently swallows the `embedder` kwarg (stage stubs accept `**kwargs`). For stage 1 to actually receive the embedder + cluster ids downstream, we extend the runner to pass a shared context. We do this small refactor in Task 8.3.

---

### Task 8.3: Pass a shared `ctx` between stages

**Files:**
- Modify: `mcp_servers/memory/dreamer_runner/runner.py`
- Modify: `mcp_servers/memory/dreamer_runner/stages/*.py` (signature update)
- Modify: `tests/memory/test_dreamer_runner.py`
- Modify: `tests/memory/test_stage_1_cluster.py`

- [ ] **Step 1: Update the runner**

Replace the runner module with a version that threads a `ctx` dict between stages:

```python
# mcp_servers/memory/dreamer_runner/runner.py
"""Orchestrates a single dream cycle, threading a `ctx` dict between stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mcp_servers.memory.dreamer_runner.lifecycle import (
    acquire_lock, heartbeat, release_lock,
)
from mcp_servers.memory.models import DreamRun
from mcp_servers.memory.repo.dream_runs import (
    finish_run, list_recent, record_stage, start_run,
)


_CYCLE_STAGES: dict[str, list[str]] = {
    "light":       ["ingest_cluster", "consolidate", "extract", "decay_reindex"],
    "full":        ["ingest_cluster", "consolidate", "extract",
                    "reflect", "recombine", "decay_reindex"],
    "maintenance": ["decay_reindex"],
}


StageFn = Callable[..., dict[str, Any]]


def run_cycle(
    *,
    conn: sqlite3.Connection,
    pid: int,
    cycle_mode: str,
    trigger_reason: str,
    model_used: str,
    stages: dict[str, StageFn],
    ctx: dict[str, Any] | None = None,
) -> DreamRun:
    if ctx is None:
        ctx = {}
    acquire_lock(conn, pid=pid)
    try:
        dr = start_run(
            conn, cycle_mode=cycle_mode, trigger_reason=trigger_reason,
            model_used=model_used,
        )
        try:
            for name in _CYCLE_STAGES[cycle_mode]:
                fn = stages[name]
                result = fn(conn=conn, dream_run_id=dr.id, ctx=ctx) or {}
                # Stage returns either a metrics dict, or a dict with
                # "metrics" + "ctx_updates" keys.
                if "metrics" in result or "ctx_updates" in result:
                    metrics = result.get("metrics", {})
                    ctx.update(result.get("ctx_updates", {}))
                else:
                    metrics = result
                record_stage(conn, dr.id, name=name, metrics=metrics)
                heartbeat(conn, pid=pid)
            finish_run(conn, dr.id, status="completed")
        except Exception as e:
            finish_run(conn, dr.id, status="failed", error=str(e))
            raise
        return list_recent(conn, limit=1)[0]
    finally:
        release_lock(conn, pid=pid)
```

- [ ] **Step 2: Update stage stubs to accept `ctx`**

For each of `stage_1_cluster.py` through `stage_6_decay_reindex.py`, change the `run(...)` signature to accept `ctx: dict[str, Any]` (defaulting to `None`/empty). Stage 1 should now use `ctx` to publish its cluster ids:

```python
# end of stage_1_cluster.py run() — change return:
return {
    "metrics": {
        "episodes_seen": len(eps),
        "clusters": len(clusters),
    },
    "ctx_updates": {
        "cluster_ids":      clusters,        # list[list[episode_id]]
        "episode_index":    {e.id: e for e in eps},
        "embedder":         embedder,
    },
}
```

For each of stages 2-6 stubs, the signature should now be:

```python
def run(*, conn: sqlite3.Connection, dream_run_id: str, ctx: dict, **_: Any) -> dict:
    return {}
```

- [ ] **Step 3: Update tests**

`tests/memory/test_stage_1_cluster.py` — assert the new return shape:

```python
def test_stage_run_writes_metrics_and_embeddings(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    _seed_episodes(conn, n=4)
    ctx: dict = {}
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, embedder=fixed_embedder)
    assert out["metrics"]["episodes_seen"] == 4
    assert out["metrics"]["clusters"] >= 1
    assert "cluster_ids" in out["ctx_updates"]
```

`tests/memory/test_dreamer_runner.py` — pass `ctx=None` and accept either shape:

```python
# update the existing test fakes to include `ctx=...` kwarg in their signatures
fake_stages = {
    name: MagicMock(return_value={}) for name in _CYCLE_STAGES["full"]
}
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_1_cluster.py tests/memory/test_dreamer_runner.py -v
git add mcp_servers/memory/dreamer_runner/ tests/memory/test_dreamer_runner.py tests/memory/test_stage_1_cluster.py
git commit -m "feat(memory): runner threads ctx between stages; stage 1 publishes cluster_ids"
```

---

## Phase 9 — Dream stage ②: consolidate

This stage takes the cluster ids from `ctx["cluster_ids"]`, asks the LLM to identify duplicates per cluster, and marks episodes accordingly.

### Task 9.1: Consolidate prompt + LLM caller

**Files:**
- Create: `mcp_servers/memory/prompts_lib/consolidate.md`
- Create: `mcp_servers/memory/dreamer_runner/llm_calls.py`
- Create: `tests/memory/test_llm_calls.py`

- [ ] **Step 1: Prompt**

```markdown
You are an offline memory consolidator.

Given several atomic memory events from the same conversation cluster,
identify which events are near-duplicates of each other (paraphrases or
restatements of the same underlying fact). Pick the single best
"survivor" per duplicate group.

Return JSON: {"groups": [{"survivor": "<episode_id>", "duplicates": ["<id>", ...]}, ...]}

Events not in any group are treated as their own singleton survivors.
Output ONLY the JSON object.

Events:
{{events}}
```

- [ ] **Step 2: Test the LLM caller helper**

```python
# tests/memory/test_llm_calls.py
import json
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage


def _stream(text):
    yield TextDelta(text=text)
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1),
                          stop_reason="end_turn")


def test_call_json_llm_parses_response() -> None:
    llm = MagicMock()
    llm.stream_chat.return_value = _stream(json.dumps({"groups": []}))
    out = call_json_llm(llm=llm, system="you are x", user="prompt body",
                        max_tokens=500)
    assert out == {"groups": []}


def test_call_json_llm_strips_code_fences() -> None:
    llm = MagicMock()
    llm.stream_chat.return_value = _stream("```json\n" + json.dumps({"k": 1}) + "\n```")
    out = call_json_llm(llm=llm, system="x", user="y", max_tokens=100)
    assert out == {"k": 1}
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/dreamer_runner/llm_calls.py
"""Shared LLM helpers for dream stages — JSON-mode calls + minimal parsing."""

from __future__ import annotations

import json

from mcp_servers.memory.providers.base import (
    ChatMessage, LLMClient, MessageComplete, TextBlock, TextDelta,
)


def _collect(events) -> str:
    out: list[str] = []
    for ev in events:
        if isinstance(ev, TextDelta):
            out.append(ev.text)
        elif isinstance(ev, MessageComplete):
            break
    return "".join(out)


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def call_json_llm(
    *,
    llm: LLMClient,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    events = llm.stream_chat(
        messages=[ChatMessage(role="user", content=[TextBlock(type="text", text=user)])],
        system=system, tools=[],
        max_tokens=max_tokens, temperature=temperature,
    )
    text = _strip_fences(_collect(events))
    return json.loads(text)
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_llm_calls.py -v
git add mcp_servers/memory/prompts_lib/consolidate.md mcp_servers/memory/dreamer_runner/llm_calls.py tests/memory/test_llm_calls.py
git commit -m "feat(memory): shared json-LLM caller + consolidate prompt"
```

---

### Task 9.2: Stage 2 implementation

**Files:**
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_2_consolidate.py`
- Create: `tests/memory/test_stage_2_consolidate.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_stage_2_consolidate.py
import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_2_consolidate import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_consolidate_marks_duplicates_and_survivors(
    conn: sqlite3.Connection,
) -> None:
    a = insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                       object_entity=None, object_value="alpha", summary="A",
                       importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                       source_refs=[])
    b = insert_episode(conn, actor="user", predicate="x", subject_entity=None,
                       object_entity=None, object_value="alpha-2", summary="A2",
                       importance=0.5, occurred_at="2026-05-12T15:00:01Z",
                       source_refs=[])
    c = insert_episode(conn, actor="user", predicate="y", subject_entity=None,
                       object_entity=None, object_value="beta", summary="B",
                       importance=0.5, occurred_at="2026-05-12T15:00:02Z",
                       source_refs=[])
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "groups": [{"survivor": a.id, "duplicates": [b.id]}],
    })

    ctx = {"cluster_ids": [[a.id, b.id, c.id]],
           "episode_index": {a.id: a, b.id: b, c.id: c}}
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm)

    statuses = dict(conn.execute(
        "SELECT id, status FROM episodes"
    ).fetchall())
    assert statuses[a.id] == "consolidated"
    assert statuses[c.id] == "consolidated"
    assert statuses[b.id] == "consolidated"  # duplicate, also marked
    # duplicates should be linked into survivor
    rows = conn.execute(
        "SELECT * FROM links WHERE link_type = 'consolidated_into'"
    ).fetchall()
    assert any(r["src_id"] == b.id and r["dst_id"] == a.id for r in rows)
    assert out["metrics"]["clusters_processed"] == 1
    assert out["metrics"]["duplicates_collapsed"] == 1
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_2_consolidate.py
"""Stage ② — LLM dedup of episodes per cluster."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.repo.episodes import set_status
from mcp_servers.memory.repo.links import add_link

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts_lib" / "consolidate.md"


def _format_events(cluster: list, index: dict) -> str:
    lines = []
    for eid in cluster:
        ep = index.get(eid)
        if ep is None:
            continue
        lines.append(f"- id={eid} importance={ep.importance:.2f} :: {ep.summary}")
    return "\n".join(lines)


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    llm=None,
    max_tokens: int = 800,
    **_: Any,
) -> dict[str, Any]:
    clusters: list[list[str]] = ctx.get("cluster_ids", [])
    index = ctx.get("episode_index", {})
    if not clusters:
        return {"metrics": {"clusters_processed": 0, "duplicates_collapsed": 0}}

    if llm is None:
        from playground.providers.registry import get_client
        llm = get_client("lmstudio", model=ctx.get("model", "local"))

    tpl = _PROMPT_PATH.read_text()
    total_groups = 0
    duplicates = 0
    for cluster in clusters:
        if len(cluster) < 2:
            # singleton — just mark consolidated
            for eid in cluster:
                set_status(conn, eid, "consolidated")
            continue
        user = tpl.replace("{{events}}", _format_events(cluster, index))
        try:
            resp = call_json_llm(
                llm=llm, system="Return only JSON.", user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            # be lenient: if LLM fails, treat every episode as its own survivor
            for eid in cluster:
                set_status(conn, eid, "consolidated")
            continue
        survivors = set()
        for group in resp.get("groups", []):
            survivor = group.get("survivor")
            if survivor is None:
                continue
            survivors.add(survivor)
            for dup in group.get("duplicates", []) or []:
                add_link(
                    conn, src_kind="episode", src_id=dup,
                    dst_kind="episode", dst_id=survivor,
                    link_type="consolidated_into", weight=1.0,
                    dream_run=dream_run_id,
                )
                duplicates += 1
            total_groups += 1
        for eid in cluster:
            set_status(conn, eid, "consolidated")

    return {
        "metrics": {
            "clusters_processed": len(clusters),
            "duplicates_collapsed": duplicates,
            "groups_found": total_groups,
        },
    }
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_2_consolidate.py -v
git add mcp_servers/memory/dreamer_runner/stages/stage_2_consolidate.py tests/memory/test_stage_2_consolidate.py
git commit -m "feat(memory): dream stage 2 — LLM dedup per cluster, link via consolidated_into"
```

---

## Phase 10 — Dream stage ③: extract → bi-temporal facts

This is the heart of semantic crystallization. The LLM proposes facts; the stage looks up matching current-belief facts and either reinforces, supersedes, or creates new.

### Task 10.1: Extract prompt + entity resolution helper

**Files:**
- Create: `mcp_servers/memory/prompts_lib/extract_facts.md`
- Create: `mcp_servers/memory/dreamer_runner/entity_resolve.py`
- Create: `tests/memory/test_entity_resolve.py`

- [ ] **Step 1: Prompt**

```markdown
You extract structured facts from a cluster of related atomic memory
events.

For each fact you assert, give:
- subject:     canonical-cased noun phrase ("Travis", "MCP pool", "Python")
- subject_kind: 'person'|'project'|'concept'|'tool'|'file'|'place'|'other'
- predicate:   snake_case verb phrase ("uses", "prefers", "depends_on")
- object_kind: 'entity' or 'value'
- object:      if object_kind=='entity', canonical-cased noun phrase;
                if object_kind=='value', a literal string ≤ 80 chars
- object_entity_kind: only when object_kind=='entity'; same enum as subject_kind
- confidence: 0.0..1.0 — how confident you are this is actually true
- valid_from_hint: ISO-8601 timestamp inferred from the events, or null

Be conservative. Only assert facts clearly supported by the events. Do
NOT speculate. Return JSON: {"facts": [...]}.

Events in this cluster:
{{events}}
```

- [ ] **Step 2: Test the entity resolver**

```python
# tests/memory/test_entity_resolve.py
import sqlite3

from mcp_servers.memory.dreamer_runner.entity_resolve import resolve_entity


def test_resolve_entity_creates_then_reuses(conn: sqlite3.Connection) -> None:
    a = resolve_entity(conn, canonical="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    b = resolve_entity(conn, canonical="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z")
    assert a == b
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/dreamer_runner/entity_resolve.py
"""Tiny helper used by stages 3+ to turn canonical name + kind into an entity id."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.entities import get_or_create


def resolve_entity(
    conn: sqlite3.Connection, *,
    canonical: str, kind: str, seen_at: str,
) -> str:
    return get_or_create(
        conn, canonical_name=canonical, kind=kind, seen_at=seen_at,
    ).id
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_entity_resolve.py -v
git add mcp_servers/memory/prompts_lib/extract_facts.md mcp_servers/memory/dreamer_runner/entity_resolve.py tests/memory/test_entity_resolve.py
git commit -m "feat(memory): extract-facts prompt + entity resolver helper"
```

---

### Task 10.2: Stage 3 — extract with reinforce / supersede logic

**Files:**
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_3_extract.py`
- Create: `tests/memory/test_stage_3_extract.py`

- [ ] **Step 1: Test (NEW fact path, REINFORCE path, SUPERSEDE path)**

```python
# tests/memory/test_stage_3_extract.py
import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.facts import insert_new_fact, get_by_id


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _ctx_with_cluster(conn) -> dict:
    e = insert_episode(conn, actor="user", predicate="x",
                       subject_entity=None, object_entity=None,
                       object_value="hi", summary="user uses python",
                       importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                       source_refs=[])
    return {
        "cluster_ids": [[e.id]],
        "episode_index": {e.id: e},
    }


def test_extract_creates_new_fact(conn: sqlite3.Connection) -> None:
    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python",
        "object_entity_kind": "concept",
        "confidence": 0.9,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_test", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_added"] == 1
    rows = conn.execute(
        "SELECT * FROM facts WHERE invalidated_at IS NULL AND valid_to IS NULL"
    ).fetchall()
    assert len(rows) == 1


def test_extract_supersedes_when_object_changes(conn: sqlite3.Connection) -> None:
    travis = get_or_create(conn, canonical_name="Travis", kind="person",
                           seen_at="2026-05-12T15:00:00Z").id
    py_old = get_or_create(conn, canonical_name="Python 3.13", kind="concept",
                           seen_at="2026-05-12T15:00:00Z").id
    insert_new_fact(
        conn, subject_entity=travis, predicate="uses",
        object_entity=py_old, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_a",
    )

    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python 3.14",
        "object_entity_kind": "concept",
        "confidence": 0.95,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_b", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_added"] == 1
    assert out["metrics"]["facts_superseded"] == 1
    current = conn.execute(
        """
        SELECT object_entity FROM facts
        WHERE subject_entity = ? AND predicate = ?
          AND valid_to IS NULL AND invalidated_at IS NULL
        """,
        (travis, "uses"),
    ).fetchone()
    assert current is not None  # there is one current belief


def test_extract_reinforces_matching_value(conn: sqlite3.Connection) -> None:
    travis = get_or_create(conn, canonical_name="Travis", kind="person",
                           seen_at="2026-05-12T15:00:00Z").id
    py = get_or_create(conn, canonical_name="Python", kind="concept",
                       seen_at="2026-05-12T15:00:00Z").id
    f = insert_new_fact(
        conn, subject_entity=travis, predicate="uses",
        object_entity=py, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.7, created_in_dream_run="dr_a",
    )
    before = get_by_id(conn, f.id).confidence

    ctx = _ctx_with_cluster(conn)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"facts": [{
        "subject": "Travis", "subject_kind": "person",
        "predicate": "uses",
        "object_kind": "entity", "object": "Python",
        "object_entity_kind": "concept",
        "confidence": 0.9,
        "valid_from_hint": "2026-05-12T15:00:00Z",
    }]})

    out = run(conn=conn, dream_run_id="dr_c", ctx=ctx, llm=llm)
    assert out["metrics"]["facts_reinforced"] == 1
    after = get_by_id(conn, f.id).confidence
    assert after > before
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_3_extract.py
"""Stage ③ — extract semantic facts from clusters; reinforce / supersede /
create as appropriate."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp_servers.memory.dreamer_runner.entity_resolve import resolve_entity
from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.repo.facts import (
    current_belief, insert_new_fact, supersede_fact,
)
from mcp_servers.memory.repo.links import add_link

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts_lib" / "extract_facts.md"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_events(cluster_ids: list[str], index: dict) -> str:
    return "\n".join(
        f"- {eid} :: {index[eid].summary}"
        for eid in cluster_ids if eid in index
    )


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    llm=None,
    max_tokens: int = 1500,
    **_: Any,
) -> dict[str, Any]:
    clusters: list[list[str]] = ctx.get("cluster_ids", [])
    index = ctx.get("episode_index", {})
    if not clusters:
        return {"metrics": {"facts_added": 0, "facts_reinforced": 0, "facts_superseded": 0}}

    if llm is None:
        from playground.providers.registry import get_client
        llm = get_client("lmstudio", model=ctx.get("model", "local"))

    tpl = _PROMPT_PATH.read_text()
    added = 0
    reinforced = 0
    superseded = 0
    seen_at = _now()

    for cluster in clusters:
        user = tpl.replace("{{events}}", _format_events(cluster, index))
        try:
            resp = call_json_llm(
                llm=llm, system="Return only JSON.", user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            continue

        for f in resp.get("facts", []):
            subject_id = resolve_entity(
                conn, canonical=f["subject"], kind=f["subject_kind"],
                seen_at=seen_at,
            )
            obj_entity = None
            obj_value = None
            if f.get("object_kind") == "entity":
                obj_entity = resolve_entity(
                    conn, canonical=f["object"],
                    kind=f.get("object_entity_kind", "other"),
                    seen_at=seen_at,
                )
            else:
                obj_value = str(f.get("object", ""))

            existing = current_belief(
                conn, subject_entity=subject_id, predicate=f["predicate"],
            )

            same_value = (
                existing is not None
                and existing.object_entity == obj_entity
                and existing.object_value == obj_value
            )

            valid_from = f.get("valid_from_hint") or seen_at
            confidence = float(f.get("confidence", 0.7))
            source_eps = list(cluster)

            if existing is None:
                new = insert_new_fact(
                    conn, subject_entity=subject_id, predicate=f["predicate"],
                    object_entity=obj_entity, object_value=obj_value,
                    valid_from=valid_from, learned_at=seen_at,
                    source_episode_ids=source_eps, confidence=confidence,
                    created_in_dream_run=dream_run_id,
                )
                for eid in source_eps:
                    add_link(conn, src_kind="fact", src_id=new.id,
                             dst_kind="episode", dst_id=eid,
                             link_type="extracted_from", weight=1.0,
                             dream_run=dream_run_id)
                added += 1
            elif same_value:
                new_conf = min(1.0, existing.confidence + 0.05)
                conn.execute(
                    "UPDATE facts SET confidence = ? WHERE id = ?",
                    (new_conf, existing.id),
                )
                reinforced += 1
            else:
                new = supersede_fact(
                    conn, old_fact_id=existing.id,
                    new_object_entity=obj_entity, new_object_value=obj_value,
                    change_time=valid_from,
                    source_episode_ids=source_eps, confidence=confidence,
                    created_in_dream_run=dream_run_id,
                )
                add_link(conn, src_kind="fact", src_id=new.id,
                         dst_kind="fact", dst_id=existing.id,
                         link_type="supersedes", weight=1.0,
                         dream_run=dream_run_id)
                superseded += 1
                added += 1

    return {
        "metrics": {
            "facts_added": added,
            "facts_reinforced": reinforced,
            "facts_superseded": superseded,
        }
    }
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_3_extract.py -v
git add mcp_servers/memory/dreamer_runner/stages/stage_3_extract.py tests/memory/test_stage_3_extract.py
git commit -m "feat(memory): dream stage 3 — extract with reinforce / supersede / new-fact paths"
```

---

### Task 10.3: Bi-temporal invariant test (integration)

**Files:**
- Create: `tests/memory/test_bitemporal_invariants.py`

A safety net that exercises stages 1-3 together to confirm the invariant "at most one fact per (subject, predicate) is currently believed" survives realistic runs.

- [ ] **Step 1: Test**

```python
# tests/memory/test_bitemporal_invariants.py
import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import (
    run as run_extract,
)
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def test_no_two_currently_believed_facts_for_same_subject_predicate(
    conn: sqlite3.Connection,
) -> None:
    ep1 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="-", summary="travis uses python 3.13",
        importance=0.5, occurred_at="2026-04-01T00:00:00Z", source_refs=[],
    )
    ep2 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="-", summary="travis upgraded to 3.14",
        importance=0.5, occurred_at="2026-05-01T00:00:00Z", source_refs=[],
    )

    llm = MagicMock()
    llm.stream_chat.side_effect = [
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "uses",
            "object_kind": "entity", "object": "Python 3.13",
            "object_entity_kind": "concept", "confidence": 0.9,
            "valid_from_hint": "2026-04-01T00:00:00Z",
        }]}),
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "uses",
            "object_kind": "entity", "object": "Python 3.14",
            "object_entity_kind": "concept", "confidence": 0.95,
            "valid_from_hint": "2026-05-01T00:00:00Z",
        }]}),
    ]

    ctx1 = {"cluster_ids": [[ep1.id]], "episode_index": {ep1.id: ep1}}
    run_extract(conn=conn, dream_run_id="dr_a", ctx=ctx1, llm=llm)
    ctx2 = {"cluster_ids": [[ep2.id]], "episode_index": {ep2.id: ep2}}
    run_extract(conn=conn, dream_run_id="dr_b", ctx=ctx2, llm=llm)

    rows = conn.execute(
        """
        SELECT subject_entity, predicate, COUNT(*) AS c
        FROM facts
        WHERE valid_to IS NULL AND invalidated_at IS NULL
        GROUP BY subject_entity, predicate
        HAVING c > 1
        """
    ).fetchall()
    assert rows == []
```

- [ ] **Step 2: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_bitemporal_invariants.py -v
git add tests/memory/test_bitemporal_invariants.py
git commit -m "test(memory): bi-temporal invariant — at most one current belief per (S,P)"
```

---

## Phase 11 — Dream stage ④: reflect (recursive)

This stage synthesizes higher-level insights from clusters whose importance exceeds a threshold. Level-1 reflections come from episode clusters; level-2+ are produced periodically by clustering existing reflections.

### Task 11.1: Reflect prompt + stage 4

**Files:**
- Create: `mcp_servers/memory/prompts_lib/reflect.md`
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py`
- Create: `tests/memory/test_stage_4_reflect.py`

- [ ] **Step 1: Prompt**

```markdown
You generate higher-level insights ("reflections") from a related group
of atomic memory events.

Read the events. If there is a clear, well-supported higher-level
insight (a generalization, a pattern, a preference, a recurring theme),
write it as a single sentence ≤ 30 words. If nothing rises above the
particulars, return null.

Output JSON only:
  {"insight": "<sentence>" | null,
   "importance": 0.0..1.0,
   "supporting_event_ids": ["ep_...", ...]}

Events:
{{events}}
```

- [ ] **Step 2: Test**

```python
# tests/memory/test_stage_4_reflect.py
import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _cluster(conn, importance: float):
    e = insert_episode(
        conn, actor="user", predicate="prefers",
        subject_entity=None, object_entity=None, object_value="brevity",
        summary="user prefers brevity", importance=importance,
        occurred_at="2026-05-12T15:00:00Z", source_refs=[],
    )
    return {"cluster_ids": [[e.id]], "episode_index": {e.id: e}}


def test_reflect_creates_level_1_above_threshold(
    conn: sqlite3.Connection,
) -> None:
    ctx = _cluster(conn, importance=0.9)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "insight": "user values terse output across contexts",
        "importance": 0.85,
        "supporting_event_ids": [list(ctx["episode_index"].keys())[0]],
    })
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 1
    rows = conn.execute("SELECT level, summary FROM reflections").fetchall()
    assert rows[0]["level"] == 1


def test_reflect_skips_low_importance_clusters(
    conn: sqlite3.Connection,
) -> None:
    ctx = _cluster(conn, importance=0.1)
    llm = MagicMock()
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 0
    llm.stream_chat.assert_not_called()


def test_reflect_skips_when_insight_is_null(conn: sqlite3.Connection) -> None:
    ctx = _cluster(conn, importance=0.9)
    llm = MagicMock()
    llm.stream_chat.return_value = _stream({"insight": None, "importance": 0.0,
                                            "supporting_event_ids": []})
    out = run(conn=conn, dream_run_id="dr_x", ctx=ctx, llm=llm,
              reflect_threshold=0.7)
    assert out["metrics"]["reflections_added"] == 0
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py
"""Stage ④ — synthesize higher-level reflections from high-importance clusters."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.repo.links import add_link
from mcp_servers.memory.repo.reflections import insert_reflection

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts_lib" / "reflect.md"


def _cluster_importance(cluster: list[str], index: dict) -> float:
    if not cluster:
        return 0.0
    return max(index[eid].importance for eid in cluster if eid in index)


def _format(cluster: list[str], index: dict) -> str:
    return "\n".join(
        f"- {eid} :: {index[eid].summary}"
        for eid in cluster if eid in index
    )


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    llm=None,
    reflect_threshold: float = 0.7,
    max_tokens: int = 600,
    **_: Any,
) -> dict[str, Any]:
    clusters: list[list[str]] = ctx.get("cluster_ids", [])
    index = ctx.get("episode_index", {})
    if not clusters:
        return {"metrics": {"reflections_added": 0}}

    qualifying = [
        c for c in clusters
        if _cluster_importance(c, index) >= reflect_threshold
    ]
    if not qualifying:
        return {"metrics": {"reflections_added": 0}}

    if llm is None:
        from playground.providers.registry import get_client
        llm = get_client("lmstudio", model=ctx.get("model", "local"))

    tpl = _PROMPT_PATH.read_text()
    added = 0
    for cluster in qualifying:
        user = tpl.replace("{{events}}", _format(cluster, index))
        try:
            resp = call_json_llm(
                llm=llm, system="Return only JSON.", user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            continue
        insight = resp.get("insight")
        if not insight:
            continue
        r = insert_reflection(
            conn,
            summary=insight,
            importance=float(resp.get("importance", 0.7)),
            level=1,
            source_kind="episode_cluster",
            source_ids=resp.get("supporting_event_ids") or cluster,
            created_in_dream_run=dream_run_id,
        )
        for eid in cluster:
            add_link(conn, src_kind="reflection", src_id=r.id,
                     dst_kind="episode", dst_id=eid,
                     link_type="reflects", weight=1.0,
                     dream_run=dream_run_id)
        added += 1
    return {"metrics": {"reflections_added": added}}
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_4_reflect.py -v
git add mcp_servers/memory/prompts_lib/reflect.md mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py tests/memory/test_stage_4_reflect.py
git commit -m "feat(memory): dream stage 4 — level-1 reflections from high-importance clusters"
```

---

### Task 11.2: Recursive level-(N+1) reflections every M cycles

**Files:**
- Extend: `mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py`
- Extend: `tests/memory/test_stage_4_reflect.py`

- [ ] **Step 1: Test**

```python
# append to tests/memory/test_stage_4_reflect.py
from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import (
    run_recursive_pass,
)
from mcp_servers.memory.repo.reflections import insert_reflection


def test_run_recursive_pass_creates_level_2(conn: sqlite3.Connection) -> None:
    r1 = insert_reflection(conn, summary="user prefers brevity", importance=0.8,
                           level=1, source_kind="episode_cluster",
                           source_ids=["ep_a"], created_in_dream_run="dr_a")
    r2 = insert_reflection(conn, summary="user dislikes verbose explanations",
                           importance=0.8, level=1, source_kind="episode_cluster",
                           source_ids=["ep_b"], created_in_dream_run="dr_a")

    from unittest.mock import MagicMock
    import json
    from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage

    def _stream(payload):
        yield TextDelta(text=json.dumps(payload))
        yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1),
                              stop_reason="end_turn")

    llm = MagicMock()
    llm.stream_chat.return_value = _stream({
        "insight": "user has a strong overall preference for brevity",
        "importance": 0.9,
        "supporting_event_ids": [r1.id, r2.id],
    })

    n = run_recursive_pass(conn=conn, dream_run_id="dr_b",
                           input_level=1, llm=llm,
                           distance_threshold=2.0)
    assert n >= 1
    rows = conn.execute(
        "SELECT level FROM reflections WHERE level = 2"
    ).fetchall()
    assert len(rows) >= 1
```

- [ ] **Step 2: Implement**

Append to `stage_4_reflect.py`:

```python
# append to mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.repo.reflections import list_by_level


def run_recursive_pass(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    input_level: int,
    llm,
    embedder: EmbeddingProvider | None = None,
    distance_threshold: float = 0.45,
    min_cluster_size: int = 2,
    max_tokens: int = 600,
) -> int:
    refls = list_by_level(conn, level=input_level, limit=200)
    if len(refls) < min_cluster_size:
        return 0
    if embedder is None:
        from mcp_servers.memory.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        embedder = SentenceTransformersProvider()

    vecs = embedder.embed_many([r.summary for r in refls])
    X = np.asarray(vecs, dtype=np.float32)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        metric="cosine", linkage="average",
    )
    labels = model.fit_predict(X)
    groups: dict[int, list] = {}
    for r, lab in zip(refls, labels):
        groups.setdefault(int(lab), []).append(r)

    tpl = _PROMPT_PATH.read_text()
    added = 0
    for group in groups.values():
        if len(group) < min_cluster_size:
            continue
        user = tpl.replace(
            "{{events}}",
            "\n".join(f"- {r.id} (level={r.level}) :: {r.summary}" for r in group),
        )
        try:
            resp = call_json_llm(
                llm=llm, system="Return only JSON.", user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            continue
        insight = resp.get("insight")
        if not insight:
            continue
        new_r = insert_reflection(
            conn, summary=insight,
            importance=float(resp.get("importance", 0.7)),
            level=input_level + 1,
            source_kind="reflection_cluster",
            source_ids=[r.id for r in group],
            created_in_dream_run=dream_run_id,
        )
        for r in group:
            add_link(conn, src_kind="reflection", src_id=new_r.id,
                     dst_kind="reflection", dst_id=r.id,
                     link_type="reflects", weight=1.0,
                     dream_run=dream_run_id)
        added += 1
    return added
```

- [ ] **Step 3: Hook into the stage `run()`**

In `stage_4_reflect.py`, after the existing level-1 pass, add at the end of `run()`:

```python
# call this every Nth dream run; for v1, every run that has >= 4 level-1 reflections
if len(list_by_level(conn, level=1, limit=4)) >= 4:
    recursive_added = run_recursive_pass(
        conn=conn, dream_run_id=dream_run_id,
        input_level=1, llm=llm,
    )
    added += recursive_added
```

(Make sure `list_by_level` is imported at the top of the module.)

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_4_reflect.py -v
git add mcp_servers/memory/dreamer_runner/stages/stage_4_reflect.py tests/memory/test_stage_4_reflect.py
git commit -m "feat(memory): recursive level-(N+1) reflections via reflection-cluster pass"
```

---

## Phase 12 — Dream stage ⑤: recombine (REM-like)

The novel stage. We sample distant node triplets, ask an LLM if there's a surprising connection, and store survivors as `hypotheses`.

### Task 12.1: Triplet sampler with deterministic seed

**Files:**
- Create: `mcp_servers/memory/dreamer_runner/triplet_sampling.py`
- Create: `tests/memory/test_triplet_sampling.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_triplet_sampling.py
import sqlite3

import pytest

from mcp_servers.memory.dreamer_runner.triplet_sampling import sample_triplets


def _seed_nodes(conn: sqlite3.Connection, kinds_ids: list[tuple[str, str]]) -> None:
    # We don't need real rows for the sampler — it takes the candidate list
    # directly. But links influence sampling weights, so we also seed some.
    pass


def test_sample_triplets_deterministic_with_seed() -> None:
    candidates = [("episode", f"ep_{i}") for i in range(10)]
    a = sample_triplets(candidates=candidates, k=4, seed=42, link_lookup=lambda *_: [])
    b = sample_triplets(candidates=candidates, k=4, seed=42, link_lookup=lambda *_: [])
    assert a == b


def test_sample_triplets_returns_k_distinct_triplets() -> None:
    candidates = [("episode", f"ep_{i}") for i in range(12)]
    out = sample_triplets(candidates=candidates, k=6, seed=1, link_lookup=lambda *_: [])
    assert len(out) == 6
    for tr in out:
        assert len(set(tr)) == 3  # three distinct nodes per triplet


def test_sample_triplets_handles_small_pool() -> None:
    candidates = [("episode", "ep_1"), ("episode", "ep_2")]
    out = sample_triplets(candidates=candidates, k=5, seed=1, link_lookup=lambda *_: [])
    assert out == []
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/triplet_sampling.py
"""Triplet sampler — biased toward high graph-distance triplets to make
recombination productive. Deterministic with `seed`."""

from __future__ import annotations

import random
from collections.abc import Callable


def sample_triplets(
    *,
    candidates: list[tuple[str, str]],
    k: int,
    seed: int,
    link_lookup: Callable[[tuple[str, str]], list[tuple[str, str]]],
    bias_distant: bool = True,
) -> list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]:
    if len(candidates) < 3:
        return []
    rng = random.Random(seed)

    def _distant_score(a, b, c) -> float:
        if not bias_distant:
            return 1.0
        # cheap proxy for graph distance: count of direct links between any pair
        directly_linked = 0
        for x, y in [(a, b), (b, c), (a, c)]:
            x_neighbors = set(link_lookup(x))
            if y in x_neighbors:
                directly_linked += 1
        # prefer triplets where no pair is directly linked
        return 1.0 / (1 + directly_linked)

    pool = list(candidates)
    out: list = []
    seen: set[tuple] = set()
    attempts = 0
    while len(out) < k and attempts < 20 * k:
        attempts += 1
        a, b, c = rng.sample(pool, 3)
        key = tuple(sorted([a, b, c]))
        if key in seen:
            continue
        # weighted accept
        score = _distant_score(a, b, c)
        if rng.random() < score:
            out.append((a, b, c))
            seen.add(key)
    return out
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_triplet_sampling.py -v
git add mcp_servers/memory/dreamer_runner/triplet_sampling.py tests/memory/test_triplet_sampling.py
git commit -m "feat(memory): deterministic distance-biased triplet sampler"
```

---

### Task 12.2: Stage 5 — recombine

**Files:**
- Create: `mcp_servers/memory/prompts_lib/recombine.md`
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_5_recombine.py`
- Create: `tests/memory/test_stage_5_recombine.py`

- [ ] **Step 1: Prompt**

```markdown
You are dreaming. You'll be shown three memories from different parts
of a knowledge graph. Most triplets have nothing connecting them. Look
for a non-obvious, plausibly-true connection between them that would be
worth the agent investigating later.

If a connection is plausible: write a single statement ≤ 30 words.
If nothing rises above coincidence: output exactly the word `none`.

Output JSON only:
  {"statement": "<sentence>" | null, "confidence": 0.0..1.0}

Memories:
A) {{a}}
B) {{b}}
C) {{c}}
```

- [ ] **Step 2: Test**

```python
# tests/memory/test_stage_5_recombine.py
import json
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.stages.stage_5_recombine import run
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.episodes import insert_episode


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _seed(conn: sqlite3.Connection, n: int = 5):
    eps = []
    for i in range(n):
        e = insert_episode(
            conn, actor="user", predicate="x",
            subject_entity=None, object_entity=None, object_value=str(i),
            summary=f"event {i}", importance=0.5,
            occurred_at=f"2026-05-12T15:00:{i:02d}Z",
            source_refs=[],
        )
        eps.append(e)
    conn.execute("UPDATE episodes SET status = 'consolidated'")
    return eps


def test_recombine_writes_hypotheses_for_non_none(
    conn: sqlite3.Connection,
) -> None:
    eps = _seed(conn, n=6)
    llm = MagicMock()
    llm.stream_chat.side_effect = [
        _stream({"statement": "event 0 and event 3 may share a cause",
                 "confidence": 0.4}),
        _stream({"statement": None, "confidence": 0.0}),
        _stream({"statement": "events relate to shared workflow",
                 "confidence": 0.5}),
    ]
    out = run(conn=conn, dream_run_id="dr_x", ctx={}, llm=llm,
              k_triplets=3, seed=42)
    rows = conn.execute(
        "SELECT statement, status FROM hypotheses ORDER BY created_at"
    ).fetchall()
    assert all(r["status"] == "open" for r in rows)
    assert out["metrics"]["hypotheses_added"] == len(rows)
    assert out["metrics"]["triplets_sampled"] == 3


def test_recombine_returns_zero_when_too_few_nodes(
    conn: sqlite3.Connection,
) -> None:
    _seed(conn, n=2)
    out = run(conn=conn, dream_run_id="dr_x", ctx={}, llm=MagicMock(),
              k_triplets=3, seed=42)
    assert out["metrics"]["triplets_sampled"] == 0
```

- [ ] **Step 3: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_5_recombine.py
"""Stage ⑤ — REM-like creative recombination of distant memory nodes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.dreamer_runner.triplet_sampling import sample_triplets
from mcp_servers.memory.repo.hypotheses import insert_hypothesis
from mcp_servers.memory.repo.links import add_link, list_links_from

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts_lib" / "recombine.md"


def _candidates(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for kind, table, where in (
        ("episode",    "episodes",    "WHERE status != 'archived'"),
        ("fact",       "facts",       "WHERE invalidated_at IS NULL"),
        ("reflection", "reflections", ""),
    ):
        for row in conn.execute(f"SELECT id FROM {table} {where}").fetchall():
            out.append((kind, row["id"]))
    return out


def _summary_for(conn: sqlite3.Connection, kind: str, node_id: str) -> str:
    if kind == "episode":
        row = conn.execute("SELECT summary FROM episodes WHERE id = ?",
                           (node_id,)).fetchone()
        return row["summary"] if row else f"<missing {kind}/{node_id}>"
    if kind == "fact":
        row = conn.execute(
            "SELECT predicate, object_value FROM facts WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return f"<missing {kind}/{node_id}>"
        return f"{row['predicate']} {row['object_value'] or '(entity)'}"
    if kind == "reflection":
        row = conn.execute("SELECT summary FROM reflections WHERE id = ?",
                           (node_id,)).fetchone()
        return row["summary"] if row else f"<missing {kind}/{node_id}>"
    return f"<unknown {kind}/{node_id}>"


def _link_lookup_factory(conn):
    def _lookup(node: tuple[str, str]) -> list[tuple[str, str]]:
        kind, nid = node
        rows = list_links_from(conn, src_kind=kind, src_id=nid)
        return [(r["dst_kind"], r["dst_id"]) for r in rows]
    return _lookup


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    llm=None,
    k_triplets: int = 8,
    seed: int = 0,
    max_tokens: int = 400,
    **_: Any,
) -> dict[str, Any]:
    cands = _candidates(conn)
    triplets = sample_triplets(
        candidates=cands, k=k_triplets, seed=seed,
        link_lookup=_link_lookup_factory(conn),
    )
    if not triplets:
        return {"metrics": {"triplets_sampled": 0, "hypotheses_added": 0}}

    if llm is None:
        from playground.providers.registry import get_client
        llm = get_client("lmstudio", model=ctx.get("model", "local"))

    tpl = _PROMPT_PATH.read_text()
    added = 0
    for (a, b, c) in triplets:
        user = (
            tpl.replace("{{a}}", _summary_for(conn, *a))
               .replace("{{b}}", _summary_for(conn, *b))
               .replace("{{c}}", _summary_for(conn, *c))
        )
        try:
            resp = call_json_llm(
                llm=llm, system="Return only JSON.", user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            continue
        statement = resp.get("statement")
        if not statement:
            continue
        h = insert_hypothesis(
            conn,
            statement=statement,
            source_node_ids=[f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}", f"{c[0]}/{c[1]}"],
            confidence=float(resp.get("confidence", 0.4)),
            created_in_dream_run=dream_run_id,
        )
        for kind, nid in (a, b, c):
            add_link(conn, src_kind="hypothesis", src_id=h.id,
                     dst_kind=kind, dst_id=nid,
                     link_type="recombines", weight=1.0,
                     dream_run=dream_run_id)
        added += 1

    return {
        "metrics": {
            "triplets_sampled": len(triplets),
            "hypotheses_added": added,
        }
    }
```

- [ ] **Step 4: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_5_recombine.py -v
git add mcp_servers/memory/prompts_lib/recombine.md mcp_servers/memory/dreamer_runner/stages/stage_5_recombine.py tests/memory/test_stage_5_recombine.py
git commit -m "feat(memory): dream stage 5 — REM-like recombination, hypotheses with recombines links"
```

---

## Phase 13 — Dream stage ⑥: decay + reindex (PageRank)

The final stage: archive low-importance nodes, recompute personalized PageRank, refresh dirty embeddings, rebuild the cached background pack.

### Task 13.1: PageRank computation

**Files:**
- Create: `mcp_servers/memory/retrieval/pagerank.py`
- Create: `tests/memory/test_pagerank.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_pagerank.py
import sqlite3

from mcp_servers.memory.retrieval.pagerank import compute_and_store
from mcp_servers.memory.repo.links import add_link


def test_compute_pagerank_writes_scores_for_all_nodes(
    conn: sqlite3.Connection,
) -> None:
    # build a tiny graph: A -> B -> C, A -> C
    add_link(conn, src_kind="entity", src_id="A",
             dst_kind="entity", dst_id="B", link_type="see_also", weight=1.0)
    add_link(conn, src_kind="entity", src_id="B",
             dst_kind="entity", dst_id="C", link_type="see_also", weight=1.0)
    add_link(conn, src_kind="entity", src_id="A",
             dst_kind="entity", dst_id="C", link_type="see_also", weight=1.0)

    n = compute_and_store(conn=conn, dream_run_id="dr_x")
    assert n == 3
    rows = conn.execute(
        "SELECT node_id, score FROM pagerank_scores"
    ).fetchall()
    scores = {r["node_id"]: r["score"] for r in rows}
    # C has more inbound than A or B
    assert scores["C"] >= scores["A"]
    assert scores["C"] >= scores["B"]
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/retrieval/pagerank.py
"""Personalized PageRank over the typed weighted link graph."""

from __future__ import annotations

import sqlite3

import networkx as nx

from mcp_servers.memory.repo.links import all_links


def _build_graph(conn: sqlite3.Connection) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in all_links(conn):
        src = f"{row['src_kind']}/{row['src_id']}"
        dst = f"{row['dst_kind']}/{row['dst_id']}"
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += row["weight"]
        else:
            g.add_edge(src, dst, weight=row["weight"])
    return g


def compute_and_store(
    *, conn: sqlite3.Connection, dream_run_id: str, damping: float = 0.85,
) -> int:
    g = _build_graph(conn)
    if g.number_of_nodes() == 0:
        return 0
    scores = nx.pagerank(g, alpha=damping, max_iter=200, weight="weight")
    rows = []
    for node, score in scores.items():
        kind, nid = node.split("/", 1)
        rows.append((kind, nid, float(score), dream_run_id))
    conn.execute("DELETE FROM pagerank_scores")
    conn.executemany(
        "INSERT INTO pagerank_scores (node_kind, node_id, score, computed_in_dream_run) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def personalized_pagerank(
    *,
    conn: sqlite3.Connection,
    personalization: dict[tuple[str, str], float],
    damping: float = 0.85,
    max_iter: int = 50,
) -> dict[tuple[str, str], float]:
    g = _build_graph(conn)
    if g.number_of_nodes() == 0:
        return {}
    p = {f"{k}/{i}": w for (k, i), w in personalization.items() if f"{k}/{i}" in g}
    if not p:
        return {}
    total = sum(p.values()) or 1.0
    p = {k: v / total for k, v in p.items()}
    scores = nx.pagerank(
        g, alpha=damping, personalization=p,
        max_iter=max_iter, weight="weight",
    )
    return {
        (n.split("/", 1)[0], n.split("/", 1)[1]): float(s)
        for n, s in scores.items()
    }
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_pagerank.py -v
git add mcp_servers/memory/retrieval/pagerank.py tests/memory/test_pagerank.py
git commit -m "feat(memory): PageRank over link graph (default + personalized)"
```

---

### Task 13.2: Decay scoring + archive

**Files:**
- Create: `mcp_servers/memory/dreamer_runner/decay.py`
- Create: `tests/memory/test_decay.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_decay.py
import sqlite3

from mcp_servers.memory.dreamer_runner.decay import archive_bottom_percentile
from mcp_servers.memory.repo.episodes import insert_episode


def test_archive_marks_bottom_percentile(conn: sqlite3.Connection) -> None:
    for i in range(20):
        ep = insert_episode(
            conn, actor="user", predicate="x", subject_entity=None,
            object_entity=None, object_value=str(i), summary=f"e{i}",
            importance=(i / 20.0),
            occurred_at=f"2026-04-{(i % 28) + 1:02d}T00:00:00Z",
            source_refs=[],
        )
        if i < 10:
            conn.execute("UPDATE episodes SET status = 'consolidated' WHERE id = ?", (ep.id,))
    n = archive_bottom_percentile(conn=conn, percentile=0.10)
    assert n >= 1
    archived = conn.execute(
        "SELECT COUNT(*) AS c FROM episodes WHERE status = 'archived'"
    ).fetchone()["c"]
    assert archived == n
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/decay.py
"""Importance-weighted decay: archive the bottom percentile of non-fresh nodes."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime


def _recency_factor(occurred_at: str) -> float:
    try:
        ts = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    except Exception:
        return 0.5
    age_days = max(0.0, (datetime.now(UTC) - ts).total_seconds() / 86400.0)
    # half-life ~ 30 days
    return math.exp(-age_days / 30.0)


def archive_bottom_percentile(
    *, conn: sqlite3.Connection, percentile: float = 0.05,
) -> int:
    rows = conn.execute(
        "SELECT id, importance, occurred_at FROM episodes WHERE status = 'consolidated'"
    ).fetchall()
    if not rows:
        return 0
    scored = [
        (r["id"], r["importance"] * _recency_factor(r["occurred_at"]))
        for r in rows
    ]
    scored.sort(key=lambda kv: kv[1])
    n_archive = max(1, int(len(scored) * percentile))
    targets = [eid for eid, _ in scored[:n_archive]]
    conn.executemany(
        "UPDATE episodes SET status = 'archived' WHERE id = ?",
        [(t,) for t in targets],
    )
    return n_archive
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_decay.py -v
git add mcp_servers/memory/dreamer_runner/decay.py tests/memory/test_decay.py
git commit -m "feat(memory): importance × recency forgetting curve, bottom-percentile archive"
```

---

### Task 13.3: Stage 6 — decay + PageRank reindex + cached background

**Files:**
- Replace: `mcp_servers/memory/dreamer_runner/stages/stage_6_decay_reindex.py`
- Create: `tests/memory/test_stage_6_decay_reindex.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_stage_6_decay_reindex.py
import sqlite3

from mcp_servers.memory.dreamer_runner.stages.stage_6_decay_reindex import run
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.links import add_link


def test_stage_6_runs_decay_and_pagerank(conn: sqlite3.Connection) -> None:
    for i in range(20):
        ep = insert_episode(
            conn, actor="user", predicate="x", subject_entity=None,
            object_entity=None, object_value=str(i), summary=f"e{i}",
            importance=i / 20.0,
            occurred_at=f"2026-04-{(i % 28) + 1:02d}T00:00:00Z",
            source_refs=[],
        )
        conn.execute("UPDATE episodes SET status = 'consolidated' WHERE id = ?", (ep.id,))
    add_link(conn, src_kind="episode", src_id="ep_a",
             dst_kind="episode", dst_id="ep_b",
             link_type="see_also", weight=1.0)

    out = run(conn=conn, dream_run_id="dr_x", ctx={})
    assert "archived" in out["metrics"]
    assert "pagerank_nodes" in out["metrics"]
    cfg = conn.execute(
        "SELECT value FROM dreamer_config WHERE key = 'background_pack_cache'"
    ).fetchone()
    assert cfg is not None
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/stages/stage_6_decay_reindex.py
"""Stage ⑥ — archive low-importance nodes, recompute PageRank, refresh cache."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from mcp_servers.memory.dreamer_runner.decay import archive_bottom_percentile
from mcp_servers.memory.retrieval.pagerank import compute_and_store


def _refresh_background_pack_cache(conn: sqlite3.Connection) -> dict:
    # Top 8 entities by current PageRank score (those with kind='entity')
    rows = conn.execute(
        """
        SELECT p.node_id AS id, p.score AS score, e.canonical_name AS name,
               e.summary AS summary
        FROM pagerank_scores p
        LEFT JOIN entities e ON e.id = p.node_id
        WHERE p.node_kind = 'entity'
        ORDER BY p.score DESC
        LIMIT 8
        """
    ).fetchall()
    entities = [dict(r) for r in rows]

    refl_rows = conn.execute(
        "SELECT id, summary, level FROM reflections "
        "WHERE level >= 1 ORDER BY created_at DESC LIMIT 4"
    ).fetchall()
    reflections = [dict(r) for r in refl_rows]

    cache = {"entities": entities, "reflections": reflections}
    conn.execute(
        """
        INSERT INTO dreamer_config (key, value) VALUES ('background_pack_cache', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (json.dumps(cache),),
    )
    return cache


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    ctx: dict[str, Any],
    archive_percentile: float = 0.05,
    **_: Any,
) -> dict[str, Any]:
    archived = archive_bottom_percentile(
        conn=conn, percentile=archive_percentile,
    )
    pr_nodes = compute_and_store(conn=conn, dream_run_id=dream_run_id)
    cache = _refresh_background_pack_cache(conn)
    return {
        "metrics": {
            "archived": archived,
            "pagerank_nodes": pr_nodes,
            "background_pack_size": len(cache.get("entities", [])),
        }
    }
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_stage_6_decay_reindex.py -v
git add mcp_servers/memory/dreamer_runner/stages/stage_6_decay_reindex.py tests/memory/test_stage_6_decay_reindex.py
git commit -m "feat(memory): dream stage 6 — decay/archive + PageRank + cached background pack"
```

---

## Phase 14 — HippoRAG retrieval (`recall`)

### Task 14.1: Vector top-K seed search

**Files:**
- Extend: `mcp_servers/memory/retrieval/vector_search.py`
- Extend: `tests/memory/test_vector_search.py`

- [ ] **Step 1: Test**

```python
# append to tests/memory/test_vector_search.py
from mcp_servers.memory.retrieval.vector_search import top_k


def test_top_k_returns_ordered_by_similarity(conn) -> None:
    # Two nodes, one similar to query and one far.
    upsert_embedding(conn, node_kind="episode", node_id="ep_close",
                     embedding=[1.0] + [0.0] * 767)
    upsert_embedding(conn, node_kind="episode", node_id="ep_far",
                     embedding=[0.0] + [1.0] + [0.0] * 766)
    out = top_k(conn, query_vec=[1.0] + [0.0] * 767, k=2)
    assert out[0][1] == "ep_close"
    assert out[1][1] == "ep_far"
    assert out[0][2] >= out[1][2]


def test_top_k_can_filter_by_kinds(conn) -> None:
    upsert_embedding(conn, node_kind="episode",    node_id="ep_1",
                     embedding=[1.0] + [0.0] * 767)
    upsert_embedding(conn, node_kind="reflection", node_id="re_1",
                     embedding=[1.0] + [0.0] * 767)
    out = top_k(conn, query_vec=[1.0] + [0.0] * 767, k=5, kinds=["episode"])
    assert [n[0] for n in out] == ["episode"]
```

- [ ] **Step 2: Implement**

```python
# append to mcp_servers/memory/retrieval/vector_search.py

import sqlite_vec  # noqa: F401 -- loaded by open_connection


def top_k(
    conn,
    *,
    query_vec: list[float],
    k: int = 20,
    kinds: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Return [(node_kind, node_id, similarity)] ordered by cosine similarity."""
    where = ""
    params: list = [_pack(query_vec), k]
    if kinds:
        marks = ",".join("?" * len(kinds))
        where = f"AND node_kind IN ({marks})"
        params = [_pack(query_vec)] + list(kinds) + [k]
    sql = (
        f"SELECT node_kind, node_id, distance "
        f"FROM embeddings WHERE embedding MATCH ? {where} "
        f"ORDER BY distance LIMIT ?"
    )
    if kinds:
        sql = (
            "SELECT node_kind, node_id, distance "
            f"FROM embeddings WHERE embedding MATCH ? AND node_kind IN ({marks}) "
            "ORDER BY distance LIMIT ?"
        )
    rows = conn.execute(sql, params).fetchall()
    # sqlite-vec returns distance (lower = closer); convert to similarity.
    return [(r["node_kind"], r["node_id"], 1.0 - float(r["distance"])) for r in rows]
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_vector_search.py -v
git add mcp_servers/memory/retrieval/vector_search.py tests/memory/test_vector_search.py
git commit -m "feat(memory): vector top-k seed search over sqlite-vec embeddings"
```

---

### Task 14.2: HippoRAG-style `recall`

**Files:**
- Create: `mcp_servers/memory/retrieval/recall.py`
- Create: `tests/memory/test_recall.py`

- [ ] **Step 1: Test (uses fixed embedder)**

```python
# tests/memory/test_recall.py
import sqlite3

from mcp_servers.memory.retrieval.recall import recall
from mcp_servers.memory.retrieval.vector_search import upsert_embedding
from mcp_servers.memory.repo.episodes import insert_episode
from mcp_servers.memory.repo.links import add_link


def _seed(conn: sqlite3.Connection, fixed_embedder):
    ep1 = insert_episode(
        conn, actor="user", predicate="x", subject_entity=None,
        object_entity=None, object_value="mcp",
        summary="MCP pool eventloop death", importance=0.7,
        occurred_at="2026-05-12T15:00:00Z", source_refs=[],
    )
    ep2 = insert_episode(
        conn, actor="agent", predicate="x", subject_entity=None,
        object_entity=None, object_value="diag",
        summary="thread holds stale loop reference", importance=0.7,
        occurred_at="2026-05-12T15:00:01Z", source_refs=[],
    )
    upsert_embedding(conn, node_kind="episode", node_id=ep1.id,
                     embedding=fixed_embedder.embed("MCP pool eventloop death"))
    upsert_embedding(conn, node_kind="episode", node_id=ep2.id,
                     embedding=fixed_embedder.embed("thread holds stale loop reference"))
    add_link(conn, src_kind="episode", src_id=ep1.id,
             dst_kind="episode", dst_id=ep2.id,
             link_type="caused", weight=1.0)
    return ep1, ep2


def test_recall_returns_seed_results(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    ep1, _ = _seed(conn, fixed_embedder)
    out = recall(
        conn=conn, query="MCP pool eventloop death",
        embedder=fixed_embedder, max_results=2,
    )
    assert len(out) >= 1
    assert any(r["node_id"] == ep1.id for r in out)


def test_recall_spreads_via_pagerank(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    ep1, ep2 = _seed(conn, fixed_embedder)
    out = recall(
        conn=conn, query="MCP pool", embedder=fixed_embedder, max_results=4,
    )
    ids = {r["node_id"] for r in out}
    # PageRank-spread should pull in ep2 even though its text is different
    assert ep1.id in ids
    assert ep2.id in ids
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/retrieval/recall.py
"""HippoRAG-style recall: vector top-K seed → personalized PageRank spread."""

from __future__ import annotations

import sqlite3
from typing import Any

from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.retrieval.pagerank import personalized_pagerank
from mcp_servers.memory.retrieval.vector_search import top_k


def _hydrate(conn: sqlite3.Connection, kind: str, node_id: str) -> dict | None:
    if kind == "episode":
        r = conn.execute(
            "SELECT id, actor, summary, importance, occurred_at, status "
            "FROM episodes WHERE id = ? AND status != 'archived'",
            (node_id,),
        ).fetchone()
    elif kind == "fact":
        r = conn.execute(
            "SELECT id, subject_entity, predicate, object_entity, object_value, "
            "       valid_from, valid_to, learned_at, confidence "
            "FROM facts WHERE id = ? AND invalidated_at IS NULL",
            (node_id,),
        ).fetchone()
    elif kind == "reflection":
        r = conn.execute(
            "SELECT id, summary, level, importance, created_at "
            "FROM reflections WHERE id = ?", (node_id,),
        ).fetchone()
    elif kind == "entity":
        r = conn.execute(
            "SELECT id, canonical_name, kind, summary, importance "
            "FROM entities WHERE id = ?", (node_id,),
        ).fetchone()
    elif kind == "hypothesis":
        r = conn.execute(
            "SELECT id, statement, status, confidence "
            "FROM hypotheses WHERE id = ? AND status = 'open'", (node_id,),
        ).fetchone()
    else:
        return None
    if r is None:
        return None
    out = dict(r)
    out["node_kind"] = kind
    out["node_id"] = node_id
    return out


def recall(
    *,
    conn: sqlite3.Connection,
    query: str,
    embedder: EmbeddingProvider | None = None,
    max_results: int = 8,
    kinds: list[str] | None = None,
    include_hypotheses: bool = False,
    seed_top_k: int = 20,
    damping: float = 0.85,
) -> list[dict[str, Any]]:
    if embedder is None:
        from mcp_servers.memory.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        embedder = SentenceTransformersProvider()

    seed_kinds = kinds
    if seed_kinds is None and not include_hypotheses:
        seed_kinds = ["episode", "fact", "reflection", "entity"]

    q_vec = embedder.embed(query)
    seeds = top_k(conn, query_vec=q_vec, k=seed_top_k, kinds=seed_kinds)

    if not seeds:
        return []

    pers = {(k, i): max(0.0, s) for (k, i, s) in seeds}
    scores = personalized_pagerank(
        conn=conn, personalization=pers, damping=damping, max_iter=50,
    )

    if not scores:
        # graph is empty / nothing reachable — fall back to seed order
        ranked = [(k, i, s) for (k, i, s) in seeds]
    else:
        ranked = sorted(
            scores.items(), key=lambda kv: kv[1], reverse=True,
        )
        ranked = [(k, i, s) for ((k, i), s) in ranked]

    out: list[dict[str, Any]] = []
    for kind, nid, score in ranked:
        if not include_hypotheses and kind == "hypothesis":
            continue
        hydrated = _hydrate(conn, kind, nid)
        if hydrated is None:
            continue
        hydrated["relevance"] = float(score)
        out.append(hydrated)
        if len(out) >= max_results:
            break
    return out
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_recall.py -v
git add mcp_servers/memory/retrieval/recall.py tests/memory/test_recall.py
git commit -m "feat(memory): HippoRAG recall — vector seed + personalized PageRank spread"
```

---

### Task 14.3: Expose `recall`, `traverse_graph`, `list_hypotheses` via MCP

**Files:**
- Extend: `mcp_servers/memory/server.py`
- Extend: `tests/memory/test_mcp_server.py`

- [ ] **Step 1: Test the new handlers**

```python
# append to tests/memory/test_mcp_server.py
from mcp_servers.memory.server import (
    handle_recall, handle_list_hypotheses, handle_traverse_graph,
)


def test_handle_recall_returns_relevance_scores(
    conn: sqlite3.Connection, fixed_embedder,
) -> None:
    from mcp_servers.memory.repo.episodes import insert_episode
    from mcp_servers.memory.retrieval.vector_search import upsert_embedding
    e = insert_episode(conn, actor="user", predicate="x",
                       subject_entity=None, object_entity=None,
                       object_value="-", summary="hello world",
                       importance=0.5, occurred_at="2026-05-12T15:00:00Z",
                       source_refs=[])
    upsert_embedding(conn, node_kind="episode", node_id=e.id,
                     embedding=fixed_embedder.embed("hello world"))
    out = handle_recall(conn=conn, query="hello world",
                        embedder=fixed_embedder, max_results=5)
    assert isinstance(out["memories"], list)
    if out["memories"]:
        assert "relevance" in out["memories"][0]


def test_handle_list_hypotheses_empty_by_default(conn: sqlite3.Connection) -> None:
    out = handle_list_hypotheses(conn=conn, status="open")
    assert out["hypotheses"] == []


def test_handle_traverse_graph_walks_links(conn: sqlite3.Connection) -> None:
    from mcp_servers.memory.repo.links import add_link
    add_link(conn, src_kind="episode", src_id="ep_a",
             dst_kind="episode", dst_id="ep_b", link_type="see_also")
    out = handle_traverse_graph(
        conn=conn, start_kind="episode", start_id="ep_a", max_hops=2,
    )
    assert "ep_b" in [n["node_id"] for n in out["nodes"]]
```

- [ ] **Step 2: Implement handlers and expose as MCP tools**

```python
# append to mcp_servers/memory/server.py
from mcp_servers.memory.repo.hypotheses import list_by_status as _list_hyp
from mcp_servers.memory.repo.links import list_links_from
from mcp_servers.memory.retrieval.recall import recall as _recall


def handle_recall(
    *, conn: sqlite3.Connection,
    query: str,
    max_results: int = 8,
    kinds: list[str] | None = None,
    embedder=None,
) -> dict:
    memories = _recall(
        conn=conn, query=query, embedder=embedder,
        max_results=max_results, kinds=kinds,
    )
    return {"memories": memories}


def handle_list_hypotheses(
    *, conn: sqlite3.Connection, status: str = "open", limit: int = 10,
) -> dict:
    rows = _list_hyp(conn, status, limit=limit)
    return {"hypotheses": [
        {"id": h.id, "statement": h.statement,
         "confidence": h.confidence, "status": h.status,
         "sources": h.source_node_ids, "created_at": h.created_at}
        for h in rows
    ]}


def handle_traverse_graph(
    *, conn: sqlite3.Connection,
    start_kind: str, start_id: str,
    max_hops: int = 2,
    link_types: list[str] | None = None,
) -> dict:
    seen: set[tuple[str, str]] = {(start_kind, start_id)}
    frontier: list[tuple[str, str]] = [(start_kind, start_id)]
    for _ in range(max_hops):
        next_frontier: list[tuple[str, str]] = []
        for (k, i) in frontier:
            for row in list_links_from(conn, src_kind=k, src_id=i):
                if link_types and row["link_type"] not in link_types:
                    continue
                key = (row["dst_kind"], row["dst_id"])
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append(key)
        frontier = next_frontier
    return {"nodes": [{"node_kind": k, "node_id": i} for (k, i) in seen]}


@mcp.tool()
def recall(query: str, max_results: int = 8, kinds: list[str] | None = None) -> dict:
    """Vector + PageRank-spread retrieval. Returns mixed-kind memories."""
    with _open() as c:
        return handle_recall(conn=c, query=query,
                             max_results=max_results, kinds=kinds)


@mcp.tool()
def list_hypotheses(status: str = "open", limit: int = 10) -> dict:
    """Surface dream-generated speculations."""
    with _open() as c:
        return handle_list_hypotheses(conn=c, status=status, limit=limit)


@mcp.tool()
def traverse_graph(
    start_kind: str, start_id: str,
    max_hops: int = 2, link_types: list[str] | None = None,
) -> dict:
    """Graph walk from a known node, optionally filtered by link types."""
    with _open() as c:
        return handle_traverse_graph(
            conn=c, start_kind=start_kind, start_id=start_id,
            max_hops=max_hops, link_types=link_types,
        )
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_mcp_server.py -v
git add mcp_servers/memory/server.py tests/memory/test_mcp_server.py
git commit -m "feat(memory): MCP tools — recall (HippoRAG), list_hypotheses, traverse_graph"
```

---

## Phase 15 — Background pack (MCP prompt)

### Task 15.1: Build the background-pack Markdown

**Files:**
- Create: `mcp_servers/memory/retrieval/background_pack.py`
- Create: `tests/memory/test_background_pack.py`

- [ ] **Step 1: Test**

```python
# tests/memory/test_background_pack.py
import json
import sqlite3

from mcp_servers.memory.retrieval.background_pack import build_pack


def test_build_pack_uses_cached_snapshot_when_present(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "INSERT INTO dreamer_config (key, value) VALUES ('background_pack_cache', ?)",
        (json.dumps({
            "entities": [
                {"id": "en_1", "name": "MCP pool",
                 "summary": "the MCP server pool",
                 "score": 0.13},
            ],
            "reflections": [
                {"id": "re_1", "summary": "user prefers brevity",
                 "level": 1},
            ],
        }),),
    )
    md = build_pack(conn=conn)
    assert "MCP pool" in md
    assert "user prefers brevity" in md


def test_build_pack_empty_when_no_cache_and_no_data(
    conn: sqlite3.Connection,
) -> None:
    md = build_pack(conn=conn)
    assert "no prior memory" in md.lower()
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/retrieval/background_pack.py
"""Build the Markdown background pack injected at conversation start."""

from __future__ import annotations

import json
import sqlite3


_EMPTY = "_No prior memory is available yet — this is a fresh start._\n"


def build_pack(
    *,
    conn: sqlite3.Connection,
    topic_hint: str | None = None,
    recency_days: int = 7,
) -> str:
    row = conn.execute(
        "SELECT value FROM dreamer_config WHERE key = 'background_pack_cache'"
    ).fetchone()
    if row is None:
        return _EMPTY
    cache = json.loads(row["value"])

    lines: list[str] = []
    lines.append("You have prior memory that may be relevant.")
    lines.append("")

    entities = cache.get("entities", [])
    if entities:
        lines.append("## Top topics you've engaged with")
        for e in entities:
            name = e.get("name") or e.get("canonical_name") or e["id"]
            summary = e.get("summary") or ""
            score = e.get("score")
            tail = f" — {summary}" if summary else ""
            score_s = f" (pagerank={score:.3f})" if isinstance(score, (int, float)) else ""
            lines.append(f"- **{name}**{score_s}{tail}")
        lines.append("")

    refls = cache.get("reflections", [])
    if refls:
        lines.append("## Recent reflections")
        for r in refls:
            lines.append(f"- {r['summary']}")
        lines.append("")

    if not entities and not refls:
        return _EMPTY
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_background_pack.py -v
git add mcp_servers/memory/retrieval/background_pack.py tests/memory/test_background_pack.py
git commit -m "feat(memory): build_pack assembles markdown background from cached snapshot"
```

---

### Task 15.2: Expose `background` as an MCP prompt

**Files:**
- Extend: `mcp_servers/memory/server.py`
- Extend: `tests/memory/test_mcp_server.py`

- [ ] **Step 1: Test**

```python
# append to tests/memory/test_mcp_server.py
import json as _json

from mcp_servers.memory.server import handle_background_prompt


def test_handle_background_prompt_returns_user_role_text(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "INSERT INTO dreamer_config (key, value) VALUES ('background_pack_cache', ?)",
        (_json.dumps({
            "entities": [{"id": "en_1", "name": "MCP pool",
                          "summary": "the pool", "score": 0.1}],
            "reflections": [],
        }),),
    )
    out = handle_background_prompt(conn=conn)
    assert out["role"] == "user"
    assert "MCP pool" in out["content"]
```

- [ ] **Step 2: Implement**

```python
# append to mcp_servers/memory/server.py

from mcp_servers.memory.retrieval.background_pack import build_pack


def handle_background_prompt(
    *, conn: sqlite3.Connection,
    topic_hint: str | None = None,
    recency_days: int = 7,
) -> dict:
    md = build_pack(conn=conn, topic_hint=topic_hint, recency_days=recency_days)
    return {"role": "user", "content": md}


@mcp.prompt()
def background(topic_hint: str = "", recency_days: int = 7) -> str:
    """Background memory pack — call at conversation start; prepend the
    returned text to your system prompt."""
    with _open() as c:
        return handle_background_prompt(
            conn=c, topic_hint=topic_hint or None, recency_days=recency_days,
        )["content"]
```

- [ ] **Step 3: Wire into Basic Chat at conversation start**

In `pages/1_Basic_Chat.py`, near the existing system-prompt assembly block (where `system_prompt = st.sidebar.text_area(...)` lands), after the user's chosen `system_prompt` is established, prepend the background pack:

```python
# Insert near the "system prompt" sidebar block:
if pool and "memory" in enabled_servers:
    try:
        msgs = pool.get_prompt("memory", "background", {})
        bg = "\n\n".join(m["content"][0]["text"] for m in msgs)
        if bg.strip():
            system_prompt = (bg + "\n\n" + (system_prompt or "")).strip()
    except Exception:
        pass  # degrade silently
```

- [ ] **Step 4: Smoke + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_mcp_server.py -v
streamlit run app.py   # send a message, verify no errors
git add mcp_servers/memory/server.py tests/memory/test_mcp_server.py pages/1_Basic_Chat.py
git commit -m "feat(memory): background MCP prompt + auto-prepend in Basic Chat"
```

---

## Phase 16 — Dreaming page (Streamlit operator UI)

### Task 16.1: Daemon controller (start/stop/status helpers)

**Files:**
- Create: `mcp_servers/memory/dreamer_runner/control.py`
- Create: `tests/memory/test_dreamer_control.py`

The Dreaming page uses `subprocess.Popen` to manage the daemon. Wrap the lifecycle in a small controller so the page logic stays clean.

- [ ] **Step 1: Test (PID file roundtrip)**

```python
# tests/memory/test_dreamer_control.py
from pathlib import Path

from mcp_servers.memory.dreamer_runner.control import (
    DaemonController, write_pid_file, read_pid_file, clear_pid_file,
)


def test_pid_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.pid"
    write_pid_file(p, 12345)
    assert read_pid_file(p) == 12345
    clear_pid_file(p)
    assert read_pid_file(p) is None


def test_controller_status_reports_not_running_when_no_pid(tmp_path: Path) -> None:
    c = DaemonController(pid_file=tmp_path / "dreamer.pid")
    assert c.status() == {"running": False, "pid": None}
```

- [ ] **Step 2: Implement**

```python
# mcp_servers/memory/dreamer_runner/control.py
"""Lifecycle controller used by the Streamlit Dreaming page."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path


def write_pid_file(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def read_pid_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def clear_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as e:
        return e.errno == errno.EPERM


class DaemonController:
    def __init__(
        self,
        pid_file: Path | None = None,
        python: str | None = None,
    ) -> None:
        self.pid_file = pid_file or (
            Path.home() / ".travisml-playground" / "dreamer.pid"
        )
        self.python = python or sys.executable

    def status(self) -> dict:
        pid = read_pid_file(self.pid_file)
        if pid is None:
            return {"running": False, "pid": None}
        if not _alive(pid):
            clear_pid_file(self.pid_file)
            return {"running": False, "pid": None}
        return {"running": True, "pid": pid}

    def start(self) -> int:
        st = self.status()
        if st["running"]:
            return int(st["pid"])
        proc = subprocess.Popen(
            [self.python, "-m", "mcp_servers.memory.dreamer", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        write_pid_file(self.pid_file, proc.pid)
        return proc.pid

    def stop(self) -> bool:
        st = self.status()
        if not st["running"]:
            return False
        try:
            os.kill(int(st["pid"]), 15)  # SIGTERM
        finally:
            clear_pid_file(self.pid_file)
        return True
```

- [ ] **Step 3: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_dreamer_control.py -v
git add mcp_servers/memory/dreamer_runner/control.py tests/memory/test_dreamer_control.py
git commit -m "feat(memory): DaemonController for Streamlit lifecycle (start/stop/status, PID file)"
```

---

### Task 16.2: `pages/2_Dreaming.py` — daemon panel + recent dreams

**Files:**
- Create: `pages/2_Dreaming.py`

This task creates the page and renders the daemon panel + recent dreams table. The remaining sections (hypotheses, entity browser, settings) land in 16.3 and 16.4.

- [ ] **Step 1: Implement the page (daemon panel + recent dreams)**

```python
# pages/2_Dreaming.py
"""Operator console for the memory dreamer."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.control import DaemonController
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.repo.dream_runs import list_recent
from playground.branding import (
    inject_brand_css, render_brand_wordmark, render_theme_toggle,
)

load_dotenv()
st.set_page_config(
    page_title="Dreaming — TravisML Playground",
    page_icon="◐", layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
inject_brand_css()
render_brand_wordmark()


@st.cache_resource(show_spinner=False)
def _conn():
    from pathlib import Path
    p = Path.home() / ".travisml-playground" / "memory.db"
    c = open_connection(p)
    apply_migrations(c)
    return c


@st.cache_resource(show_spinner=False)
def _controller() -> DaemonController:
    return DaemonController()


st.html('<h1 style="font-size:36px;margin-bottom:8px;">Dream<em>ing</em></h1>')
st.caption("Operator console for the memory + dreaming subsystem.")
st.divider()


# ----- daemon panel -----
ctrl = _controller()
conn = _conn()
status = ctrl.status()

cols = st.columns([2, 2, 6])
with cols[0]:
    st.html('<div class="tml-label">Daemon</div>')
    st.write("running" if status["running"] else "stopped")
    if status["running"]:
        st.caption(f"pid {status['pid']}")

with cols[1]:
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("raw_turn_refs", "episodes", "facts", "reflections",
                  "hypotheses", "links")
    }
    st.html('<div class="tml-label">Memory size</div>')
    for k, v in counts.items():
        st.caption(f"{k}: {v}")

with cols[2]:
    st.html('<div class="tml-label">Controls</div>')
    btn_cols = st.columns(3)
    if btn_cols[0].button(
        "Start daemon" if not status["running"] else "Stop daemon",
        use_container_width=True,
    ):
        if status["running"]:
            ctrl.stop()
        else:
            ctrl.start()
        st.rerun()
    cycle = btn_cols[1].selectbox(
        "Dream now…", ["light", "full", "maintenance"], key="cycle_choice",
    )
    if btn_cols[2].button("Dream now", use_container_width=True):
        try:
            import os
            run_cycle(
                conn=conn, pid=os.getpid(),
                cycle_mode=cycle, trigger_reason="manual",
                model_used="vllm/local", stages=all_stages(),
            )
            st.success("dream cycle completed")
        except Exception as e:
            st.error(f"dream failed: {e}")
        st.rerun()

st.divider()


# ----- recent dreams -----
st.html('<div class="tml-label">Recent dreams</div>')
recent = list_recent(conn, limit=10)
if not recent:
    st.caption("No dream runs yet.")
else:
    for dr in recent:
        with st.container(border=True):
            head = f"**{dr.started_at}** · {dr.cycle_mode} · {dr.status}"
            if dr.ended_at:
                head += f" · ended {dr.ended_at}"
            st.markdown(head)
            if dr.stages:
                cells = st.columns(min(6, max(1, len(dr.stages))))
                for i, (name, metrics) in enumerate(dr.stages.items()):
                    with cells[i % len(cells)]:
                        st.caption(name)
                        st.json(metrics, expanded=False)
            if dr.error:
                st.error(dr.error)

render_theme_toggle()
```

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
# Open "Dreaming" in the sidebar.
# Click "Start daemon" → status should flip to running.
# Click "Dream now" with cycle="maintenance" → expect a "dream cycle completed" toast.
# Refresh — the recent dreams panel should list one dr_ row.
```

- [ ] **Step 3: Commit**

```bash
git add pages/2_Dreaming.py
git commit -m "feat(dreaming): operator page — daemon panel + recent-dreams audit"
```

---

### Task 16.3: Dreaming page — open hypotheses with Corroborate/Refute/Set-aside

**Files:**
- Extend: `pages/2_Dreaming.py`

- [ ] **Step 1: Append the hypotheses section**

After the recent-dreams block in `pages/2_Dreaming.py`, add:

```python
from mcp_servers.memory.repo.hypotheses import list_by_status, resolve

st.divider()
st.html('<div class="tml-label">Open hypotheses</div>')

open_hyps = list_by_status(conn, "open", limit=20)
if not open_hyps:
    st.caption("No open hypotheses yet — run a full dream cycle to generate some.")
else:
    for h in open_hyps:
        with st.container(border=True):
            st.markdown(f"**?** {h.statement}")
            st.caption(
                f"sources: {', '.join(h.source_node_ids[:3])} "
                f"· confidence {h.confidence:.2f} · created {h.created_at}"
            )
            c1, c2, c3, _ = st.columns([2, 2, 2, 6])
            if c1.button("Corroborate", key=f"corr_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="corroborated", resolved_by="operator")
                st.rerun()
            if c2.button("Refute", key=f"ref_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="refuted", resolved_by="operator")
                st.rerun()
            if c3.button("Set aside", key=f"aside_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="set_aside", resolved_by="operator")
                st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add pages/2_Dreaming.py
git commit -m "feat(dreaming): open-hypotheses panel with corroborate/refute/set-aside"
```

---

### Task 16.4: Dreaming page — entity browser + settings (collapsible)

**Files:**
- Extend: `pages/2_Dreaming.py`

- [ ] **Step 1: Append entity browser + settings**

```python
# append to pages/2_Dreaming.py
import json

from mcp_servers.memory.repo.entities import list_top_importance

st.divider()
st.html('<div class="tml-label">Entity browser</div>')

q = st.text_input("search entities", key="_entity_search")
if q:
    rows = conn.execute(
        "SELECT * FROM entities WHERE canonical_name LIKE ? "
        "ORDER BY importance DESC LIMIT 25",
        (f"%{q}%",),
    ).fetchall()
else:
    rows = [{
        "id": e.id, "canonical_name": e.canonical_name, "kind": e.kind,
        "summary": e.summary, "importance": e.importance,
    } for e in list_top_importance(conn, limit=25)]

for r in rows:
    name = r["canonical_name"] if isinstance(r, dict) else r["canonical_name"]
    kind = r["kind"] if isinstance(r, dict) else r["kind"]
    eid = r["id"] if isinstance(r, dict) else r["id"]
    st.caption(f"**{name}** · kind={kind} · id={eid}")


with st.expander("Settings", expanded=False):
    st.html('<div class="tml-label">Trigger thresholds</div>')
    row = conn.execute(
        "SELECT value FROM dreamer_config WHERE key = 'triggers'"
    ).fetchone()
    cfg = json.loads(row["value"]) if row else {
        "light_min_episodes": 20,
        "light_interval_min": 15,
        "full_idle_min": 30,
        "scheduled_full_at": "03:30",
    }
    cfg["light_min_episodes"] = st.number_input(
        "Light cycle: min pending episodes",
        min_value=1, value=int(cfg["light_min_episodes"]),
    )
    cfg["light_interval_min"] = st.number_input(
        "Light cycle: every N minutes of activity",
        min_value=1, value=int(cfg["light_interval_min"]),
    )
    cfg["full_idle_min"] = st.number_input(
        "Full cycle: after N idle minutes",
        min_value=1, value=int(cfg["full_idle_min"]),
    )
    cfg["scheduled_full_at"] = st.text_input(
        "Scheduled full cycle (HH:MM, 24h)",
        value=str(cfg["scheduled_full_at"]),
    )
    if st.button("Save settings"):
        conn.execute(
            "INSERT INTO dreamer_config (key, value) VALUES ('triggers', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (json.dumps(cfg),),
        )
        st.success("saved")
```

- [ ] **Step 2: Smoke-test and commit**

```bash
streamlit run app.py
# Open Dreaming page. Confirm:
#   - entity browser shows the entities created by your prior smoke runs
#   - search filters them
#   - settings expander saves to dreamer_config
git add pages/2_Dreaming.py
git commit -m "feat(dreaming): entity browser + collapsible settings (trigger thresholds)"
```

---

## Phase 17 — Eval harness

The eval harness lives in `tests/eval/memory/` and is run on demand (not by default `pytest`). It seeds conversations, runs a full dream cycle, asks held-out questions via `recall`, and grades answers with an LLM judge.

### Task 17.1: Scenario format + first scenario

**Files:**
- Create: `tests/eval/memory/conftest.py`
- Create: `tests/eval/memory/scenarios/01_user_preferences/conversations/2026-05-01.json`
- Create: `tests/eval/memory/scenarios/01_user_preferences/questions.yaml`
- Create: `tests/eval/memory/scenarios/01_user_preferences/expected.yaml`

- [ ] **Step 1: Write a seed conversation fixture**

```json
{
  "schema_version": 1,
  "id": "2026-05-01T15-00-00-eval1",
  "page": "basic_chat",
  "started_at": "2026-05-01T15:00:00Z",
  "ended_at": "2026-05-01T15:10:00Z",
  "config": {"provider": "lmstudio", "model": "vllm/local"},
  "messages": [
    {"role": "user", "ts": "2026-05-01T15:00:01Z", "content": [
      {"type": "text", "text": "Keep your replies terse. I'd rather a one-liner than a paragraph."}
    ]},
    {"role": "assistant", "ts": "2026-05-01T15:00:02Z", "content": [
      {"type": "text", "text": "got it."}
    ]},
    {"role": "user", "ts": "2026-05-01T15:00:30Z", "content": [
      {"type": "text", "text": "Also: I'm on Python 3.14 across all projects now."}
    ]},
    {"role": "assistant", "ts": "2026-05-01T15:00:31Z", "content": [
      {"type": "text", "text": "noted."}
    ]}
  ],
  "events": []
}
```

- [ ] **Step 2: questions.yaml**

```yaml
# tests/eval/memory/scenarios/01_user_preferences/questions.yaml
- id: q1
  query: "What output style does the user prefer?"
  reference: |
    The user prefers terse, brief replies — one-liners over paragraphs.

- id: q2
  query: "Which Python version is the user using?"
  reference: |
    Python 3.14, across all projects.
```

- [ ] **Step 3: expected.yaml** (post-dream state expectations, used as a soft check)

```yaml
# tests/eval/memory/scenarios/01_user_preferences/expected.yaml
facts:
  - subject_canonical_name_contains: "user"
    predicate_in: ["prefers", "uses"]
  - subject_canonical_name_contains: "Travis"
    predicate_in: ["uses"]
reflections_min: 0     # may or may not synthesize
episodes_min: 2
```

- [ ] **Step 4: conftest stub (no shared fixtures yet beyond pytest defaults)**

```python
# tests/eval/memory/conftest.py
"""Eval harness fixtures live here as needed."""
```

- [ ] **Step 5: Commit**

```bash
git add tests/eval/memory/
git commit -m "test(eval): first eval scenario — user preferences + Python version"
```

---

### Task 17.2: Eval runner that replays a scenario end-to-end

**Files:**
- Create: `tests/eval/memory/runner.py`

- [ ] **Step 1: Implement the runner**

```python
# tests/eval/memory/runner.py
"""Replay a scenario, run a full dream cycle, query held-out questions.

Usage:
    .agent-playground/bin/python -m tests.eval.memory.runner \
        tests/eval/memory/scenarios/01_user_preferences
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.extractor.pump import pump_once
from mcp_servers.memory.repo.raw_turns import record_turn
from mcp_servers.memory.retrieval.recall import recall


def _seed_raw_turns(conn, conv_path: Path) -> None:
    data = json.loads(conv_path.read_text())
    for i, m in enumerate(data["messages"]):
        record_turn(
            conn,
            conversation_id=data["id"],
            turn_index=i,
            role=m["role"],
            occurred_at=m.get("ts", "2026-05-01T15:00:00Z"),
        )


def _load_llm(provider: str = "lmstudio", model: str | None = None):
    from playground.providers.registry import get_client
    return get_client(provider, model or os.getenv("DREAMER_MODEL", "local"))


def run_scenario(scenario_dir: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="memeval-"))
    conversations_root = tmp / "conversations"
    db_path = tmp / "memory.db"
    try:
        # mirror conversation files into the temp root
        page_dir = conversations_root / "basic_chat"
        page_dir.mkdir(parents=True)
        for conv in (scenario_dir / "conversations").glob("*.json"):
            shutil.copy(conv, page_dir / conv.name)

        conn = open_connection(db_path)
        apply_migrations(conn)
        for conv in (scenario_dir / "conversations").glob("*.json"):
            _seed_raw_turns(conn, conv)

        llm = _load_llm()
        # 1) extract atomic episodes
        pump_once(conn=conn, llm=llm, conversations_root=conversations_root)
        # 2) run a full dream cycle
        run_cycle(
            conn=conn, pid=os.getpid(), cycle_mode="full",
            trigger_reason="manual", model_used="vllm/local",
            stages=all_stages(), ctx={"model": "local"},
        )

        # 3) ask held-out questions via recall
        qs = yaml.safe_load((scenario_dir / "questions.yaml").read_text())
        report = {"questions": []}
        for q in qs:
            memories = recall(conn=conn, query=q["query"], max_results=5)
            report["questions"].append({
                "id": q["id"], "query": q["query"],
                "reference": q["reference"].strip(),
                "memories": memories,
            })
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scenario_dir", type=Path)
    args = p.parse_args(argv)
    report = run_scenario(args.scenario_dir)
    sys.stdout.write(json.dumps(report, indent=2, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manual run + commit**

```bash
.agent-playground/bin/python -m tests.eval.memory.runner \
    tests/eval/memory/scenarios/01_user_preferences
# expect a JSON report on stdout with memories returned per question
git add tests/eval/memory/runner.py
git commit -m "test(eval): end-to-end runner — seed → pump → dream → recall"
```

---

### Task 17.3: LLM-judge grading

**Files:**
- Extend: `tests/eval/memory/runner.py`

- [ ] **Step 1: Add a `grade_with_judge` helper and call it from `run_scenario`**

Append to `tests/eval/memory/runner.py`:

```python
def grade_with_judge(*, llm, query: str, reference: str, memories: list[dict]) -> dict:
    from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
    rendered = "\n".join(
        f"- ({m.get('node_kind')}) {m.get('summary') or m.get('predicate') or m.get('statement')}"
        for m in memories
    ) or "(no memories returned)"
    user = (
        f"You are grading a memory system. The user asked:\n"
        f"  {query}\n"
        f"The correct answer is:\n  {reference.strip()}\n"
        f"The retrieved memories were:\n{rendered}\n\n"
        f"Decide if the retrieved memories *would let the agent answer the "
        f"question correctly*. Return JSON: "
        f"{{\"verdict\": \"pass\"|\"fail\"|\"partial\", \"reason\": \"...\"}}."
    )
    return call_json_llm(
        llm=llm, system="Return only JSON.", user=user, max_tokens=400,
    )
```

Replace the `return report` near the end of `run_scenario` with:

```python
    judge = _load_llm()
    for q in report["questions"]:
        q["grade"] = grade_with_judge(
            llm=judge, query=q["query"],
            reference=q["reference"], memories=q["memories"],
        )
    summary = {"pass": 0, "partial": 0, "fail": 0}
    for q in report["questions"]:
        summary[q["grade"].get("verdict", "fail")] = summary.get(
            q["grade"].get("verdict", "fail"), 0) + 1
    report["summary"] = summary
    return report
```

- [ ] **Step 2: Run + commit**

```bash
.agent-playground/bin/python -m tests.eval.memory.runner \
    tests/eval/memory/scenarios/01_user_preferences
# Expect a summary block at the bottom with pass/partial/fail counts.
git add tests/eval/memory/runner.py
git commit -m "test(eval): LLM-judge grading on retrieved memories"
```

---

## Phase 18 — Wire-up + smoke

### Task 18.1: End-to-end smoke through Basic Chat → dream → recall

**Files:**
- Create: `tests/memory/test_e2e_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/memory/test_e2e_smoke.py
"""End-to-end smoke: seed turns, pump extraction, full dream cycle, recall."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.extractor.pump import pump_once
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage
from mcp_servers.memory.repo.raw_turns import record_turn
from mcp_servers.memory.retrieval.recall import recall


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _seed_conversation(root: Path, conv_id: str) -> None:
    page = root / "basic_chat"; page.mkdir(parents=True, exist_ok=True)
    (page / f"{conv_id}.json").write_text(json.dumps({
        "id": conv_id, "page": "basic_chat",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "I prefer terse replies, no paragraphs."}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "I'm on Python 3.14 now."}
            ]},
        ],
    }))


def test_e2e_smoke(conn: sqlite3.Connection, tmp_path: Path, fixed_embedder) -> None:
    _seed_conversation(tmp_path, "c_smoke")
    record_turn(conn, conversation_id="c_smoke", turn_index=0, role="user",
                occurred_at="2026-05-12T15:00:00Z")
    record_turn(conn, conversation_id="c_smoke", turn_index=1, role="user",
                occurred_at="2026-05-12T15:00:30Z")

    extract_llm = MagicMock()
    extract_llm.stream_chat.side_effect = [
        _stream({"episodes": [{
            "actor": "user", "predicate": "expressed_preference",
            "subject": "Travis", "object": "terse output",
            "summary": "user prefers terse output",
            "importance": 0.8,
        }]}),
        _stream({"episodes": [{
            "actor": "user", "predicate": "uses",
            "subject": "Travis", "object": "Python 3.14",
            "summary": "user uses Python 3.14",
            "importance": 0.8,
        }]}),
    ]
    pump_once(conn=conn, llm=extract_llm, conversations_root=tmp_path)
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM episodes"
    ).fetchone()["c"] == 2

    dream_llm = MagicMock()
    # consolidate: no dups; extract: produce both facts; reflect: skip; recombine: none.
    dream_llm.stream_chat.side_effect = [
        _stream({"groups": []}),
        _stream({"groups": []}),
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "prefers",
            "object_kind": "value", "object": "terse output",
            "confidence": 0.9, "valid_from_hint": "2026-05-12T15:00:00Z",
        }]}),
        _stream({"facts": [{
            "subject": "Travis", "subject_kind": "person",
            "predicate": "uses",
            "object_kind": "entity", "object": "Python 3.14",
            "object_entity_kind": "concept",
            "confidence": 0.95, "valid_from_hint": "2026-05-12T15:00:30Z",
        }]}),
        # reflect calls (clusters with low importance won't actually invoke)
        # recombine — empty pool may not invoke either
    ]

    # provide our fixed embedder via ctx
    run_cycle(
        conn=conn, pid=os.getpid(), cycle_mode="full",
        trigger_reason="manual", model_used="test/fake",
        stages=all_stages(), ctx={"model": "test/fake"},
    )

    # We may not have called every LLM mock — that's fine, smoke just needs
    # facts + at least one current belief.
    rows = conn.execute(
        "SELECT predicate, object_value, object_entity FROM facts "
        "WHERE valid_to IS NULL AND invalidated_at IS NULL"
    ).fetchall()
    preds = {r["predicate"] for r in rows}
    assert "prefers" in preds and "uses" in preds

    out = recall(conn=conn, query="output style preferences",
                 embedder=fixed_embedder, max_results=3)
    assert any("terse" in (m.get("summary") or m.get("object_value") or "") for m in out)
```

- [ ] **Step 2: Run + commit**

```bash
.agent-playground/bin/pytest tests/memory/test_e2e_smoke.py -v
git add tests/memory/test_e2e_smoke.py
git commit -m "test(memory): end-to-end smoke (seed → pump → dream → recall)"
```

---

### Task 18.2: Final integration pass — provider comment + README polish + full suite green

**Files:**
- Modify: `playground/providers/lmstudio_client.py` (comment only)
- Modify: `README.md` (small section addition)

- [ ] **Step 1: Comment the lmstudio provider as OpenAI-compatible**

Open `playground/providers/lmstudio_client.py` and add a single top-of-module note immediately after the docstring (or after the imports if no docstring exists):

```python
# Note: this client talks to any OpenAI-compatible local inference server
# at LMSTUDIO_BASE_URL. The "lmstudio" name is historical — point it at
# LM Studio, vLLM, llama.cpp's OpenAI server, or anything compatible.
```

No code changes.

- [ ] **Step 2: Extend README with a "Memory + Dreaming" section**

Append to `README.md`:

```markdown
## Memory + Dreaming

The bundled `memory` MCP server gives the playground agent persistent
cross-conversation memory. A separate background dreamer process runs a
six-stage consolidation cycle that produces a bi-temporal knowledge
graph plus speculative hypotheses you can curate from the new
**Dreaming** page.

Quick start:

1. Set `LMSTUDIO_BASE_URL` to your local OpenAI-compatible inference
   server (vLLM, LM Studio, etc.).
2. `streamlit run app.py` — the memory server starts automatically.
3. Send a few messages in Basic Chat. Then open **Dreaming** → click
   **Start daemon** and **Dream now (full)**.

Design: `docs/superpowers/specs/2026-05-11-memory-dreaming-mcp-design.md`.
```

- [ ] **Step 3: Full test suite green**

```bash
.agent-playground/bin/pytest -q
.agent-playground/bin/ruff check .
```

Expected: all green; only `slow` tests skipped by default.

- [ ] **Step 4: Final commit**

```bash
git add playground/providers/lmstudio_client.py README.md
git commit -m "docs: memory + dreaming README section; clarify lmstudio is OpenAI-compatible"
```

---

## Spec coverage map (self-review)

| Spec section | Phase / Task |
|---|---|
| §1 Summary | n/a |
| §2 Goals — single persistent agent, hot path cheap, bi-temporal, operator console, local-first, evaluable | Phases 1, 4, 16, 17 |
| §2 Non-goals (multi-user, skills, federation, judge, OpenTel, web UI, shredding) | Explicitly out of plan |
| §3.1 Process topology (Streamlit / memory-mcp / dreamer) | Phases 6, 7, 16 |
| §3.2 Boundaries | Implicit in Phases 6 & 7 (single-writer via lock; agent-only via MCP) |
| §3.3 Daemon lifecycle (Streamlit-supervised + CLI, advisory lock) | Tasks 7.1, 7.2, 16.1 |
| §3.3 Triggers (light / full / maintenance) | Task 16.4 settings + Phase 7 cycle modes |
| §4.1 Schema (12 tables incl. sqlite-vec) | Task 0.5 |
| §4.2 Bi-temporal semantics + supersession | Tasks 4.2, 4.3, 4.4, 10.2, 10.3 |
| §5 Hot path (record_turn, queued extraction, background pack) | Phases 1, 3, 15 |
| §6 Six stages | Phases 8–13 |
| §6.2 Cycle modes | Task 7.2 (`_CYCLE_STAGES`) |
| §6.3 Resilience (idempotent stages, advisory lock reclaim) | Tasks 7.1, 7.2 |
| §7 Recall (vector seed + personalized PageRank) | Phase 14 |
| §7.2 Background pack assembly | Phase 15 |
| §8.1 Tools (record_turn, recall, search_*, get_entity, traverse_graph, list_hypotheses, confirm_hypothesis, correct_fact, forget, force_dream, get_dreamer_status, query_dream_runs) | Phases 6, 7, 14 (and 18.x for any leftovers) |
| §8.2 Prompts (background, review_hypotheses) | Phase 15 (background); `review_hypotheses` left as a v2 extension — flagged below |
| §8.3 Resources (memory://status, etc.) | Task 6.4 (status). Other resource URIs (entity/timeline/dream_runs/<id>) are read via existing tools — flagged below as a v2 extension |
| §9 Dreaming page | Phase 16 |
| §10 Tech stack | Task 0.1 (deps) |
| §11 Testing & eval | Phase 17 |
| §12 Failure modes | Tasks 1.3 (graceful degrade), 3.3 (poison after retries), 7.1 (lock reclaim), 12.x (novelty filter), 13.x (decay caps), 18.x (smoke) |
| §13 Out of scope | Honored throughout |
| §14 Open implementation questions | Left to implementation judgment |

**Acknowledged gaps in this plan (intentional v1 trims):**

- `review_hypotheses` MCP prompt (§8.2) — not built; the operator page handles curation directly.
- Per-resource MCP URIs `memory://hypotheses/open`, `memory://entity/<name>`, `memory://timeline/<id>`, `memory://dream_runs/<id>` (§8.3) — only `memory://status` shipped. Other reads are available via tools (`list_hypotheses`, `get_entity`). Resource URIs are a thin wrapper around the same handlers; deferred without functional loss.
- `correct_fact`, `forget` (§8.1) — not in v1 tool surface; operators can curate via Dreaming page or SQL. Add when a use case appears.
- Trigger daemon loop (§3.3) — the `serve` CLI is a sleeping stub. Auto-firing on idle / queue / schedule is a small follow-up (replace the sleep loop in `dreamer.py::cmd_serve` with a poll over `dreamer_config['triggers']`). Manual triggering from the Dreaming page works in v1.

These are explicit deferrals, not placeholders — the rest of the plan is fully fleshed.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-12-memory-dreaming-mcp.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
