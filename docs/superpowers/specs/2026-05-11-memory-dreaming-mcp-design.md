# Memory + Dreaming MCP Server — Design

**Status:** Draft for review
**Date:** 2026-05-11
**Author:** Travis Lelle
**Scope:** v1 of a bundled MCP server (`mcp_servers/memory/`) that gives the TravisML Agent Playground a single persistent agent identity with long-term memory and offline "dreaming" consolidation.

---

## 1. Summary

A bundled MCP server plus a long-running background "dreamer" process that together provide cross-conversation memory for the playground agent. Hot path stays cheap: each turn is appended to storage and queued for atomic-episode extraction. Cold path runs a six-stage dream cycle (consolidate → extract → reflect → recombine → decay → reindex) that turns episodic memory into a bi-temporal knowledge graph with first-class hypotheses. Retrieval combines vector search with personalized PageRank over a typed link graph (HippoRAG-style), so multi-hop connections surface.

The distinctive bets:

1. **Bi-temporal facts.** Tracks *when a fact was true in the world* separately from *when learned*. Changes never overwrite — they supersede with explicit lineage. No data loss when beliefs evolve.
2. **Hypotheses as first-class citizens.** A REM-like recombine stage samples distant memory nodes and asks an LLM for surprising connections. Results are stored as hypotheses (open → corroborated | refuted | set aside) without polluting fact recall.
3. **Recursive reflections.** Generative-Agents-style synthesis of higher-level insights, applied recursively (reflections of reflections).
4. **Vector + PageRank-spread retrieval.** Multi-hop activation over a typed link graph, not just vector top-K.

No public system in early 2026 combines all four.

---

## 2. Goals & non-goals

### Goals

- A single persistent agent identity that accumulates memory across every conversation in the playground.
- Hot path adds zero blocking LLM work; turns return at SQLite-insert latency.
- Offline dreaming does the expensive consolidation, extraction, reflection, and creative recombination.
- Bi-temporal correctness: contradictions never destroy history.
- An operator console (`pages/2_Dreaming.py`) where the user watches and curates the agent's mind.
- Local-first, zero-ops storage. One SQLite file. No external services required.
- Evaluable: a seeded benchmark in `tests/eval/memory/` lets us measure recall accuracy across versions.

### Non-goals (v1)

- Multi-user / multi-agent shared memory (single user, single agent identity).
- Procedural / skill memory (we did not add a skills table; this is v2).
- Federation across multiple memory stores.
- LLM-as-judge for hypothesis corroboration (operator decides; auto-judging is v2).
- OpenTelemetry traces (structured logs to stderr + `dream_runs` table are sufficient).
- Independent web UI (Dreaming Streamlit page is the only operator surface).
- Privacy-grade deletion via cryptographic shredding (v2).

---

## 3. Architecture

### 3.1 Process topology

Three processes sharing one SQLite database:

| Process | Lifetime | Role |
|---|---|---|
| **Streamlit playground** | Foreground, started by the user (`streamlit run app.py`) | Hosts `pages/1_Basic_Chat.py` and the new `pages/2_Dreaming.py`. Spawns and supervises the dreamer by default. |
| **memory-mcp** (stdio) | Spawned per Streamlit page session via `mcp.json` | The agent's only contact point with the memory subsystem. Serves tools / prompts / resources. Writes hot-path data; reads via retrieval algorithms. |
| **memory-dreamer** (long-running daemon) | Started by the Streamlit Dreaming page (default) OR via CLI `python -m mcp_servers.memory.dreamer serve` | Runs the six-stage dream cycle. Auto-triggers on idle / queue depth. Single-writer for facts/reflections/hypotheses/links. |

Shared storage: **SQLite** at `~/.travisml-playground/memory.db` in WAL mode with the `sqlite-vec` extension loaded. The same connection-pool patterns we already use for other on-disk state apply.

### 3.2 Boundaries

- **The agent only ever talks to `memory-mcp`.** It does not know the dreamer exists; it sees only the result of dream cycles in its retrieval output.
- **memory-mcp is read-mostly + cheap writes.** It appends raw turn refs and enqueues extraction. It does not run any dream-cycle stages.
- **memory-dreamer is the sole writer for the consolidated layer** (facts, reflections, hypotheses, links, embeddings, pagerank_scores). It coordinates via a SQLite advisory lock; `memory-mcp` reads do not block.
- **Raw conversations remain in `conversations/*.json`.** The memory store references them via `(conversation_id, turn_index)` pairs and never duplicates content.

### 3.3 Daemon lifecycle

The dreamer is a long-running background process. Default mode: the Streamlit Dreaming page spawns it as a `subprocess.Popen` child on first load and supervises it (restart on crash, stop on Streamlit exit). Alternative mode: the user runs `python -m mcp_servers.memory.dreamer serve` standalone (e.g., as a launchd / systemd unit). Both modes are coordinated by the SQLite advisory lock — at most one dreamer holds the write lock at a time.

Triggers (any of them wake a sleeping dreamer):

- **Light cycle** — stages ①②③⑥ (no reflect, no recombine). Fires when extraction queue is drained AND ≥ N fresh consolidated episodes pending (default N=20), OR every M minutes of activity (default M=15).
- **Full cycle** — all six stages. Fires after > X minutes of idle time (default X=30), OR scheduled (default: nightly at 03:30), OR manually from the Dreaming page.
- **Maintenance cycle** — stage ⑥ only. Once weekly. Just decay + reindex; cheap.

Settings are exposed via the Dreaming page settings panel and stored in a `dreamer_config` SQLite table (no env vars).

---

## 4. Data model

All tables live in the single `memory.db` file. Schema is versioned in a `schema_version` table; migrations live in `mcp_servers/memory/migrations/`.

### 4.1 Tables

```sql
-- 1. Pointer rows into existing conversations/*.json — no content duplication.
CREATE TABLE raw_turn_refs (
  id              TEXT PRIMARY KEY,            -- rt_<ulid>
  conversation_id TEXT NOT NULL,               -- conversation filename stem
  turn_index      INTEGER NOT NULL,
  role            TEXT NOT NULL,               -- 'user' | 'assistant' | 'tool'
  occurred_at     TIMESTAMP NOT NULL,
  recorded_at     TIMESTAMP NOT NULL,
  extraction_status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'done' | 'failed' | 'poison'
  UNIQUE (conversation_id, turn_index)
);

-- 2. Atomic episodic events extracted from raw turns.
CREATE TABLE episodes (
  id              TEXT PRIMARY KEY,            -- ep_<ulid>
  actor           TEXT NOT NULL,               -- 'user' | 'agent' | 'tool:<name>'
  predicate       TEXT NOT NULL,               -- normalized verb
  subject_entity  TEXT,                        -- entity id (or NULL)
  object_entity   TEXT,                        -- entity id (or NULL)
  object_value    TEXT,                        -- scalar literal if not entity
  summary         TEXT NOT NULL,               -- one-sentence description
  importance      REAL NOT NULL DEFAULT 0.5,   -- 0..1, set by extractor, refined by dreamer
  occurred_at     TIMESTAMP NOT NULL,          -- from raw turn
  created_at      TIMESTAMP NOT NULL,
  status          TEXT NOT NULL DEFAULT 'fresh',  -- 'fresh' | 'consolidated' | 'archived'
  source_refs     JSON NOT NULL                -- [{ "raw_turn_ref_id": "rt_...", ... }]
);

CREATE INDEX idx_episodes_status ON episodes(status);
CREATE INDEX idx_episodes_occurred_at ON episodes(occurred_at);

-- 3. Canonicalized entities (people, projects, concepts, files, tools, …).
CREATE TABLE entities (
  id             TEXT PRIMARY KEY,             -- en_<ulid>
  canonical_name TEXT NOT NULL UNIQUE,
  kind           TEXT NOT NULL,                -- 'person'|'project'|'concept'|'tool'|'file'|'place'|'other'
  aliases        JSON NOT NULL DEFAULT '[]',
  summary        TEXT,
  first_seen     TIMESTAMP NOT NULL,
  last_seen      TIMESTAMP NOT NULL,
  importance     REAL NOT NULL DEFAULT 0.5
);

-- 4. Bi-temporal facts.
CREATE TABLE facts (
  id              TEXT PRIMARY KEY,            -- fa_<ulid>
  subject_entity  TEXT NOT NULL,               -- references entities(id)
  predicate       TEXT NOT NULL,
  object_entity   TEXT,                        -- entity id, or NULL if scalar
  object_value    TEXT,                        -- scalar literal, or NULL if entity

  -- Bi-temporal time:
  valid_from      TIMESTAMP NOT NULL,          -- when this began being true in the world
  valid_to        TIMESTAMP,                   -- NULL = still true
  learned_at      TIMESTAMP NOT NULL,          -- when the agent first learned this
  invalidated_at  TIMESTAMP,                   -- NULL = still believed; set when superseded

  -- Provenance and lineage:
  source_episode_ids JSON NOT NULL,            -- ["ep_x", "ep_y"]
  confidence      REAL NOT NULL DEFAULT 0.7,
  supersedes      TEXT,                        -- fact id this replaced (NULL if first)
  superseded_by   TEXT,                        -- fact id that replaced this (NULL if current)

  -- Auditing:
  created_in_dream_run TEXT NOT NULL,          -- dr_<ulid>
  CHECK (object_entity IS NOT NULL OR object_value IS NOT NULL)
);

CREATE INDEX idx_facts_subject_predicate ON facts(subject_entity, predicate);
CREATE INDEX idx_facts_valid_to ON facts(valid_to);
CREATE INDEX idx_facts_learned_at ON facts(learned_at);

-- 5. Recursive reflections — higher-level insights synthesized from clusters.
CREATE TABLE reflections (
  id           TEXT PRIMARY KEY,               -- re_<ulid>
  summary      TEXT NOT NULL,
  importance   REAL NOT NULL,
  level        INTEGER NOT NULL,               -- 1 = direct cluster summary; 2+ = reflections of reflections
  source_kind  TEXT NOT NULL,                  -- 'episode_cluster' | 'reflection_cluster'
  source_ids   JSON NOT NULL,
  created_at   TIMESTAMP NOT NULL,
  created_in_dream_run TEXT NOT NULL
);

-- 6. Hypotheses — speculative connections from the recombine stage.
CREATE TABLE hypotheses (
  id              TEXT PRIMARY KEY,            -- hy_<ulid>
  statement       TEXT NOT NULL,
  source_node_ids JSON NOT NULL,               -- the distant memories that were paired
  confidence      REAL NOT NULL,
  status          TEXT NOT NULL DEFAULT 'open',-- 'open'|'corroborated'|'refuted'|'set_aside'
  resolved_at     TIMESTAMP,
  resolved_by     TEXT,                        -- 'operator'|'agent'|'auto-decay'
  resolution_note TEXT,
  created_at      TIMESTAMP NOT NULL,
  created_in_dream_run TEXT NOT NULL
);

CREATE INDEX idx_hypotheses_status ON hypotheses(status);

-- 7. Typed weighted links — the glue layer for graph traversal.
--    src and dst can be any of: episode, fact, entity, reflection, hypothesis.
CREATE TABLE links (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  src_kind    TEXT NOT NULL,
  src_id      TEXT NOT NULL,
  dst_kind    TEXT NOT NULL,
  dst_id      TEXT NOT NULL,
  link_type   TEXT NOT NULL,                   -- 'about'|'caused'|'contradicts'|'reflects'|'recombines'|'see_also'|'extracted_from'|'supersedes'
  weight      REAL NOT NULL DEFAULT 1.0,
  created_in_dream_run TEXT,
  UNIQUE (src_kind, src_id, dst_kind, dst_id, link_type)
);

CREATE INDEX idx_links_src ON links(src_kind, src_id);
CREATE INDEX idx_links_dst ON links(dst_kind, dst_id);

-- 8. Vector index — sqlite-vec virtual table.
CREATE VIRTUAL TABLE embeddings USING vec0(
  node_kind TEXT,
  node_id   TEXT,
  embedding FLOAT[768]                         -- nomic-embed-text-v1.5 dimension
);

-- 9. Cached PageRank scores (recomputed each full dream run).
CREATE TABLE pagerank_scores (
  node_kind TEXT NOT NULL,
  node_id   TEXT NOT NULL,
  score     REAL NOT NULL,
  computed_in_dream_run TEXT NOT NULL,
  PRIMARY KEY (node_kind, node_id)
);

-- 10. Audit log of dream cycles.
CREATE TABLE dream_runs (
  id             TEXT PRIMARY KEY,             -- dr_<ulid>
  started_at     TIMESTAMP NOT NULL,
  ended_at       TIMESTAMP,
  cycle_mode     TEXT NOT NULL,                -- 'light'|'full'|'maintenance'
  trigger_reason TEXT NOT NULL,                -- 'idle_timeout'|'queue_depth'|'scheduled'|'manual'
  stages         JSON NOT NULL,                -- per-stage metrics & timing
  model_used     TEXT NOT NULL,
  status         TEXT NOT NULL,                -- 'running'|'completed'|'failed'|'aborted'
  error          TEXT
);

-- 11. Daemon configuration (persisted; editable from Dreaming page settings).
CREATE TABLE dreamer_config (
  key   TEXT PRIMARY KEY,
  value JSON NOT NULL
);

-- 12. Schema version marker.
CREATE TABLE schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TIMESTAMP NOT NULL
);
```

### 4.2 Bi-temporal semantics

Facts have two independent time dimensions:

- **Validity time** (`valid_from`, `valid_to`) — when the fact was true *in the world*.
- **Transaction time** (`learned_at`, `invalidated_at`) — when the agent knew about it.

Default fact lookup is `as_of_now AND currently_believed`:

```sql
SELECT * FROM facts
WHERE (valid_to IS NULL OR valid_to > NOW())
  AND invalidated_at IS NULL;
```

Time-travel queries (`search_facts(as_of='2026-04-01')`) walk through both dimensions:

```sql
SELECT * FROM facts
WHERE valid_from <= '2026-04-01'
  AND (valid_to IS NULL OR valid_to > '2026-04-01')
  AND learned_at <= '2026-04-01'
  AND (invalidated_at IS NULL OR invalidated_at > '2026-04-01');
```

**Supersession** is how contradictions are recorded without data loss:

1. Dreamer encounters a new fact `f_new` whose `(subject, predicate)` matches an existing current `f_old`.
2. The values differ.
3. Write `f_old.valid_to = NOW()`, `f_old.superseded_by = f_new.id`.
4. Write `f_new` with `supersedes = f_old.id`, `valid_from = NOW()`, `learned_at = NOW()`.
5. Add a `supersedes`-typed link in `links`.

**Invariant** (asserted by tests): at most one fact with a given `(subject_entity, predicate)` has `valid_to IS NULL AND invalidated_at IS NULL` at any moment.

---

## 5. Hot path

Per turn in `pages/1_Basic_Chat.py`:

1. The playground emits a normal user/assistant message; existing `conversations/*.json` persistence runs unchanged.
2. The playground calls `memory-mcp.record_turn(conversation_id, turn_index, role, content, occurred_at)`.
3. `memory-mcp` does a single `INSERT INTO raw_turn_refs ... extraction_status='pending'`. Returns immediately.
4. The extractor worker (a thread inside `memory-mcp`, low priority) polls `raw_turn_refs WHERE extraction_status='pending'`:
   - Loads the raw content from `conversations/<id>.json` at the indexed turn.
   - Calls the extractor LLM (resolved via the playground's local-server provider; default: a vLLM endpoint serving the user's chosen local model) with a structured-output prompt for `[{actor, predicate, subject, object, summary, importance, occurred_at}, ...]`.
   - Inserts 0..N rows into `episodes` (status='fresh').
   - Generates embeddings for new episodes via the local embedder; inserts into `embeddings`.
   - Sets `raw_turn_refs.extraction_status='done'` (or `'failed'`/`'poison'` after N retries).

**Latency budget on the synchronous part:** one SQLite `INSERT`. No LLM calls block the turn return.

**Background pack assembly** is also hot-path, but it runs once per conversation start, not per turn. `memory-mcp.get_prompt("background", {topic_hint?, recency_days?})` returns a Markdown block built from:

1. Top-N entities by cached `pagerank_scores`, optionally filtered by `topic_hint` (vector-similar entities).
2. Last K reflections of `level >= 1`.
3. Recently changed facts (`learned_at` within `recency_days` OR `valid_to` recent).

Total time: a few SQLite reads; sub-100ms target.

---

## 6. Dream cycle

The dreamer awakens (by trigger), acquires the SQLite advisory write lock, opens a new `dream_runs` row with `status='running'`, and runs the configured stages in order. Each stage is **idempotent** (safe to retry if the dreamer crashes between stages). Stage failures do not roll back prior stages.

### 6.1 The six stages

#### ① Ingest + cluster (NREM-like)

- Pull `episodes WHERE status='fresh'`.
- For any without embeddings, embed via the local embedder.
- Compute pairwise cosine similarity within a configurable recency window (default 14 days OR last 200 episodes, whichever is broader).
- Agglomerative clustering (sklearn) with a distance threshold; produces `episode_clusters` (in-memory for this run, not persisted as a table — clustering is ephemeral).
- Compute cluster importance: `max(member.importance) + size_bonus`.

#### ② Consolidate

- For each cluster, the dreamer LLM is asked to identify duplicates / near-duplicates and produce a deduped list.
- The merged "survivor" episode keeps its id; duplicates are linked to it via a `consolidated_into` link (and their `status` becomes `'consolidated'`).
- All surviving episodes in the run have `status` updated to `'consolidated'` as well, so they won't be reprocessed by future ingest stages.

#### ③ Extract (semantic crystallization)

- For each cluster, the dreamer LLM produces structured fact candidates: `[{subject, predicate, object_or_value, valid_from_hint, confidence}]`.
- For each candidate, look up existing facts by `(subject, predicate)` that are currently valid.
- **No match → insert** a new `facts` row with `valid_from` = the cluster's median `occurred_at`, `learned_at` = `NOW()`.
- **Match with same value → reinforce**: bump existing fact's `confidence` (capped at 1.0); add provenance episode ids.
- **Match with different value → supersede**: close old fact (`valid_to`, `superseded_by`), write new fact (`supersedes`, fresh times), add `supersedes` link.
- Add `extracted_from` links from each fact to its source episodes.

#### ④ Reflect (recursive abstraction)

- Clusters where `importance >= reflect_threshold` (default 0.7) get a level-1 reflection.
- The dreamer LLM is asked: "what higher-level insight do these episodes suggest? Be specific and only assert if it's clearly supported."
- The result is written to `reflections` with `level=1`, `source_kind='episode_cluster'`.
- Every `reflect_recursion_interval` dream runs (default 5), the dreamer clusters existing `reflections` and generates level-(N+1).
- Add `reflects` links from each reflection to its source nodes.

#### ⑤ Recombine (REM-like — the novel stage)

This stage is what makes "dreaming" non-trivial.

- Sample K node triplets from the full node set (default K=8 per full cycle). Sampling biased toward:
  - **High graph distance** (3+ hops via PageRank-spread reachability).
  - **At least one pair with unusual embedding closeness** (closer than their graph distance would predict).
- For each triplet, the dreamer LLM is prompted:
  > Given three memories: A, B, C. Is there a non-obvious, plausibly-true connection between them that the agent should investigate? Answer with one of:
  > - `none` if no connection is plausible
  > - A single concise statement (≤ 30 words) describing the connection
- Filter responses for novelty (cosine similarity vs existing facts/hypotheses below threshold).
- Write surviving statements as `hypotheses` with `status='open'`, `confidence` from the LLM.
- Add `recombines` links from each hypothesis to its source nodes.

These hypotheses are surfaced in the Dreaming page for operator review (corroborate / refute / set aside). They are NOT included in default `recall` results — they only appear via `list_hypotheses()` or if the agent explicitly queries them.

#### ⑥ Decay + reindex

- Compute `forgetting_score = importance * recency_factor * access_count_factor` for every node.
- Nodes in the bottom percentile (default 5%) are moved to `status='archived'`. Archived nodes still exist in the DB — they just don't surface in retrieval. (Forgetting is reversible.)
- Recompute personalized PageRank over the link graph using `networkx.pagerank`. Persist scores to `pagerank_scores`.
- Refresh embeddings for any node whose summary text changed during this run.
- Rebuild a cached "background pack" snapshot (the top-N entities + recent reflections) as a JSON blob in `dreamer_config` for fast hot-path assembly.

The run ends: `dream_runs.ended_at = NOW()`, `status='completed'`, stage metrics serialized. Lock released.

### 6.2 Cycle modes

| Mode | Stages | Trigger | Typical wall-time |
|---|---|---|---|
| Light | ① ② ③ ⑥ | Queue drained + N fresh episodes pending, or every M minutes active | seconds to ~10s |
| Full | All six | Idle > X minutes, scheduled (nightly), or manual | ~30s–several minutes |
| Maintenance | ⑥ only | Weekly | ~5s |

Settings persisted in `dreamer_config`.

### 6.3 Resilience

- The dreamer wraps each stage in a try/except. A stage failure is logged to `dream_runs.stages[stage_name].error` and the next stage proceeds. If too many stages fail, the run is marked `'failed'` and the dreamer sleeps with an exponential backoff.
- The advisory write lock is held for the entire run; if the dreamer crashes, the next dreamer startup detects an orphaned lock (PID check) and reclaims it.
- All schema mutations are in transactions, scoped per stage. A SIGKILL mid-stage leaves the DB consistent.

---

## 7. Retrieval

### 7.1 Algorithm — vector seed + personalized PageRank

```
recall(query: str, max_results=8, kinds=None) -> Memory[]:

  q_vec = embed(query)

  # 1. Seed set: vector top-K over the candidate nodes
  seeds = vec_search(q_vec, top_k=20, kinds=kinds)
  # seeds is List[(node_kind, node_id, similarity_score)]

  # 2. Build a personalized restart distribution from seed weights
  restart = {(k, id): max(0.0, sim) for (k, id, sim) in seeds}
  restart = normalize(restart)

  # 3. Run personalized PageRank over the link graph (cached in pagerank_scores
  #    we precompute a default; for query-time personalization we run a
  #    fresh PR using NetworkX with personalization=restart)
  scores = personalized_pagerank(
      graph=load_link_graph(),
      personalization=restart,
      damping=0.85,
      max_iter=20,
  )

  # 4. Optionally re-rank top candidates with a cross-encoder (off by default;
  #    enabled via dreamer_config['rerank.enabled']=true)
  top = sorted(scores, key=lambda kv: -kv[1])[:max_results]
  return hydrate(top)
```

**Why vector + PageRank beats vector alone:** Vector top-K can only return memories whose text is similar to the query. PageRank-spread also catches memories that are connected via the graph — e.g., "the user discussed X; X is `caused_by` Y; Y has related episodes E1, E2." Vector misses Y entirely if its text doesn't mention X; PageRank surfaces it through the link.

**Cost.** Personalized PageRank over a graph of ~10k nodes runs in tens of ms with NetworkX. For larger graphs, we cache a default (non-personalized) score and approximate personalization via random-walk approximation; this is a v1.x optimization.

### 7.2 Background pack assembly

Called at conversation start (and only there). Builds a Markdown block:

```
You have prior memory you might find relevant.

## Top topics you've engaged with
- **MCP pool** (recent): you debugged eventloop deaths in the streamlit pool 4 days ago — resolved.
- **Anthropic-native shape**: canonical message shape for cross-provider chat; do not deviate.
- **Terse-output preference**: the user has repeatedly preferred short replies; default to brevity.

## Recent reflections
- You tend to spend more debugging time on async/threading bugs than on logic bugs (4 episodes, last 30 days).
- Streamlit session-state is the most frequent source of "intermittent" bug reports (3 episodes, last 14 days).

## Recently changed beliefs
- The playground uses Python 3.14 (since 2026-05-08; previously 3.13).
```

The format is intentionally Markdown so it merges cleanly into the existing system prompt.

---

## 8. MCP surface

### 8.1 Tools

| Tool | Caller | Purpose |
|---|---|---|
| `record_turn(conversation_id, turn_index, role, content, occurred_at)` | Playground (not the agent) | Hot path write |
| `recall(query, max_results=8, kinds=None)` | Agent | Main retrieval |
| `search_episodes(query?, since?, until?, actor?, limit=20)` | Agent | Structured episode search |
| `search_facts(subject?, predicate?, as_of?, include_invalidated=False)` | Agent | Time-travel fact lookup |
| `get_entity(name)` | Agent | Entity dossier (facts + recent episodes + links) |
| `traverse_graph(start_id, max_hops=2, link_types=None)` | Agent | Graph walk from a known node |
| `list_hypotheses(status='open', limit=10)` | Agent / operator | Surface dream speculations |
| `confirm_hypothesis(id, status: 'corroborated'|'refuted'|'set_aside')` | Agent / operator | Curate hypotheses |
| `correct_fact(id, new_object_entity=None, new_object_value=None, reason=None)` | Agent / operator | Marks a fact wrong; writes a superseding fact with the corrected value |
| `forget(node_kind, id, reason)` | Operator | Explicit deletion (still bi-temporal — archives, not erases) |
| `force_dream(cycle: 'light'|'full'|'maintenance')` | Operator | Manual trigger |
| `get_dreamer_status()` | Operator | Daemon state |
| `query_dream_runs(since?, limit=20)` | Operator | Audit history |

Admin tools (`force_dream`, `get_dreamer_status`, `query_dream_runs`, `forget`) are exposed as MCP tools so the agent can introspect its own dream state and surface it conversationally if asked. They are not magical — the operator still gates major actions via the Dreaming page.

### 8.2 Prompts

| Prompt | Args | Purpose |
|---|---|---|
| `background` | `topic_hint?: str`, `recency_days?: int=7` | Auto-injected at conversation start; returns the Markdown background pack. |
| `review_hypotheses` | (none) | Returns a user-role message asking the agent to walk through open hypotheses and call `confirm_hypothesis` where appropriate. Lets the agent help curate. |

### 8.3 Resources (read-only URIs)

| URI | Returns |
|---|---|
| `memory://status` | Daemon state, queue depth, last dream run summary |
| `memory://hypotheses/open` | JSON list of open hypotheses |
| `memory://entity/<name>` | Full entity dossier (facts + episodes + links) |
| `memory://timeline/<entity_id>` | Chronological fact stream for that entity (great for "how has X evolved?") |
| `memory://dream_runs/<id>` | Full report of one dream cycle (per-stage metrics, what changed) |

---

## 9. Operator UX — `pages/2_Dreaming.py`

A new Streamlit page that follows the existing brand palette (`playground.branding.inject_brand_css` etc.). Sections, top to bottom:

1. **Daemon panel.** Live indicator (running / sleeping / dreaming), queue depth, memory totals (episodes, facts, reflections, hypotheses), last cycle summary. Controls: `Stop daemon` / `Start daemon` toggle, and a `Dream now` button with a cycle selector (light / full / maintenance).
2. **Recent dreams.** Table of the last 10 `dream_runs` rows with wall-time, stage bars, counts per stage. Click a row to open the full per-stage report (`memory://dream_runs/<id>`).
3. **Open hypotheses.** List of hypotheses with `status='open'`. Each shows the statement, sources, confidence; buttons to corroborate, refute, or set aside. Sorted by confidence descending.
4. **Entity browser.** Search box (vector-similar entity names). Result list shows PageRank, episode count, fact count, last-seen. Click an entity → full dossier panel.
5. **Settings (collapsible).** Trigger thresholds, models per stage, embedding model, token budgets, reflect threshold, recombine sample size. Persisted in `dreamer_config`.

The page does not depend on any specific conversation context — it's a global operator console.

---

## 10. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| MCP server framework | `FastMCP` from `mcp` SDK | Same pattern as `mcp_servers/notes` |
| Storage | SQLite (WAL mode) + `sqlite-vec` | Single file; embedded extension |
| Embeddings (default) | `sentence-transformers` with `nomic-embed-text-v1.5` (768-dim), in-process | Local, zero per-turn cost, no external service |
| Embeddings (alt) | vLLM `/v1/embeddings` (when an embedding model is loaded), or hosted (Voyage / OpenAI / Cohere) | Pluggable via `EmbeddingProvider` protocol |
| Extractor LLM (default) | **Local via vLLM**, routed through the playground's local-server provider (OpenAI-compatible) | No per-turn API cost; "good enough" reasoning for atomic-episode extraction |
| Extractor LLM (alt) | Anthropic Haiku-class or OpenAI | Configured per `dreamer_config['extractor.provider/model']` |
| Dreamer LLM (default) | **Local via vLLM**, same provider as extractor | Local capacity is sufficient for the consolidate/extract/reflect stages at the playground's scale; recombine benefits from a stronger model when desired |
| Dreamer LLM (alt) | Anthropic Sonnet-class / Haiku-class, or OpenAI, configurable per stage | E.g., `dreamer_config['dreamer.stage_5.provider'] = 'anthropic'` to upgrade just the recombine stage |
| Clustering | sklearn agglomerative | HDBSCAN as a tuning option |
| Graph + PageRank | `networkx.pagerank` | Personalized PR; pure Python |
| Process management | `subprocess.Popen` from Streamlit; CLI alternative `python -m mcp_servers.memory.dreamer serve` | Coordinated via SQLite advisory lock |
| Migrations | Plain SQL files in `mcp_servers/memory/migrations/`, applied by version order | No ORM; raw SQL throughout |

New PyPI dependencies (added to `pyproject.toml`):

- `sqlite-vec`
- `sentence-transformers`
- `scikit-learn`
- `networkx`
- (`hdbscan` — optional, behind extras)

---

## 11. Testing & evaluation

### 11.1 Unit tests

- **Hot path.** `record_turn` is idempotent; duplicate (`conversation_id`, `turn_index`) returns the existing row. Extraction queue invariants. Extractor failure → `extraction_status='failed'` after retries, `'poison'` after `max_retries`.
- **Schema invariants.** Bi-temporal: at most one current fact per `(subject_entity, predicate)`. Supersession chains are acyclic. `valid_to >= valid_from` always.
- **Each dream stage.** Mocked LLM responses via `respx` (already in dev deps). Each stage's inputs/outputs are independently testable.
- **Recombine determinism.** A seeded triplet sampler so recombine tests are reproducible.

### 11.2 Integration tests

- End-to-end dream cycle on a seeded conversation fixture; assert against golden snapshots of the resulting `facts`, `reflections`, `hypotheses`, and `links`.
- Crash-safety: send SIGKILL to a running dreamer mid-stage; on restart, assert DB consistency and that the next run completes.
- Concurrent reader during writer: open a read connection while the dreamer holds the write lock; assert reads succeed and return pre-write state.

### 11.3 Eval suite — `tests/eval/memory/`

A directory of seeded conversation scenarios and held-out questions to measure the system's recall quality over time. Each scenario:

```
tests/eval/memory/scenarios/<name>/
  conversations/      # seeded fixture conversations (replay format)
  questions.yaml      # held-out questions with reference answers
  expected.yaml       # expected facts, reflections, hypotheses after seeding
```

A pytest-driven runner replays each scenario, runs full dream cycles, then asks the held-out questions via `recall` and compares answers using an LLM judge. Reports per-scenario recall@K, fact accuracy, hypothesis precision.

This is how we'll know if the system gets *better* across versions.

---

## 12. Failure modes & safety

| Failure | Mitigation |
|---|---|
| Dreamer crashes mid-cycle | Per-stage transactions; orphaned-lock reclaim on next startup; partial dream_runs row left with `status='failed'` for inspection |
| Extractor LLM returns garbage | Strict structured-output schema; failed extractions retry N times, then `extraction_status='poison'` and skip |
| Recombine generates spam hypotheses | Novelty filter (cosine vs existing); auto-archive (`status='set_aside'`) hypotheses unresolved for > 30 days (configurable) |
| Memory bloat | Stage ⑥ decay + archive prunes the bottom 5% per cycle; archived nodes are excluded from retrieval but kept for forensics |
| Embeddings model change | `embeddings.model_id` recorded per row; on model swap, dreamer reembeds on next full cycle |
| SQLite corruption | WAL mode + checkpointing; `dream_runs` provides a partial reconstruction path; raw `conversations/*.json` is always the source of truth and can rebuild `raw_turn_refs` |
| Dreamer monopolizes CPU | Per-cycle wall-clock budget + LLM token budget in `dreamer_config`; cycle aborts cleanly when exceeded |
| Sensitive content in episodes | A redaction pass at extraction time strips obvious secrets (API keys via regex, env-style strings); deeper redaction is v2 |

---

## 13. Out of scope (explicit)

- Multi-user / multi-agent shared memory.
- Procedural / skill memory (no skills table; deferred).
- Federation / replication across memory stores.
- LLM-as-judge hypothesis corroboration (operator-driven for v1).
- OpenTelemetry tracing.
- Independent web UI outside Streamlit.
- Privacy-grade cryptographic deletion.

---

## 14. Open questions for implementation

These are intentionally deferred to implementation-plan time, not the design:

1. **Cross-encoder rerank** — does the marginal accuracy gain justify the added latency and dependency? Decide during eval-suite tuning.
2. **Recombine prompt design** — exact wording materially affects hypothesis quality. Iterate against the eval suite.
3. **Embedding dim 768 vs 1024** — `nomic-embed-text-v1.5` is 768; some alternatives are 1024. Pick during tech-stack pinning.
4. **Hypothesis surfacing in `recall` results** — currently excluded; could be opt-in via `kinds=['hypothesis']`. Confirm with usage.
5. **Dreamer auto-restart strategy on Streamlit reload** — pick exact policy (kill+restart vs reuse) during implementation.

These are not unknowns affecting feasibility; they're judgment calls best made with running code.

---
