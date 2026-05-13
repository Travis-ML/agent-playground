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
