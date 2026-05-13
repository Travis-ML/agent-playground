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
from mcp_servers.memory.retrieval.vector_search import upsert_embedding


def _stream(payload):
    yield TextDelta(text=json.dumps(payload))
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


def _seed_conversation(root: Path, conv_id: str) -> None:
    page = root / "basic_chat"
    page.mkdir(parents=True, exist_ok=True)
    (page / f"{conv_id}.json").write_text(json.dumps({
        "id": conv_id,
        "page": "basic_chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I prefer terse replies, no paragraphs."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I'm on Python 3.14 now."}
                ],
            },
        ],
    }))


def test_e2e_smoke(conn: sqlite3.Connection, tmp_path: Path, fixed_embedder) -> None:
    _seed_conversation(tmp_path, "c_smoke")
    record_turn(
        conn,
        conversation_id="c_smoke",
        turn_index=0,
        role="user",
        occurred_at="2026-05-12T15:00:00Z",
    )
    record_turn(
        conn,
        conversation_id="c_smoke",
        turn_index=1,
        role="user",
        occurred_at="2026-05-12T15:00:30Z",
    )

    # Extraction: two raw turns → two episodes
    extract_llm = MagicMock()
    extract_llm.stream_chat.side_effect = [
        _stream(
            {
                "episodes": [
                    {
                        "actor": "user",
                        "predicate": "expressed_preference",
                        "subject": "Travis",
                        "object": "terse output",
                        "summary": "user prefers terse output",
                        "importance": 0.8,
                    }
                ]
            }
        ),
        _stream(
            {
                "episodes": [
                    {
                        "actor": "user",
                        "predicate": "uses",
                        "subject": "Travis",
                        "object": "Python 3.14",
                        "summary": "user uses Python 3.14",
                        "importance": 0.8,
                    }
                ]
            }
        ),
    ]
    pump_once(conn=conn, llm=extract_llm, conversations_root=tmp_path)
    assert (
        conn.execute("SELECT COUNT(*) AS c FROM episodes").fetchone()["c"]
        == 2
    )

    # Dream cycle: consolidate → extract → reflect → recombine → decay_reindex
    dream_llm = MagicMock()
    # Consolidate: skipped for single-episode clusters
    # Extract: one call per cluster (2 clusters → 2 calls)
    # Reflect/Recombine: may not invoke
    dream_llm.stream_chat.side_effect = [
        # extract (stage 3): one fact per episode/cluster
        _stream(
            {
                "facts": [
                    {
                        "subject": "Travis",
                        "subject_kind": "person",
                        "predicate": "prefers",
                        "object_kind": "value",
                        "object": "terse output",
                        "confidence": 0.9,
                        "valid_from_hint": "2026-05-12T15:00:00Z",
                    }
                ]
            }
        ),
        _stream(
            {
                "facts": [
                    {
                        "subject": "Travis",
                        "subject_kind": "person",
                        "predicate": "uses",
                        "object_kind": "entity",
                        "object": "Python 3.14",
                        "object_entity_kind": "concept",
                        "confidence": 0.95,
                        "valid_from_hint": "2026-05-12T15:00:30Z",
                    }
                ]
            }
        ),
        # reflect (stage 4): may not invoke if clusters are low-importance
        # recombine (stage 5): may not invoke if triplet pool too small
        # Add empty responses as fallback
        _stream({"groups": []}),
        _stream({"facts": []}),
        _stream({"hypotheses": []}),
    ]

    # Wrap stages to inject embedder and dream_llm
    real_stages = all_stages()

    def wrap_stage(stage_fn, llm, embedder):
        """Inject llm and embedder into stage calls."""
        def wrapped(conn, dream_run_id, ctx, **kwargs):
            return stage_fn(
                conn=conn,
                dream_run_id=dream_run_id,
                ctx=ctx,
                llm=llm,
                embedder=embedder,
                **kwargs
            )
        return wrapped

    wrapped_stages = {
        name: wrap_stage(fn, dream_llm, fixed_embedder)
        for name, fn in real_stages.items()
    }

    run_cycle(
        conn=conn,
        pid=os.getpid(),
        cycle_mode="full",
        trigger_reason="manual",
        model_used="test/fake",
        stages=wrapped_stages,
        ctx={"model": "test/fake"},
    )

    # Verify facts: expect prefers + uses predicates
    rows = conn.execute(
        "SELECT predicate FROM facts WHERE valid_to IS NULL AND invalidated_at IS NULL"
    ).fetchall()
    preds = {r["predicate"] for r in rows}
    assert "prefers" in preds and "uses" in preds

    # Embed facts for recall (facts aren't auto-embedded by stages yet)
    facts = conn.execute(
        "SELECT id, object_value, predicate FROM facts WHERE valid_to IS NULL AND invalidated_at IS NULL"
    ).fetchall()
    for fact in facts:
        # Embed based on predicate or object_value
        embed_text = fact["object_value"] or fact["predicate"]
        vec = fixed_embedder.embed(embed_text)
        upsert_embedding(conn, node_kind="fact", node_id=fact["id"], embedding=vec)

    # Recall: search for "output style preferences" should find terse
    out = recall(conn=conn, query="output style preferences", embedder=fixed_embedder, max_results=3)
    assert any(
        "terse"
        in (m.get("summary") or m.get("object_value") or "")
        for m in out
    )
