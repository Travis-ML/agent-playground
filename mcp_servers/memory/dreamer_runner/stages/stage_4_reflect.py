"""Stage ④ — synthesize higher-level reflections from high-importance clusters."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.repo.links import add_link
from mcp_servers.memory.repo.reflections import insert_reflection, list_by_level

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

    # call this every Nth dream run; for v1, every run that has >= 4 level-1 reflections
    if len(list_by_level(conn, level=1, limit=4)) >= 4:
        recursive_added = run_recursive_pass(
            conn=conn, dream_run_id=dream_run_id,
            input_level=1, llm=llm,
        )
        added += recursive_added

    return {"metrics": {"reflections_added": added}}


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
    for r, lab in zip(refls, labels, strict=True):
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
