"""Stage ① — embed any missing episode embeddings; cluster fresh episodes."""

from __future__ import annotations

import sqlite3
import struct
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from mcp_servers.memory.embeddings.base import EmbeddingProvider
from mcp_servers.memory.repo.episodes import list_by_status
from mcp_servers.memory.retrieval.vector_search import (
    has_embedding,
    upsert_embedding,
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
    for eid, lab in zip(episode_ids, labels, strict=True):
        groups.setdefault(int(lab), []).append(eid)
    return list(groups.values())


def run(
    *,
    conn: sqlite3.Connection,
    dream_run_id: str,
    embedder: EmbeddingProvider | None = None,
    distance_threshold: float = 0.5,
    ctx: dict | None = None,
    **_: Any,
) -> dict[str, Any]:
    eps = list_by_status(conn, "fresh")
    if not eps:
        return {
            "metrics": {"episodes_seen": 0, "clusters": 0},
            "ctx_updates": {
                "cluster_ids": [],
                "episode_index": {},
                "embedder": embedder,
            },
        }

    if embedder is None:
        from mcp_servers.memory.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        embedder = SentenceTransformersProvider()

    missing = [e for e in eps if not has_embedding(conn, "episode", e.id)]
    vecs: list[list[float]] = []
    if missing:
        vecs = embedder.embed_many([e.summary for e in missing])
        for e, v in zip(missing, vecs, strict=True):
            upsert_embedding(conn, node_kind="episode", node_id=e.id, embedding=v)

    # gather embeddings for ALL fresh episodes (including ones we just wrote)
    all_vecs: list[list[float]] = []
    summary_vecs = {e.id: v for e, v in zip(missing, vecs, strict=True)} if missing else {}
    for e in eps:
        if e.id in summary_vecs:
            all_vecs.append(summary_vecs[e.id])
        else:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE node_kind='episode' AND node_id = ?",
                (e.id,),
            ).fetchone()
            all_vecs.append(list(struct.unpack(f"{embedder.dim}f", row["embedding"])))

    clusters = cluster_episodes(
        episode_ids=[e.id for e in eps],
        embeddings=all_vecs,
        distance_threshold=distance_threshold,
    )
    return {
        "metrics": {
            "episodes_seen": len(eps),
            "clusters": len(clusters),
        },
        "ctx_updates": {
            "cluster_ids": clusters,
            "episode_index": {e.id: e for e in eps},
            "embedder": embedder,
        },
    }
