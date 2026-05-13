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
