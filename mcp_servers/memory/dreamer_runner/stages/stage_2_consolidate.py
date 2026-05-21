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
        llm = get_client("local", model=ctx.get("model", "local"))

    tpl = _PROMPT_PATH.read_text()
    total_groups = 0
    duplicates = 0
    for cluster in clusters:
        if len(cluster) < 2:
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
