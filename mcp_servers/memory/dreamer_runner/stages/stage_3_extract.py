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
