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
