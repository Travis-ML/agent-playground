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
