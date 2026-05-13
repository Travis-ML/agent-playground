"""Build the Markdown background pack injected at conversation start."""

from __future__ import annotations

import json
import sqlite3

_EMPTY = "_No prior memory is available yet — this is a fresh start._\n"


def build_pack(
    *,
    conn: sqlite3.Connection,
    topic_hint: str | None = None,
    recency_days: int = 7,
) -> str:
    row = conn.execute(
        "SELECT value FROM dreamer_config WHERE key = 'background_pack_cache'"
    ).fetchone()
    if row is None:
        return _EMPTY
    cache = json.loads(row["value"])

    lines: list[str] = []
    lines.append("You have prior memory that may be relevant.")
    lines.append("")

    entities = cache.get("entities", [])
    if entities:
        lines.append("## Top topics you've engaged with")
        for e in entities:
            name = e.get("name") or e.get("canonical_name") or e["id"]
            summary = e.get("summary") or ""
            score = e.get("score")
            tail = f" — {summary}" if summary else ""
            score_s = f" (pagerank={score:.3f})" if isinstance(score, (int, float)) else ""
            lines.append(f"- **{name}**{score_s}{tail}")
        lines.append("")

    refls = cache.get("reflections", [])
    if refls:
        lines.append("## Recent reflections")
        for r in refls:
            lines.append(f"- {r['summary']}")
        lines.append("")

    if not entities and not refls:
        return _EMPTY
    return "\n".join(lines).rstrip() + "\n"
