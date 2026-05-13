"""Importance-weighted decay: archive the bottom percentile of non-fresh nodes."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime


def _recency_factor(occurred_at: str) -> float:
    try:
        ts = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    except Exception:
        return 0.5
    age_days = max(0.0, (datetime.now(UTC) - ts).total_seconds() / 86400.0)
    # half-life ~ 30 days
    return math.exp(-age_days / 30.0)


def archive_bottom_percentile(
    *, conn: sqlite3.Connection, percentile: float = 0.05,
) -> int:
    rows = conn.execute(
        "SELECT id, importance, occurred_at FROM episodes WHERE status = 'consolidated'"
    ).fetchall()
    if not rows:
        return 0
    scored = [
        (r["id"], r["importance"] * _recency_factor(r["occurred_at"]))
        for r in rows
    ]
    scored.sort(key=lambda kv: kv[1])
    n_archive = max(1, int(len(scored) * percentile))
    targets = [eid for eid, _ in scored[:n_archive]]
    conn.executemany(
        "UPDATE episodes SET status = 'archived' WHERE id = ?",
        [(t,) for t in targets],
    )
    return n_archive
