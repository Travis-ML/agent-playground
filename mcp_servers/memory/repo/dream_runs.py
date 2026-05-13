"""Dream-runs audit log + per-stage metrics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from mcp_servers.memory.ids import new_dream_run_id
from mcp_servers.memory.models import DreamRun


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def start_run(
    conn: sqlite3.Connection, *,
    cycle_mode: str, trigger_reason: str, model_used: str,
) -> DreamRun:
    dr_id = new_dream_run_id()
    started_at = _now()
    conn.execute(
        """
        INSERT INTO dream_runs
            (id, started_at, ended_at, cycle_mode, trigger_reason,
             stages, model_used, status)
        VALUES (?, ?, NULL, ?, ?, '{}', ?, 'running')
        """,
        (dr_id, started_at, cycle_mode, trigger_reason, model_used),
    )
    return DreamRun(
        id=dr_id, started_at=started_at, ended_at=None,
        cycle_mode=cycle_mode, trigger_reason=trigger_reason,
        stages={}, model_used=model_used, status="running",
    )


def record_stage(
    conn: sqlite3.Connection, run_id: str, *, name: str, metrics: dict,
) -> None:
    row = conn.execute(
        "SELECT stages FROM dream_runs WHERE id = ?", (run_id,)
    ).fetchone()
    stages = json.loads(row["stages"] or "{}")
    stages[name] = metrics
    conn.execute(
        "UPDATE dream_runs SET stages = ? WHERE id = ?",
        (json.dumps(stages), run_id),
    )


def finish_run(
    conn: sqlite3.Connection, run_id: str, *,
    status: str, error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE dream_runs SET ended_at = ?, status = ?, error = ? WHERE id = ?",
        (_now(), status, error, run_id),
    )


def list_recent(conn: sqlite3.Connection, *, limit: int = 20) -> list[DreamRun]:
    rows = conn.execute(
        "SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> DreamRun:
    return DreamRun(
        id=r["id"], started_at=r["started_at"], ended_at=r["ended_at"],
        cycle_mode=r["cycle_mode"], trigger_reason=r["trigger_reason"],
        stages=json.loads(r["stages"] or "{}"),
        model_used=r["model_used"], status=r["status"], error=r["error"],
    )
