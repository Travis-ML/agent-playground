"""Orchestrates a single dream cycle, threading a `ctx` dict between stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mcp_servers.memory.dreamer_runner.lifecycle import (
    acquire_lock,
    heartbeat,
    release_lock,
)
from mcp_servers.memory.models import DreamRun
from mcp_servers.memory.repo.dream_runs import (
    finish_run,
    list_recent,
    record_stage,
    start_run,
)

_CYCLE_STAGES: dict[str, list[str]] = {
    "light":       ["ingest_cluster", "consolidate", "extract", "decay_reindex"],
    "full":        ["ingest_cluster", "consolidate", "extract",
                    "reflect", "recombine", "decay_reindex"],
    "maintenance": ["decay_reindex"],
}


StageFn = Callable[..., dict[str, Any]]


def run_cycle(
    *,
    conn: sqlite3.Connection,
    pid: int,
    cycle_mode: str,
    trigger_reason: str,
    model_used: str,
    stages: dict[str, StageFn],
    ctx: dict[str, Any] | None = None,
) -> DreamRun:
    if ctx is None:
        ctx = {}
    acquire_lock(conn, pid=pid)
    try:
        dr = start_run(
            conn, cycle_mode=cycle_mode, trigger_reason=trigger_reason,
            model_used=model_used,
        )
        try:
            for name in _CYCLE_STAGES[cycle_mode]:
                fn = stages[name]
                result = fn(conn=conn, dream_run_id=dr.id, ctx=ctx) or {}
                # Stage returns either a metrics dict, or a dict with
                # "metrics" + "ctx_updates" keys.
                if "metrics" in result or "ctx_updates" in result:
                    metrics = result.get("metrics", {})
                    ctx.update(result.get("ctx_updates", {}))
                else:
                    metrics = result
                record_stage(conn, dr.id, name=name, metrics=metrics)
                heartbeat(conn, pid=pid)
            finish_run(conn, dr.id, status="completed")
        except Exception as e:
            finish_run(conn, dr.id, status="failed", error=str(e))
            raise
        return list_recent(conn, limit=1)[0]
    finally:
        release_lock(conn, pid=pid)
