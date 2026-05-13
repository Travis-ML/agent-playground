"""Dreamer daemon CLI.

Usage:
    python -m mcp_servers.memory.dreamer serve              # background loop
    python -m mcp_servers.memory.dreamer run --cycle full   # single cycle, exit
    python -m mcp_servers.memory.dreamer status             # print status
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.repo.dream_runs import list_recent

_DEFAULT_DB = Path.home() / ".travisml-playground" / "memory.db"


def _open():
    p = Path(os.getenv("TRAVISML_MEMORY_DB", str(_DEFAULT_DB)))
    conn = open_connection(p)
    apply_migrations(conn)
    return conn


def cmd_run(args: argparse.Namespace) -> int:
    conn = _open()
    stages = all_stages()
    dr = run_cycle(
        conn=conn, pid=os.getpid(),
        cycle_mode=args.cycle,
        trigger_reason="manual",
        model_used=args.model,
        stages=stages,
    )
    print(json.dumps({"dream_run_id": dr.id, "status": dr.status}))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _open()
    recent = list_recent(conn, limit=5)
    print(json.dumps([{
        "id": r.id, "cycle_mode": r.cycle_mode, "status": r.status,
        "started_at": r.started_at, "ended_at": r.ended_at,
        "stages": list(r.stages.keys()),
    } for r in recent], indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Trigger loop. Polls for trigger conditions; runs cycles. Exits on
    SIGINT. v1 cadence is hard-coded; tuning lives in dreamer_config (Phase 16)."""
    while True:
        # Phase 7 stub: no trigger logic yet — just sleep. Phase 8+ adds it.
        time.sleep(60)
        # full cycle every minute is far too aggressive; this stub is
        # replaced by triggers.py in a later phase.
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory.dreamer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--cycle", choices=["light", "full", "maintenance"],
                       default="full")
    p_run.add_argument("--model", default=os.getenv("DREAMER_MODEL", "vllm/local"))
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve")
    p_serve.set_defaults(func=cmd_serve)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
