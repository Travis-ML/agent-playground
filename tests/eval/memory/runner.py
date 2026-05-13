"""Replay a scenario, run a full dream cycle, query held-out questions.

Usage:
    .agent-playground/bin/python -m tests.eval.memory.runner \
        tests/eval/memory/scenarios/01_user_preferences
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.extractor.pump import pump_once
from mcp_servers.memory.repo.raw_turns import record_turn
from mcp_servers.memory.retrieval.recall import recall


def _seed_raw_turns(conn, conv_path: Path) -> None:
    data = json.loads(conv_path.read_text())
    for i, m in enumerate(data["messages"]):
        record_turn(
            conn,
            conversation_id=data["id"],
            turn_index=i,
            role=m["role"],
            occurred_at=m.get("ts", "2026-05-01T15:00:00Z"),
        )


def _load_llm(provider: str = "lmstudio", model: str | None = None):
    from playground.providers.registry import get_client
    return get_client(provider, model or os.getenv("DREAMER_MODEL", "local"))


def run_scenario(scenario_dir: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="memeval-"))
    conversations_root = tmp / "conversations"
    db_path = tmp / "memory.db"
    try:
        # mirror conversation files into the temp root
        page_dir = conversations_root / "basic_chat"
        page_dir.mkdir(parents=True)
        for conv in (scenario_dir / "conversations").glob("*.json"):
            shutil.copy(conv, page_dir / conv.name)

        conn = open_connection(db_path)
        apply_migrations(conn)
        for conv in (scenario_dir / "conversations").glob("*.json"):
            _seed_raw_turns(conn, conv)

        llm = _load_llm()
        # 1) extract atomic episodes
        pump_once(conn=conn, llm=llm, conversations_root=conversations_root)
        # 2) run a full dream cycle
        run_cycle(
            conn=conn, pid=os.getpid(), cycle_mode="full",
            trigger_reason="manual", model_used="vllm/local",
            stages=all_stages(), ctx={"model": "local"},
        )

        # 3) ask held-out questions via recall
        qs = yaml.safe_load((scenario_dir / "questions.yaml").read_text())
        report = {"questions": []}
        for q in qs:
            memories = recall(conn=conn, query=q["query"], max_results=5)
            report["questions"].append({
                "id": q["id"], "query": q["query"],
                "reference": q["reference"].strip(),
                "memories": memories,
            })
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scenario_dir", type=Path)
    args = p.parse_args(argv)
    report = run_scenario(args.scenario_dir)
    sys.stdout.write(json.dumps(report, indent=2, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
