import os
import sqlite3
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.runner import run_cycle


def test_run_cycle_records_dream_run_with_stages(
    conn: sqlite3.Connection,
) -> None:
    fake_stages = {
        "ingest_cluster":   MagicMock(return_value={"clusters": 0}),
        "consolidate":      MagicMock(return_value={"deduped": 0}),
        "extract":          MagicMock(return_value={"facts_added": 0}),
        "reflect":          MagicMock(return_value={"reflections_added": 0}),
        "recombine":        MagicMock(return_value={"hypotheses_added": 0}),
        "decay_reindex":    MagicMock(return_value={"archived": 0}),
    }
    dr = run_cycle(
        conn=conn, pid=os.getpid(),
        cycle_mode="full", trigger_reason="manual",
        model_used="vllm/test",
        stages=fake_stages,
    )
    assert dr.status == "completed"
    assert set(dr.stages.keys()) == set(fake_stages.keys())


def test_light_cycle_runs_only_subset(conn: sqlite3.Connection) -> None:
    calls = {n: MagicMock(return_value={}) for n in [
        "ingest_cluster", "consolidate", "extract",
        "reflect", "recombine", "decay_reindex",
    ]}
    run_cycle(
        conn=conn, pid=os.getpid(), cycle_mode="light",
        trigger_reason="manual", model_used="vllm/test", stages=calls,
    )
    # light = ingest_cluster, consolidate, extract, decay_reindex
    for name in ("ingest_cluster", "consolidate", "extract", "decay_reindex"):
        calls[name].assert_called_once()
    for name in ("reflect", "recombine"):
        calls[name].assert_not_called()
