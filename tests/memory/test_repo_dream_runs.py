import sqlite3

from mcp_servers.memory.repo.dream_runs import (
    finish_run,
    list_recent,
    record_stage,
    start_run,
)


def test_start_finish_records_lifecycle(conn: sqlite3.Connection) -> None:
    dr = start_run(conn, cycle_mode="full", trigger_reason="manual",
                   model_used="vllm/gemma-4-31b")
    record_stage(conn, dr.id, name="cluster",
                 metrics={"clusters": 4, "wall_ms": 120})
    finish_run(conn, dr.id, status="completed")

    rows = list_recent(conn, limit=10)
    assert len(rows) == 1
    assert rows[0].status == "completed"
    assert rows[0].stages.get("cluster") == {"clusters": 4, "wall_ms": 120}


def test_record_stage_merges(conn: sqlite3.Connection) -> None:
    dr = start_run(conn, cycle_mode="full", trigger_reason="manual",
                   model_used="vllm/gemma-4-31b")
    record_stage(conn, dr.id, name="stage1", metrics={"items": 10})
    record_stage(conn, dr.id, name="stage2", metrics={"items": 5})
    finish_run(conn, dr.id, status="completed")

    rows = list_recent(conn, limit=10)
    assert len(rows) == 1
    assert rows[0].stages.get("stage1") == {"items": 10}
    assert rows[0].stages.get("stage2") == {"items": 5}


def test_record_stage_overwrites_same_name(conn: sqlite3.Connection) -> None:
    dr = start_run(conn, cycle_mode="full", trigger_reason="manual",
                   model_used="vllm/gemma-4-31b")
    record_stage(conn, dr.id, name="stage1", metrics={"items": 10})
    record_stage(conn, dr.id, name="stage1", metrics={"items": 20})
    finish_run(conn, dr.id, status="completed")

    rows = list_recent(conn, limit=10)
    assert len(rows) == 1
    assert rows[0].stages.get("stage1") == {"items": 20}


def test_finish_with_error(conn: sqlite3.Connection) -> None:
    dr = start_run(conn, cycle_mode="full", trigger_reason="manual",
                   model_used="vllm/gemma-4-31b")
    finish_run(conn, dr.id, status="failed", error="timeout after 300s")

    rows = list_recent(conn, limit=10)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error == "timeout after 300s"
