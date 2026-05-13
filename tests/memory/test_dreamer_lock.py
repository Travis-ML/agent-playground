import os
import sqlite3

import pytest

from mcp_servers.memory.dreamer_runner.lifecycle import (
    LockHeld,
    acquire_lock,
    heartbeat,
    release_lock,
)


def test_acquire_then_release(conn: sqlite3.Connection) -> None:
    acquire_lock(conn, pid=os.getpid())
    row = conn.execute("SELECT pid FROM dreamer_lock WHERE id = 1").fetchone()
    assert row["pid"] == os.getpid()
    release_lock(conn, pid=os.getpid())
    row = conn.execute("SELECT * FROM dreamer_lock WHERE id = 1").fetchone()
    assert row is None


def test_acquire_blocks_when_already_held_by_live_pid(
    conn: sqlite3.Connection,
) -> None:
    acquire_lock(conn, pid=os.getpid())
    with pytest.raises(LockHeld):
        acquire_lock(conn, pid=os.getpid() + 99_999, allow_steal_stale=False)


def test_acquire_steals_stale_lock_with_dead_pid(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "INSERT INTO dreamer_lock (id, pid, acquired_at, heartbeat) "
        "VALUES (1, 999999999, '2026-05-12T15:00:00Z', '2026-05-12T15:00:00Z')"
    )
    acquire_lock(conn, pid=os.getpid(), allow_steal_stale=True)
    row = conn.execute("SELECT pid FROM dreamer_lock WHERE id = 1").fetchone()
    assert row["pid"] == os.getpid()


def test_heartbeat_updates_only_for_owner(conn: sqlite3.Connection) -> None:
    acquire_lock(conn, pid=os.getpid())
    heartbeat(conn, pid=os.getpid())
    with pytest.raises(LockHeld):
        heartbeat(conn, pid=os.getpid() + 1)
