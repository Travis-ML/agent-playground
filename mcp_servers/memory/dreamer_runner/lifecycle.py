"""Dreamer lifecycle helpers: advisory write-lock with PID + heartbeat.

Only one dreamer holds the lock at a time. If a previous dreamer crashed
without releasing, the next dreamer detects the stale lock via PID check
and reclaims it.
"""

from __future__ import annotations

import errno
import os
import sqlite3
from datetime import UTC, datetime


class LockHeld(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
    except OSError as e:
        return e.errno == errno.EPERM


def acquire_lock(
    conn: sqlite3.Connection, *, pid: int, allow_steal_stale: bool = True,
) -> None:
    existing = conn.execute(
        "SELECT pid FROM dreamer_lock WHERE id = 1"
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO dreamer_lock (id, pid, acquired_at, heartbeat) "
            "VALUES (1, ?, ?, ?)",
            (pid, _now(), _now()),
        )
        return
    if existing["pid"] == pid:
        # already ours
        return
    if allow_steal_stale and not _pid_alive(existing["pid"]):
        conn.execute(
            "UPDATE dreamer_lock SET pid = ?, acquired_at = ?, heartbeat = ? "
            "WHERE id = 1",
            (pid, _now(), _now()),
        )
        return
    raise LockHeld(f"dreamer_lock held by pid={existing['pid']}")


def heartbeat(conn: sqlite3.Connection, *, pid: int) -> None:
    cur = conn.execute(
        "UPDATE dreamer_lock SET heartbeat = ? WHERE id = 1 AND pid = ?",
        (_now(), pid),
    )
    if cur.rowcount == 0:
        raise LockHeld(f"not the lock owner: pid={pid}")


def release_lock(conn: sqlite3.Connection, *, pid: int) -> None:
    conn.execute("DELETE FROM dreamer_lock WHERE id = 1 AND pid = ?", (pid,))
