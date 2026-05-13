"""Lifecycle controller used by the Streamlit Dreaming page."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path


def write_pid_file(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def read_pid_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def clear_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as e:
        return e.errno == errno.EPERM


class DaemonController:
    def __init__(
        self,
        pid_file: Path | None = None,
        python: str | None = None,
    ) -> None:
        self.pid_file = pid_file or (
            Path.home() / ".travisml-playground" / "dreamer.pid"
        )
        self.python = python or sys.executable

    def status(self) -> dict:
        pid = read_pid_file(self.pid_file)
        if pid is None:
            return {"running": False, "pid": None}
        if not _alive(pid):
            clear_pid_file(self.pid_file)
            return {"running": False, "pid": None}
        return {"running": True, "pid": pid}

    def start(self) -> int:
        st = self.status()
        if st["running"]:
            return int(st["pid"])
        proc = subprocess.Popen(
            [self.python, "-m", "mcp_servers.memory.dreamer", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        write_pid_file(self.pid_file, proc.pid)
        return proc.pid

    def stop(self) -> bool:
        st = self.status()
        if not st["running"]:
            return False
        try:
            os.kill(int(st["pid"]), 15)  # SIGTERM
        finally:
            clear_pid_file(self.pid_file)
        return True
