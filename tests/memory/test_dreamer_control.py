from pathlib import Path

from mcp_servers.memory.dreamer_runner.control import (
    DaemonController, write_pid_file, read_pid_file, clear_pid_file,
)


def test_pid_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.pid"
    write_pid_file(p, 12345)
    assert read_pid_file(p) == 12345
    clear_pid_file(p)
    assert read_pid_file(p) is None


def test_controller_status_reports_not_running_when_no_pid(tmp_path: Path) -> None:
    c = DaemonController(pid_file=tmp_path / "dreamer.pid")
    assert c.status() == {"running": False, "pid": None}
