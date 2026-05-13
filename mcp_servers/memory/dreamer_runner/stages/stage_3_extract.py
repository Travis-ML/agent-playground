"""Stage stub — implementation lands in its dedicated phase."""

from __future__ import annotations

import sqlite3
from typing import Any


def run(*, conn: sqlite3.Connection, dream_run_id: str, ctx: dict, **_: Any) -> dict:
    return {}
