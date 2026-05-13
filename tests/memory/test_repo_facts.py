"""Tests for facts repo."""

from __future__ import annotations

import sqlite3

from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.facts import (
    current_belief, insert_new_fact,
)


def _ent(conn: sqlite3.Connection, name: str) -> str:
    return get_or_create(
        conn, canonical_name=name, kind="concept",
        seen_at="2026-05-12T15:00:00Z",
    ).id


def test_insert_new_fact_creates_current_belief(conn: sqlite3.Connection) -> None:
    user = _ent(conn, "Travis")
    python = _ent(conn, "Python")
    f = insert_new_fact(
        conn,
        subject_entity=user,
        predicate="uses",
        object_entity=python,
        object_value=None,
        valid_from="2026-05-12T15:00:00Z",
        learned_at="2026-05-12T15:01:00Z",
        source_episode_ids=["ep_a"],
        confidence=0.9,
        created_in_dream_run="dr_test",
    )
    assert f.valid_to is None and f.invalidated_at is None
    assert f.supersedes is None and f.superseded_by is None
    found = current_belief(conn, subject_entity=user, predicate="uses")
    assert found and found.id == f.id
