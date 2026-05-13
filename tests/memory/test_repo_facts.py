"""Tests for facts repo."""

from __future__ import annotations

import sqlite3

import pytest

from mcp_servers.memory.repo.entities import get_or_create
from mcp_servers.memory.repo.facts import (
    current_belief, insert_new_fact, list_facts_for_subject_predicate,
    supersede_fact,
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


def get_by_id_via_repo(conn, fact_id):
    from mcp_servers.memory.repo.facts import get_by_id
    return get_by_id(conn, fact_id)


def test_supersede_closes_old_and_creates_new(conn: sqlite3.Connection) -> None:
    user = _ent(conn, "Travis")
    py3_13 = _ent(conn, "Python 3.13")
    py3_14 = _ent(conn, "Python 3.14")
    old = insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=py3_13, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=["ep_1"], confidence=0.9,
        created_in_dream_run="dr_1",
    )

    new = supersede_fact(
        conn, old_fact_id=old.id,
        new_object_entity=py3_14, new_object_value=None,
        change_time="2026-05-12T15:00:00Z",
        source_episode_ids=["ep_2"], confidence=0.95,
        created_in_dream_run="dr_2",
    )

    old_refreshed = get_by_id_via_repo(conn, old.id)
    assert old_refreshed.valid_to == "2026-05-12T15:00:00Z"
    assert old_refreshed.invalidated_at == "2026-05-12T15:00:00Z"
    assert old_refreshed.superseded_by == new.id

    assert new.valid_from == "2026-05-12T15:00:00Z"
    assert new.supersedes == old.id
    assert new.valid_to is None and new.invalidated_at is None

    cb = current_belief(conn, subject_entity=user, predicate="uses")
    assert cb and cb.id == new.id


def test_at_most_one_current_belief_per_subject_predicate(
    conn: sqlite3.Connection,
) -> None:
    user = _ent(conn, "Travis")
    a = _ent(conn, "A")
    b = _ent(conn, "B")
    f1 = insert_new_fact(
        conn, subject_entity=user, predicate="uses",
        object_entity=a, object_value=None,
        valid_from="2026-04-01T00:00:00Z", learned_at="2026-04-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_1",
    )
    supersede_fact(
        conn, old_fact_id=f1.id,
        new_object_entity=b, new_object_value=None,
        change_time="2026-05-01T00:00:00Z",
        source_episode_ids=[], confidence=0.9, created_in_dream_run="dr_2",
    )

    cur = list_facts_for_subject_predicate(
        conn, subject_entity=user, predicate="uses", currently_believed=True,
    )
    assert len(cur) == 1
