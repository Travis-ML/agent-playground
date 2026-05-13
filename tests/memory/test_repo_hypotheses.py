import sqlite3

from mcp_servers.memory.repo.hypotheses import (
    insert_hypothesis,
    list_by_status,
    resolve,
)


def test_insert_lists_under_open_status(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="X relates to Y",
        source_node_ids=["ep_1", "ep_2"], confidence=0.42,
        created_in_dream_run="dr_1",
    )
    assert h.status == "open"
    assert [x.id for x in list_by_status(conn, "open")] == [h.id]


def test_resolve_corroborated(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="A causes B", source_node_ids=[],
        confidence=0.5, created_in_dream_run="dr_1",
    )
    resolve(conn, h.id, status="corroborated",
            resolved_by="operator", note="confirmed in conversation")
    out = list_by_status(conn, "corroborated")
    assert [o.id for o in out] == [h.id]
    assert out[0].resolution_note == "confirmed in conversation"


def test_resolve_refuted(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="B causes A", source_node_ids=[],
        confidence=0.3, created_in_dream_run="dr_1",
    )
    resolve(conn, h.id, status="refuted",
            resolved_by="operator", note="disproven")
    out = list_by_status(conn, "refuted")
    assert [o.id for o in out] == [h.id]
    assert out[0].status == "refuted"


def test_resolve_set_aside(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="C and D", source_node_ids=[],
        confidence=0.4, created_in_dream_run="dr_1",
    )
    resolve(conn, h.id, status="set_aside",
            resolved_by="operator", note="insufficient evidence")
    out = list_by_status(conn, "set_aside")
    assert [o.id for o in out] == [h.id]


def test_resolve_invalid_status_raises(conn: sqlite3.Connection) -> None:
    h = insert_hypothesis(
        conn, statement="X", source_node_ids=[],
        confidence=0.5, created_in_dream_run="dr_1",
    )
    try:
        resolve(conn, h.id, status="invalid_status",
                resolved_by="operator")
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "invalid status" in str(e)
