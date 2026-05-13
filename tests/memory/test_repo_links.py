import sqlite3

from mcp_servers.memory.repo.links import (
    add_link,
    list_links_from,
    list_links_to,
)


def test_add_link_idempotent(conn: sqlite3.Connection) -> None:
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="entity", dst_id="en_x",
             link_type="about", weight=1.0, dream_run="dr_1")
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="entity", dst_id="en_x",
             link_type="about", weight=2.0, dream_run="dr_2")
    rows = list_links_from(conn, src_kind="episode", src_id="ep_1")
    assert len(rows) == 1


def test_list_links_to_filters_destination(conn: sqlite3.Connection) -> None:
    add_link(conn, src_kind="episode", src_id="ep_1",
             dst_kind="fact", dst_id="fa_a",
             link_type="extracted_from", weight=1.0)
    add_link(conn, src_kind="episode", src_id="ep_2",
             dst_kind="fact", dst_id="fa_a",
             link_type="extracted_from", weight=1.0)
    rows = list_links_to(conn, dst_kind="fact", dst_id="fa_a")
    assert {r["src_id"] for r in rows} == {"ep_1", "ep_2"}


def test_list_links_from_filters_source(conn: sqlite3.Connection) -> None:
    add_link(conn, src_kind="fact", src_id="fa_x",
             dst_kind="entity", dst_id="en_1",
             link_type="subject", weight=1.0)
    add_link(conn, src_kind="fact", src_id="fa_x",
             dst_kind="entity", dst_id="en_2",
             link_type="object", weight=1.0)
    rows = list_links_from(conn, src_kind="fact", src_id="fa_x")
    assert {r["dst_id"] for r in rows} == {"en_1", "en_2"}
