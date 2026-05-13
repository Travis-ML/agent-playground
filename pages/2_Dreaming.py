"""Operator console for the memory dreamer."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations
from mcp_servers.memory.dreamer_runner.control import DaemonController
from mcp_servers.memory.dreamer_runner.runner import run_cycle
from mcp_servers.memory.dreamer_runner.stages import all_stages
from mcp_servers.memory.repo.dream_runs import list_recent
from mcp_servers.memory.repo.hypotheses import list_by_status, resolve
from playground.branding import (
    inject_brand_css, render_brand_wordmark, render_theme_toggle,
)

load_dotenv()
st.set_page_config(
    page_title="Dreaming — TravisML Playground",
    page_icon="◐", layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
inject_brand_css()
render_brand_wordmark()


@st.cache_resource(show_spinner=False)
def _conn():
    from pathlib import Path
    p = Path.home() / ".travisml-playground" / "memory.db"
    c = open_connection(p)
    apply_migrations(c)
    return c


@st.cache_resource(show_spinner=False)
def _controller() -> DaemonController:
    return DaemonController()


st.html('<h1 style="font-size:36px;margin-bottom:8px;">Dream<em>ing</em></h1>')
st.caption("Operator console for the memory + dreaming subsystem.")
st.divider()


# ----- daemon panel -----
ctrl = _controller()
conn = _conn()
status = ctrl.status()

cols = st.columns([2, 2, 6])
with cols[0]:
    st.html('<div class="tml-label">Daemon</div>')
    st.write("running" if status["running"] else "stopped")
    if status["running"]:
        st.caption(f"pid {status['pid']}")

with cols[1]:
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("raw_turn_refs", "episodes", "facts", "reflections",
                  "hypotheses", "links")
    }
    st.html('<div class="tml-label">Memory size</div>')
    for k, v in counts.items():
        st.caption(f"{k}: {v}")

with cols[2]:
    st.html('<div class="tml-label">Controls</div>')
    btn_cols = st.columns(3)
    if btn_cols[0].button(
        "Start daemon" if not status["running"] else "Stop daemon",
        use_container_width=True,
    ):
        if status["running"]:
            ctrl.stop()
        else:
            ctrl.start()
        st.rerun()
    cycle = btn_cols[1].selectbox(
        "Dream now…", ["light", "full", "maintenance"], key="cycle_choice",
    )
    if btn_cols[2].button("Dream now", use_container_width=True):
        try:
            import os
            run_cycle(
                conn=conn, pid=os.getpid(),
                cycle_mode=cycle, trigger_reason="manual",
                model_used="vllm/local", stages=all_stages(),
            )
            st.success("dream cycle completed")
        except Exception as e:
            st.error(f"dream failed: {e}")
        st.rerun()

st.divider()


# ----- recent dreams -----
st.html('<div class="tml-label">Recent dreams</div>')
recent = list_recent(conn, limit=10)
if not recent:
    st.caption("No dream runs yet.")
else:
    for dr in recent:
        with st.container(border=True):
            head = f"**{dr.started_at}** · {dr.cycle_mode} · {dr.status}"
            if dr.ended_at:
                head += f" · ended {dr.ended_at}"
            st.markdown(head)
            if dr.stages:
                cells = st.columns(min(6, max(1, len(dr.stages))))
                for i, (name, metrics) in enumerate(dr.stages.items()):
                    with cells[i % len(cells)]:
                        st.caption(name)
                        st.json(metrics, expanded=False)
            if dr.error:
                st.error(dr.error)

st.divider()
st.html('<div class="tml-label">Open hypotheses</div>')

open_hyps = list_by_status(conn, "open", limit=20)
if not open_hyps:
    st.caption("No open hypotheses yet — run a full dream cycle to generate some.")
else:
    for h in open_hyps:
        with st.container(border=True):
            st.markdown(f"**?** {h.statement}")
            st.caption(
                f"sources: {', '.join(h.source_node_ids[:3])} "
                f"· confidence {h.confidence:.2f} · created {h.created_at}"
            )
            c1, c2, c3, _ = st.columns([2, 2, 2, 6])
            if c1.button("Corroborate", key=f"corr_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="corroborated", resolved_by="operator")
                st.rerun()
            if c2.button("Refute", key=f"ref_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="refuted", resolved_by="operator")
                st.rerun()
            if c3.button("Set aside", key=f"aside_{h.id}", use_container_width=True):
                resolve(conn, h.id, status="set_aside", resolved_by="operator")
                st.rerun()

render_theme_toggle()
