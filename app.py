"""TravisML Agent Playground — Home page."""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from playground.branding import (
    inject_brand_css,
    render_brand_wordmark,
    render_theme_toggle,
)

load_dotenv()

st.set_page_config(
    page_title="TravisML Agent Playground",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_brand_css()
render_brand_wordmark()


# --- Hero ---------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex;gap:22px;flex-wrap:wrap;margin-bottom:36px;">
      <span class="tml-label">Edition / 001</span>
      <span class="tml-label">Local · v0.1</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="font-size:clamp(40px,7vw,82px);line-height:0.95;
               letter-spacing:-0.035em;max-width:14ch;margin-bottom:28px;">
      Build, test, debug <em>agents</em>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="border-top:1px solid var(--line);padding-top:24px;
                font-size:17px;color:var(--text-200);max-width:60ch;
                line-height:1.6;margin-bottom:48px;">
      <strong style="color:var(--text-100);font-weight:500;">
        TravisML Agent Playground
      </strong> is a branded harness for experimenting with agentic systems —
      chat, tools, memory, prompts, and MCP servers — across multiple model
      providers.
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Provider status grid ----------------------------------------------

st.markdown('<div class="tml-label">Providers</div>', unsafe_allow_html=True)


def _status_card(title: str, model_summary: str, connected: bool) -> str:
    accent = "var(--accent)" if connected else "#C97A2A"
    label = "Connected" if connected else "Awaiting setup"
    return f"""
    <div style="background:var(--bg-deep);border:1px solid var(--line);
                padding:22px;height:100%;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  letter-spacing:0.16em;text-transform:uppercase;
                  color:{accent};display:flex;align-items:center;gap:8px;
                  margin-bottom:8px;">
        <span style="display:inline-block;width:5px;height:5px;
                     background:{accent};"></span>{label}
      </div>
      <div style="font-family:'Fraunces',serif;font-weight:600;font-size:18px;
                  color:var(--text-100);letter-spacing:-0.01em;
                  margin-bottom:6px;">{title}</div>
      <div style="font-size:13px;color:var(--text-300);line-height:1.55;">
        {model_summary}
      </div>
    </div>
    """


anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
lmstudio_url = html.escape(os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))

cols = st.columns(3, gap="medium")
with cols[0]:
    st.markdown(
        _status_card("Anthropic / Claude", "opus-4-7, sonnet-4-6, haiku-4-5", anthropic_ok),
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        _status_card("OpenAI / GPT", "gpt-4o, gpt-4o-mini, o1", openai_ok),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        _status_card("LM Studio / Local", lmstudio_url, False),
        unsafe_allow_html=True,
    )

st.write("")
st.write("")


# --- MCP servers list ---------------------------------------------------

mcp_path = Path("mcp.json")
st.markdown('<div class="tml-label">MCP servers</div>', unsafe_allow_html=True)

if mcp_path.exists():
    st.caption(f"Configured in `{mcp_path}`. Toggle them per-page in Basic Chat.")
else:
    st.info("No `mcp.json` yet — it'll be created in Phase 8 of the plan.")


render_theme_toggle()
