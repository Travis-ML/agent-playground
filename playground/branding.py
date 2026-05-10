"""Brand application — palettes, font loading, wordmark, theme toggle."""

from __future__ import annotations

import textwrap
from typing import TypedDict

import streamlit as st


class Palette(TypedDict):
    bg_void: str
    bg_deep: str
    bg_panel: str
    bg_elevated: str
    line: str
    line_strong: str
    text_100: str
    text_200: str
    text_300: str
    text_400: str
    accent: str
    accent_bright: str


LIGHT: Palette = {
    "bg_void": "#F4F1EA",
    "bg_deep": "#EDE9DE",
    "bg_panel": "#E6E1D2",
    "bg_elevated": "#DDD8C8",
    "line": "rgba(15,30,22,0.10)",
    "line_strong": "rgba(15,30,22,0.24)",
    "text_100": "#0F1E16",
    "text_200": "#2A3D32",
    "text_300": "#5A6D62",
    "text_400": "#8B9C92",
    "accent": "#047A5E",
    "accent_bright": "#0BA37A",
}

DARK: Palette = {
    "bg_void": "#0D1612",
    "bg_deep": "#131C17",
    "bg_panel": "#1A241D",
    "bg_elevated": "#222D25",
    "line": "rgba(244,241,234,0.08)",
    "line_strong": "rgba(244,241,234,0.20)",
    "text_100": "#F4F1EA",
    "text_200": "#D4D0C2",
    "text_300": "#9AA89E",
    "text_400": "#6E7C73",
    "accent": "#0BA37A",
    "accent_bright": "#14C490",
}

_FONTS_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&"
    "family=JetBrains+Mono:wght@400;500&"
    "family=Sora:wght@300;400;500&display=swap"
)


def get_theme() -> Palette:
    """Return the active palette; defaults to light."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    return DARK if st.session_state.theme == "dark" else LIGHT


def inject_brand_css() -> None:
    """Emit brand CSS scoped to the current theme. Call once per page."""
    t = get_theme()
    css = textwrap.dedent(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="{_FONTS_HREF}" rel="stylesheet">
        <style>
        :root {{
            --bg-void: {t['bg_void']};
            --bg-deep: {t['bg_deep']};
            --bg-panel: {t['bg_panel']};
            --bg-elevated: {t['bg_elevated']};
            --line: {t['line']};
            --line-strong: {t['line_strong']};
            --text-100: {t['text_100']};
            --text-200: {t['text_200']};
            --text-300: {t['text_300']};
            --text-400: {t['text_400']};
            --accent: {t['accent']};
            --accent-bright: {t['accent_bright']};
        }}

        html, body, [class*="st-"] {{
            font-family: 'Sora', sans-serif;
            font-weight: 300;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--text-100);
        }}
        h1 em, h2 em, h3 em {{
            font-style: italic;
            color: var(--accent);
            font-feature-settings: "ss01";
        }}

        code, pre, .stCodeBlock {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .stApp {{
            background: var(--bg-void);
            color: var(--text-200);
        }}
        [data-testid="stSidebar"] {{
            background: var(--bg-deep);
            border-right: 1px solid var(--line);
        }}

        .tml-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--text-300);
        }}
        .tml-label::before {{
            content: '';
            display: inline-block;
            width: 5px;
            height: 5px;
            background: var(--accent);
            margin-right: 8px;
            vertical-align: 2px;
        }}

        .block-container {{
            padding-top: 2rem;
            max-width: 1100px;
        }}
        </style>
        """
    ).strip()
    st.markdown(css, unsafe_allow_html=True)


def render_brand_wordmark() -> None:
    """Render the 'TravisML / Playground' wordmark in the sidebar."""
    html = textwrap.dedent(
        """
        <div style="font-family:'Fraunces',serif;font-weight:600;font-size:22px;
                    line-height:1.05;color:var(--text-100);margin-bottom:6px;">
          TravisML<br>
          <em style="font-style:italic;font-weight:500;color:var(--accent);
                     font-feature-settings:'ss01';">Playground</em>
        </div>
        <div class="tml-label" style="margin-bottom:24px;">Agent harness · v0.1</div>
        """
    ).strip()
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Sidebar widget — sticks to the bottom of the sidebar."""
    with st.sidebar:
        st.divider()
        current = st.session_state.get("theme", "light")
        is_dark = st.toggle("Dark mode", value=(current == "dark"), key="_theme_toggle")
        next_theme = "dark" if is_dark else "light"
        if next_theme != current:
            st.session_state.theme = next_theme
            st.rerun()
