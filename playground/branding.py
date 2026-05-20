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

_ICONS_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=block"
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
        <link href="{_ICONS_HREF}" rel="stylesheet">
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

        /* Restore Material Symbols icon font for icon-bearing elements.
           Streamlit's icons use ligatures (e.g. "keyboard_double_arrow_left"
           resolves to an arrow glyph). Our broad [class*="st-"] font-family rule
           was overriding the icon font and breaking the ligatures. */
        .material-icons,
        .material-symbols-outlined,
        .material-symbols-rounded,
        .material-symbols-sharp,
        .material-icons-outlined,
        .material-icons-round,
        .material-icons-sharp,
        .material-icons-two-tone,
        [class*="material-symbol"],
        [class*="material-icon"],
        [data-testid="stIconMaterial"],
        [data-testid="stExpanderIcon"],
        [data-testid="stExpanderIconCheck"],
        [data-testid="stExpanderIconError"],
        [data-testid="stExpanderIconSpinner"],
        span[translate="no"],
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stSidebarCollapsedControl"] span,
        [data-testid="collapsedControl"] span,
        [aria-label*="Close sidebar"] span,
        [aria-label*="Open sidebar"] span,
        [aria-label*="collapse"] span,
        [data-testid="stHeader"] [kind="header"] span,
        [data-testid="stToolbar"] [kind="header"] span {{
          font-family: 'Material Symbols Outlined', 'Material Icons' !important;
          font-weight: normal !important;
          font-style: normal !important;
          font-feature-settings: 'liga' !important;
          text-transform: none !important;
          letter-spacing: normal !important;
          word-wrap: normal !important;
          white-space: nowrap !important;
          direction: ltr !important;
          -webkit-font-smoothing: antialiased !important;
          font-size: 24px !important;
          line-height: 1 !important;
        }}

        .stApp {{
            background: var(--bg-void);
            color: var(--text-200);
        }}
        [data-testid="stSidebar"] {{
            background: var(--bg-deep);
            border-right: 1px solid var(--line);
        }}

        /* Render the brand wordmark above Streamlit's auto-generated page
           nav by flipping their flex order. Both elements live as siblings
           inside stSidebarContent. */
        [data-testid="stSidebarContent"] {{
            display: flex !important;
            flex-direction: column !important;
        }}
        [data-testid="stSidebarUserContent"] {{ order: 1 !important; }}
        [data-testid="stSidebarNav"] {{ order: 2 !important; }}

        /* Replace Streamlit's default sidebar collapse/expand controls with
           thin vertical edge buttons. Streamlit's default position (top-
           center of the sidebar header) and low default opacity make these
           hard to find on this brand's palette, and they depend on the
           Material Symbols font being applied to the right testid. We hide
           Streamlit's icon children and inject our own chevrons via
           ::before using JetBrains Mono — no icon-font dependency. */
        [data-testid="stSidebar"] {{
            position: relative !important;
        }}

        [data-testid="stSidebarCollapseButton"] {{
            position: absolute !important;
            top: 50% !important;
            right: 0 !important;
            transform: translateY(-50%) !important;
            z-index: 100 !important;
            width: 18px !important;
            height: 64px !important;
            min-width: unset !important;
            padding: 0 !important;
            background: var(--bg-deep) !important;
            border: none !important;
            border-left: 1px solid var(--line) !important;
            border-radius: 0 !important;
            opacity: 1 !important;
            color: var(--text-200) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: color 0.15s ease, border-color 0.15s ease, width 0.15s ease !important;
        }}
        [data-testid="stSidebarCollapseButton"] > * {{
            display: none !important;
        }}
        [data-testid="stSidebarCollapseButton"]::before {{
            content: "‹" !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 400 !important;
            font-size: 16px !important;
            line-height: 1 !important;
            color: inherit !important;
        }}
        [data-testid="stSidebarCollapseButton"]:hover {{
            color: var(--accent) !important;
            border-left-color: var(--accent) !important;
            width: 22px !important;
        }}

        [data-testid="stSidebarCollapsedControl"] {{
            opacity: 1 !important;
            width: 18px !important;
            height: 64px !important;
            min-width: unset !important;
            padding: 0 !important;
            background: var(--bg-deep) !important;
            border: 1px solid var(--line) !important;
            border-left: none !important;
            border-radius: 0 !important;
            color: var(--text-200) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: color 0.15s ease, border-color 0.15s ease, width 0.15s ease !important;
        }}
        [data-testid="stSidebarCollapsedControl"] > * {{
            display: none !important;
        }}
        [data-testid="stSidebarCollapsedControl"]::before {{
            content: "›" !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 400 !important;
            font-size: 16px !important;
            line-height: 1 !important;
            color: inherit !important;
        }}
        [data-testid="stSidebarCollapsedControl"]:hover {{
            color: var(--accent) !important;
            border-color: var(--accent) !important;
            width: 22px !important;
        }}

        .tml-label {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 10px !important;
            letter-spacing: 0.18em !important;
            text-transform: uppercase !important;
            color: var(--text-300) !important;
        }}
        .tml-label::before {{
            content: '' !important;
            display: inline-block !important;
            width: 5px !important;
            height: 5px !important;
            background: var(--accent) !important;
            margin-right: 8px !important;
            vertical-align: 2px !important;
        }}

        .block-container {{
            padding-top: 2rem;
            max-width: 1100px;
        }}

        /* Streamlit chrome uses high-specificity emotion-css classes; !important
           is required to override them so the theme actually flips on toggle. */

        /* ========== Comprehensive Streamlit chrome theming ========== */

        /* Top header / toolbar (where Deploy button + main menu live) */
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {{
          background: var(--bg-void) !important;
          color: var(--text-100) !important;
        }}
        [data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stHeader"] svg,
        [data-testid="stToolbar"] svg {{
          color: var(--text-100) !important;
          fill: var(--text-100) !important;
        }}
        [data-testid="stHeader"] button:hover,
        [data-testid="stToolbar"] button:hover {{
          background: var(--bg-elevated) !important;
        }}

        /* Bottom area where chat input lives */
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"],
        .stBottom,
        .stBottom > div,
        [data-testid="stBottomBlockContainer"] > div {{
          background: var(--bg-void) !important;
          border-top: 1px solid var(--line) !important;
        }}

        /* Chat input field — multiple nested wrappers */
        [data-testid="stChatInput"],
        [data-testid="stChatInputContainer"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        .stChatInput,
        .stChatInput > div {{
          background: var(--bg-deep) !important;
          border-color: var(--line-strong) !important;
          box-shadow: none !important;
        }}

        /* Chat input textarea / inner editable area */
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] [contenteditable] {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border-color: transparent !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stChatInput"] [contenteditable][placeholder]::before {{
          color: var(--text-400) !important;
        }}

        /* Chat input send button */
        [data-testid="stChatInput"] button {{
          background: var(--bg-elevated) !important;
          color: var(--text-100) !important;
        }}
        [data-testid="stChatInput"] button:hover {{
          background: var(--accent-bright) !important;
          color: var(--bg-void) !important;
        }}

        /* Sidebar page nav links (app, Basic Chat, etc.) */
        [data-testid="stSidebarNav"] a {{
          color: var(--text-200) !important;
        }}
        [data-testid="stSidebarNav"] a:hover {{
          color: var(--text-100) !important;
          background: var(--bg-elevated) !important;
        }}
        [data-testid="stSidebarNav"] a span {{
          color: inherit !important;
        }}
        [data-testid="stSidebarNav"] [aria-current="page"] {{
          background: var(--bg-elevated) !important;
        }}
        [data-testid="stSidebarNav"] [aria-current="page"] a {{
          color: var(--text-100) !important;
        }}

        /* Widget labels (selectbox, slider, text inputs, etc.) */
        .stSelectbox label,
        .stTextArea label,
        .stTextInput label,
        .stNumberInput label,
        .stSlider label,
        .stMultiSelect label,
        .stCheckbox label,
        .stRadio label,
        .stToggle label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p {{
          color: var(--text-200) !important;
        }}

        /* Form widget bodies */
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border-color: var(--line-strong) !important;
        }}
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border-color: var(--line-strong) !important;
        }}
        .stSlider [data-baseweb="slider"] [role="slider"] {{
          background: var(--accent) !important;
        }}

        /* Selectbox dropdown popover */
        [data-baseweb="popover"] [data-baseweb="menu"] {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border-color: var(--line-strong) !important;
        }}
        [data-baseweb="popover"] [role="option"] {{
          color: var(--text-200) !important;
        }}
        [data-baseweb="popover"] [role="option"]:hover {{
          background: var(--bg-elevated) !important;
          color: var(--text-100) !important;
        }}

        /* Chat message bodies */
        [data-testid="stChatMessage"] {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border: 1px solid var(--line);
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {{
          color: var(--text-100) !important;
        }}

        /* Captions / small text */
        [data-testid="stCaptionContainer"],
        .stCaption,
        small {{
          color: var(--text-300) !important;
        }}

        /* Buttons (default Streamlit buttons) */
        .stButton button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border: 1px solid var(--line-strong) !important;
        }}
        .stButton button:hover {{
          background: var(--bg-elevated) !important;
          border-color: var(--accent) !important;
          color: var(--text-100) !important;
        }}

        /* Expanders (used by tool-call rendering) */
        [data-testid="stExpander"] {{
          background: var(--bg-deep) !important;
          border: 1px solid var(--line) !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details summary {{
          color: var(--text-200) !important;
        }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{
          color: var(--text-100) !important;
        }}

        /* Code blocks (used in tool-call expanders) */
        .stCodeBlock,
        .stCodeBlock pre,
        .stCodeBlock code {{
          background: var(--bg-elevated) !important;
          color: var(--text-100) !important;
        }}

        /* Toggle widget (theme switch itself) */
        .stToggle [data-baseweb="checkbox"] div {{
          color: var(--text-200) !important;
        }}

        /* Dividers */
        hr,
        [data-testid="stDivider"] {{
          border-color: var(--line) !important;
        }}

        /* Alert / info / warning / error messages */
        [data-testid="stAlert"] {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border: 1px solid var(--line) !important;
        }}

        /* Raw JSON tree (st.json) — make it match dark/light theme */
        [data-testid="stJson"],
        [data-testid="stJson"] > div,
        [data-testid="stJson"] pre,
        .stJson,
        .stJson > div,
        .stJson pre {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
          border: 1px solid var(--line) !important;
          border-radius: 0 !important;
        }}
        [data-testid="stJson"] *,
        .stJson * {{
          background: transparent !important;
        }}
        /* JSON syntax-highlighted spans (string, number, boolean, null, key) */
        [data-testid="stJson"] .react-json-view,
        [data-testid="stJson"] .object-key-val,
        [data-testid="stJson"] .object-content,
        [data-testid="stJson"] .icon-container,
        [data-testid="stJson"] .copy-icon,
        [data-testid="stJson"] .pretty-json-container {{
          background: var(--bg-deep) !important;
          color: var(--text-100) !important;
        }}
        [data-testid="stJson"] .object-key,
        [data-testid="stJson"] .variable-row {{
          color: var(--text-200) !important;
        }}
        [data-testid="stJson"] .string-value {{
          color: var(--accent-bright) !important;
        }}
        [data-testid="stJson"] .number-value {{
          color: #C97A2A !important;
        }}
        [data-testid="stJson"] .boolean-value,
        [data-testid="stJson"] .null-value {{
          color: var(--text-300) !important;
          font-style: italic;
        }}
        [data-testid="stJson"] .icon-container,
        [data-testid="stJson"] .icon-container svg {{
          color: var(--text-300) !important;
          fill: var(--text-300) !important;
        }}

        /* Move wordmark above the auto-generated page nav */
        section[data-testid="stSidebar"] > div:first-child > div:first-child {{
          display: flex;
          flex-direction: column;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
          order: 2;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
          order: 1;
        }}
        </style>
        """
    ).strip()
    st.html(css)


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
    st.sidebar.html(html)


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
