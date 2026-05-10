"""Basic Chat — multi-provider streaming chat, no tools yet."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
from dotenv import load_dotenv

from playground.branding import (
    inject_brand_css,
    render_brand_wordmark,
    render_theme_toggle,
)
from playground.chat_ui import render_message, render_text_stream
from playground.persistence import ConversationStore
from playground.providers.base import ChatMessage, MessageComplete, TextBlock
from playground.providers.config import load_providers_config
from playground.providers.registry import (
    get_client,
    list_available_providers,
    list_models,
)

load_dotenv()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


st.set_page_config(
    page_title="Basic Chat — TravisML Playground",
    page_icon="◐",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

inject_brand_css()
render_brand_wordmark()


# ---------------- Sidebar config ----------------

st.sidebar.markdown('<div class="tml-label">Model</div>', unsafe_allow_html=True)

providers_cfg = load_providers_config()
available = list_available_providers(check_lmstudio=False)
if not available:
    st.error("No providers available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")
    st.stop()

provider = st.sidebar.selectbox("Provider", available, key="provider")
pcfg = providers_cfg[provider]
models = list_models(provider, pcfg.models) or [pcfg.default_model or ""]
model = st.sidebar.selectbox(
    "Model",
    models,
    index=models.index(pcfg.default_model) if pcfg.default_model in models else 0,
    key="model",
)

st.sidebar.markdown('<div class="tml-label">Sampling</div>', unsafe_allow_html=True)
max_tokens = st.sidebar.number_input(
    "max_tokens", min_value=1, max_value=128_000,
    value=pcfg.default_max_tokens, key="max_tokens",
)
temperature = st.sidebar.slider(
    "temperature", 0.0, 2.0, pcfg.default_temperature, 0.05, key="temperature",
)


# ---------------- Conversation state ----------------

store = ConversationStore()

if "conversation" not in st.session_state or st.session_state.get("conv_provider") != provider \
        or st.session_state.get("conv_model") != model:
    st.session_state.conversation = store.new(
        "basic_chat",
        config={
            "provider": provider,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system_prompt": {"source": None, "text": ""},
            "tools": {"local": [], "mcp": [], "builtin": []},
            "mcp_servers_enabled": [],
        },
    )
    st.session_state.messages = []
    st.session_state.conv_provider = provider
    st.session_state.conv_model = model

conv = st.session_state.conversation
messages: list[ChatMessage] = st.session_state.messages

# ---------------- Transcript ----------------

st.markdown(
    '<h1 style="font-size:36px;margin-bottom:8px;">Basic <em>chat</em></h1>',
    unsafe_allow_html=True,
)
st.caption(f"Conversation `{conv.id}` · provider: `{provider}/{model}`")
st.divider()

for m in messages:
    render_message(m)


# ---------------- Input + send ----------------

if prompt := st.chat_input("Ask anything..."):
    user_msg = ChatMessage(role="user", content=[TextBlock(type="text", text=prompt)])
    messages.append(user_msg)
    conv.append_message({
        "role": "user",
        "ts": _now_iso(),
        "content": [{"type": "text", "text": prompt}],
    })
    render_message(user_msg)

    with st.chat_message("assistant", avatar="◐"):
        client = get_client(provider, model)
        events = client.stream_chat(
            messages=messages,
            system=None,
            tools=[],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        full_text, last = render_text_stream(events)

    asst_msg = ChatMessage(role="assistant", content=[TextBlock(type="text", text=full_text)])
    messages.append(asst_msg)
    save_msg = {
        "role": "assistant",
        "ts": _now_iso(),
        "content": [{"type": "text", "text": full_text}],
    }
    if isinstance(last, MessageComplete):
        save_msg["usage"] = {
            "input_tokens": last.usage.input_tokens,
            "output_tokens": last.usage.output_tokens,
            "cache_read_tokens": last.usage.cache_read_tokens,
        }
    conv.append_message(save_msg)


render_theme_toggle()
