"""Basic Chat — multi-provider streaming chat with local tool-use loop."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

import playground.tools.examples  # noqa: F401, E402  -- registers echo, get_current_time
from playground.branding import (
    inject_brand_css,
    render_brand_wordmark,
    render_theme_toggle,
)
from playground.chat_ui import (
    render_message,
    render_tool_call_block,
    stream_assistant_turn,
)
from playground.persistence import ConversationStore
from playground.prompts.loader import list_prompts, load_prompt
from playground.providers.base import (
    ChatMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from playground.providers.config import load_providers_config
from playground.providers.registry import (
    get_client,
    list_available_providers,
    list_models,
)
from playground.tools import call_local_tool, get_local_tools

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


# ---------------- System prompt ----------------

st.sidebar.markdown('<div class="tml-label">System prompt</div>', unsafe_allow_html=True)
prompts_available = ["(none)"] + list_prompts()
prompt_choice = st.sidebar.selectbox(
    "Load from library",
    prompts_available,
    key="prompt_choice",
)
default_text = (
    "" if prompt_choice == "(none)" else load_prompt(prompt_choice)
)
system_prompt = st.sidebar.text_area(
    "System prompt", value=default_text, height=180, key="system_prompt_text",
)


# ---------------- Local tools ----------------

st.sidebar.markdown('<div class="tml-label">Local tools</div>', unsafe_allow_html=True)
local_tool_defs = get_local_tools()
enabled_local: list[str] = st.sidebar.multiselect(
    "Enabled",
    [t.name for t in local_tool_defs],
    default=[t.name for t in local_tool_defs],
    key="enabled_local_tools",
)
active_tools = [t for t in local_tool_defs if t.name in enabled_local]


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
            "system_prompt": {
                "source": prompt_choice if prompt_choice != "(none)" else None,
                "text": system_prompt or "",
            },
            "tools": {"local": [t.name for t in active_tools], "mcp": [], "builtin": []},
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

    MAX_ITERS = 10
    for _ in range(MAX_ITERS):
        with st.chat_message("assistant", avatar="◐"):
            text_box = st.empty()
            text_buf: list[str] = []

            def _on_text(t: str, _buf: list[str] = text_buf, _box: Any = text_box) -> None:
                _buf.append(t)
                _box.markdown("".join(_buf))

            client = get_client(provider, model)
            full_text, tool_calls, final = stream_assistant_turn(
                lambda c=client: c.stream_chat(
                    messages=messages,
                    system=system_prompt or None,
                    tools=active_tools,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                ),
                on_text=_on_text,
            )

            content_blocks: list = []
            if full_text:
                content_blocks.append(TextBlock(type="text", text=full_text))
            for tc in tool_calls:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use", id=tc.id, name=tc.name, input=tc.input,
                        source={"kind": "local"},
                    )
                )
            asst_msg = ChatMessage(role="assistant", content=content_blocks)
            messages.append(asst_msg)
            save_msg = {
                "role": "assistant",
                "ts": _now_iso(),
                "content": [
                    ({"type": "text", "text": b.text} if isinstance(b, TextBlock)
                     else {"type": "tool_use", "id": b.id, "name": b.name,
                           "input": b.input, "source": b.source})
                    for b in content_blocks
                ],
            }
            if final:
                save_msg["usage"] = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cache_read_tokens": final.usage.cache_read_tokens,
                }
            conv.append_message(save_msg)

            if not tool_calls:
                break

            tool_result_blocks = []
            for tc in tool_calls:
                t0 = time.time()
                is_err = False
                try:
                    out = call_local_tool(tc.name, tc.input)
                    out_text = out if isinstance(out, str) else json.dumps(out)
                except Exception as e:
                    out_text = f"{type(e).__name__}: {e}"
                    is_err = True
                duration_ms = int((time.time() - t0) * 1000)
                render_tool_call_block(
                    name=tc.name, source={"kind": "local"}, input=tc.input,
                    result_text=out_text, duration_ms=duration_ms, is_error=is_err,
                )
                tool_result_blocks.append(
                    ToolResultBlock(
                        type="tool_result", tool_use_id=tc.id,
                        content=[{"type": "text", "text": out_text}],
                        is_error=is_err, duration_ms=duration_ms,
                    )
                )

            tr_msg = ChatMessage(role="user", content=tool_result_blocks)
            messages.append(tr_msg)
            conv.append_message({
                "role": "user",
                "ts": _now_iso(),
                "content": [
                    {"type": "tool_result", "tool_use_id": b.tool_use_id,
                     "content": b.content, "is_error": b.is_error,
                     "duration_ms": b.duration_ms}
                    for b in tool_result_blocks
                ],
            })


render_theme_toggle()
