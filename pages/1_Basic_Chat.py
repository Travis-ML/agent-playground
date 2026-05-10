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
from playground.mcp.client import MCPClientPool, MCPTool
from playground.mcp.config import load_mcp_config
from playground.persistence import ConversationStore
from playground.prompts.loader import list_prompts, load_prompt
from playground.providers.base import (
    ChatMessage,
    TextBlock,
    ToolDefinition,
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


# ---------------- MCP servers ----------------

mcp_servers = load_mcp_config()
mcp_tool_defs: list = []
mcp_tools_meta: list[MCPTool] = []
enabled_servers: list[str] = []
pool: MCPClientPool | None = None

if mcp_servers:
    if "mcp_pool" not in st.session_state:
        try:
            new_pool = MCPClientPool()
            new_pool.start(mcp_servers)
            st.session_state.mcp_pool = new_pool
        except Exception as e:
            st.session_state.mcp_pool = None
            st.sidebar.error(f"MCP pool failed to start: {e}")
    pool = st.session_state.get("mcp_pool")

    if pool:
        st.sidebar.markdown('<div class="tml-label">MCP servers</div>', unsafe_allow_html=True)
        for name, cfg in mcp_servers.items():
            label = f"{name} — {cfg.description}" if cfg.description else name
            if st.sidebar.checkbox(label, value=cfg.enabled, key=f"_mcp_{name}"):
                enabled_servers.append(name)

        if st.sidebar.button("Reload mcp.json"):
            pool.shutdown()
            st.session_state.pop("mcp_pool", None)
            st.rerun()

        try:
            mcp_tools_meta = pool.list_tools(enabled_servers)
            mcp_tool_defs = [t.to_tool_definition() for t in mcp_tools_meta]
        except Exception as e:
            st.sidebar.error(f"MCP list_tools failed: {e}")
            mcp_tools_meta = []
            mcp_tool_defs = []

        # Map name → server for dispatch
        st.session_state._mcp_tool_to_server = {t.name: t.server for t in mcp_tools_meta}

# Append MCP tools to the active tool list passed to providers
active_tools = active_tools + mcp_tool_defs


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
            "tools": {
                "local": [t.name for t in local_tool_defs if t.name in enabled_local],
                "mcp": [
                    {"server": s, "tools": [t.name for t in mcp_tools_meta if t.server == s]}
                    for s in enabled_servers
                ],
                "builtin": (
                    ["read_mcp_resource"]
                    if (mcp_servers and pool and enabled_servers) else []
                ),
            },
            "mcp_servers_enabled": list(enabled_servers),
        },
    )
    st.session_state.messages = []
    st.session_state.conv_provider = provider
    st.session_state.conv_model = model

conv = st.session_state.conversation
messages: list[ChatMessage] = st.session_state.messages


# ---------------- MCP prompts ----------------

if mcp_servers and pool and enabled_servers:
    try:
        mcp_prompts = pool.list_prompts(enabled_servers)
    except Exception as e:
        st.sidebar.error(f"MCP list_prompts failed: {e}")
        mcp_prompts = []

    if mcp_prompts:
        st.sidebar.markdown('<div class="tml-label">MCP prompts</div>', unsafe_allow_html=True)
        prompt_options = {f"{p.server}/{p.name}": p for p in mcp_prompts}
        sel = st.sidebar.selectbox(
            "Prompt", list(prompt_options.keys()), key="_mcp_prompt_sel",
        )
        chosen = prompt_options[sel]
        arg_values: dict = {}
        for arg in chosen.arguments:
            arg_values[arg["name"]] = st.sidebar.text_input(
                f"  arg: {arg['name']}",
                value=arg.get("default", ""),
                help=arg.get("description"),
                key=f"_mcp_prompt_arg_{arg['name']}",
            )

        col1, col2 = st.sidebar.columns(2)
        if col1.button("Use as user message"):
            try:
                msgs = pool.get_prompt(chosen.server, chosen.name, arg_values)
            except Exception as e:
                st.sidebar.error(f"get_prompt failed: {e}")
                msgs = []
            for m in msgs:
                st.session_state.messages.append(
                    ChatMessage(
                        role="user" if m["role"] == "user" else "assistant",
                        content=[TextBlock(type="text", text=m["content"][0]["text"])],
                    )
                )
                conv.append_message({**m, "ts": _now_iso()})
                conv.add_event({
                    "ts": _now_iso(),
                    "type": "prompt_inserted",
                    "server": chosen.server,
                    "prompt": chosen.name,
                    "args": arg_values,
                })
            st.rerun()
        if col2.button("Use as system prompt"):
            try:
                msgs = pool.get_prompt(chosen.server, chosen.name, arg_values)
            except Exception as e:
                st.sidebar.error(f"get_prompt failed: {e}")
                msgs = []
            new_text = "\n\n".join(m["content"][0]["text"] for m in msgs)
            st.session_state.system_prompt_text = new_text
            conv.add_event({
                "ts": _now_iso(),
                "type": "system_prompt_replaced_by_mcp",
                "server": chosen.server, "prompt": chosen.name, "args": arg_values,
            })
            st.rerun()


# ---------------- MCP resources ----------------

attached_resources: list[dict[str, str]] = []
if mcp_servers and pool and enabled_servers:
    try:
        resources = pool.list_resources(enabled_servers)
    except Exception as e:
        st.sidebar.error(f"MCP list_resources failed: {e}")
        resources = []

    if resources:
        st.sidebar.markdown('<div class="tml-label">MCP resources</div>', unsafe_allow_html=True)
        for r in resources:
            uri_label = r.uri.split("/")[-1] or r.uri
            label = f"{r.server}/{uri_label}"
            if st.sidebar.checkbox(label, key=f"_mcp_res_{r.server}_{r.uri}"):
                attached_resources.append({
                    "server": r.server, "uri": r.uri, "mime_type": r.mime_type,
                })
        if st.sidebar.button("Refresh resources"):
            st.rerun()


# ---------------- Builtin tools (when MCP active) ----------------

builtin_tools: list[ToolDefinition] = []
if mcp_servers and pool and enabled_servers:
    builtin_tools.append(
        ToolDefinition(
            name="read_mcp_resource",
            description=(
                "Read an MCP resource by URI (use when you see a uri the user "
                "mentioned or one returned by another tool)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name"},
                    "uri": {"type": "string", "description": "Resource URI"},
                },
                "required": ["server", "uri"],
            },
        )
    )

# Append builtin tools to active tools list
active_tools = active_tools + builtin_tools


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
    preamble_blocks: list = []
    for ar in attached_resources:
        try:
            content_text = (
                pool.read_resource(ar["server"], ar["uri"])
                if pool else "[pool unavailable]"
            )
        except Exception as e:
            content_text = f"[failed to read {ar['uri']}: {e}]"
        preamble_blocks.append(
            TextBlock(
                type="text",
                text=(
                    f'<resource uri="{ar["uri"]}" mimeType="{ar["mime_type"]}">\n'
                    f'{content_text}\n</resource>'
                ),
            )
        )
        conv.add_event({
            "ts": _now_iso(),
            "type": "resource_attached",
            "server": ar["server"], "uri": ar["uri"],
        })

    user_msg = ChatMessage(
        role="user",
        content=preamble_blocks + [TextBlock(type="text", text=prompt)],
    )
    messages.append(user_msg)
    conv.append_message({
        "role": "user",
        "ts": _now_iso(),
        "content": [
            {"type": "text", "text": b.text} for b in user_msg.content
        ],
    })
    render_message(user_msg)

    tool_to_server: dict[str, str] = st.session_state.get("_mcp_tool_to_server", {})
    local_names = {t.name for t in get_local_tools()}

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
                if tc.name in local_names:
                    src: dict[str, str] = {"kind": "local"}
                elif tc.name in tool_to_server:
                    src = {"kind": "mcp", "server": tool_to_server[tc.name]}
                elif tc.name == "read_mcp_resource":
                    src = {"kind": "builtin"}
                else:
                    src = {"kind": "unknown"}
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use", id=tc.id, name=tc.name, input=tc.input,
                        source=src,
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
                source: dict[str, str] = {"kind": "local"}
                try:
                    if tc.name in local_names:
                        out = call_local_tool(tc.name, tc.input)
                        out_text = out if isinstance(out, str) else json.dumps(out)
                    elif tc.name in tool_to_server:
                        server = tool_to_server[tc.name]
                        source = {"kind": "mcp", "server": server}
                        if pool is None:
                            raise RuntimeError("MCP pool unavailable")
                        out_text = pool.call_tool(server, tc.name, tc.input)
                    elif tc.name == "read_mcp_resource":
                        server = tc.input.get("server", "")
                        uri = tc.input.get("uri", "")
                        source = {"kind": "builtin"}
                        if pool is None:
                            out_text = "[pool unavailable]"
                            is_err = True
                        else:
                            out_text = pool.read_resource(server, uri)
                    else:
                        out_text = f"Unknown tool: {tc.name}"
                        is_err = True
                except Exception as e:
                    out_text = f"{type(e).__name__}: {e}"
                    is_err = True
                duration_ms = int((time.time() - t0) * 1000)
                render_tool_call_block(
                    name=tc.name, source=source, input=tc.input,
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
