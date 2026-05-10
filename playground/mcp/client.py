"""Manage connections to MCP servers — sync façade over async stdio clients."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from playground.mcp.config import MCPServerConfig
from playground.providers.base import ToolDefinition


@dataclass
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


@dataclass
class MCPPrompt:
    server: str
    name: str
    description: str
    arguments: list[dict[str, Any]]


@dataclass
class MCPResource:
    server: str
    uri: str
    name: str
    description: str
    mime_type: str


class MCPClientPool:
    """One async loop in a background thread; sync façade for Streamlit."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._stack: AsyncExitStack | None = None
        self._configs: dict[str, MCPServerConfig] = {}

    # ---------- lifecycle ----------

    def start(self, servers: dict[str, MCPServerConfig]) -> None:
        if self._loop is not None:
            return
        self._configs = {n: c for n, c in servers.items() if c.enabled}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-pool",
        )
        self._thread.start()
        self._submit(self._open_all()).result(timeout=30)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        try:
            self._submit(self._close_all()).result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._sessions.clear()

    async def _open_all(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cfg in self._configs.items():
            params = StdioServerParameters(
                command=cfg.command, args=list(cfg.args), env={**cfg.env} or None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session

    async def _close_all(self) -> None:
        if self._stack:
            await self._stack.__aexit__(None, None, None)
            self._stack = None

    # ---------- queries ----------

    def list_tools(self, servers: list[str]) -> list[MCPTool]:
        async def _go() -> list[MCPTool]:
            out: list[MCPTool] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                resp = await self._sessions[name].list_tools()
                for t in resp.tools:
                    out.append(
                        MCPTool(
                            server=name, name=t.name,
                            description=t.description or "",
                            input_schema=t.inputSchema,
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    def list_prompts(self, servers: list[str]) -> list[MCPPrompt]:
        async def _go() -> list[MCPPrompt]:
            out: list[MCPPrompt] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                try:
                    resp = await self._sessions[name].list_prompts()
                except Exception:
                    continue
                for p in resp.prompts:
                    out.append(
                        MCPPrompt(
                            server=name, name=p.name,
                            description=p.description or "",
                            arguments=[a.model_dump() if hasattr(a, "model_dump") else dict(a)
                                       for a in (p.arguments or [])],
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    def list_resources(self, servers: list[str]) -> list[MCPResource]:
        async def _go() -> list[MCPResource]:
            out: list[MCPResource] = []
            for name in servers:
                if name not in self._sessions:
                    continue
                try:
                    resp = await self._sessions[name].list_resources()
                except Exception:
                    continue
                for r in resp.resources:
                    out.append(
                        MCPResource(
                            server=name, uri=str(r.uri),
                            name=r.name or str(r.uri),
                            description=r.description or "",
                            mime_type=r.mimeType or "",
                        )
                    )
            return out
        return self._submit(_go()).result(timeout=10)

    # ---------- actions ----------

    def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        async def _go() -> str:
            resp = await self._sessions[server].call_tool(tool, args)
            parts: list[str] = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return self._submit(_go()).result(timeout=60)

    def get_prompt(self, server: str, prompt: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        async def _go() -> list[dict[str, Any]]:
            resp = await self._sessions[server].get_prompt(prompt, args)
            out: list[dict[str, Any]] = []
            for m in resp.messages:
                role = m.role
                content = m.content
                text = content.text if hasattr(content, "text") else ""
                out.append({"role": role, "content": [{"type": "text", "text": text}]})
            return out
        return self._submit(_go()).result(timeout=30)

    def read_resource(self, server: str, uri: str) -> str:
        async def _go() -> str:
            resp = await self._sessions[server].read_resource(uri)
            parts: list[str] = []
            for c in resp.contents:
                if hasattr(c, "text"):
                    parts.append(c.text or "")
                elif hasattr(c, "blob"):
                    parts.append(f"[binary content, {len(c.blob)} bytes]")
            return "\n".join(parts)
        return self._submit(_go()).result(timeout=30)

    # ---------- helpers ----------

    def _submit(self, coro) -> Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
