"""MCP (Model Context Protocol) client manager.

This module provides:

* A connection pool keyed by server name. Each enabled server is connected on
  demand (lazy connect) and the resulting :class:`ClientSession` is kept
  alive for the lifetime of the process.
* :meth:`MCPManager.list_all_tools` -- aggregate ``tools/list`` across every
  enabled server, returning OpenAI ``tools=[...]`` items with the server name
  prefixed into the function name (``"<server>__<tool>"``) so the LLM can
  unambiguously route calls.
* :meth:`MCPManager.call_tool` -- dispatch a single tool call back to the
  owning server.

The actual ``mcp`` python package is imported lazily so that a checkout
without ``mcp`` installed still boots; the manager simply reports zero
servers in that case (and the admin UI shows a hint).

Per the design note in :mod:`core.config_manager` we keep this manager
single-process. If you ever scale to multiple workers, swap the in-memory
session dict for a per-worker pool -- the rest of the manager interface stays
the same.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from .config_manager import MCPServerConfig, get_config_manager

LOG = logging.getLogger(__name__)


class MCPManager:
    """Process-wide MCP client manager."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # ``sessions`` maps server name -> (session_obj, exit_stack). We keep
        # the exit stack so we can close transports cleanly on shutdown /
        # config update.
        self._sessions: Dict[str, Tuple[Any, AsyncExitStack]] = {}
        # Cached tool list per server: server_name -> list of OpenAI tool dicts.
        self._tools_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ----- connection -------------------------------------------------------

    async def _connect(self, cfg: MCPServerConfig) -> Optional[Any]:
        """Open a session for one server. Returns ``None`` on failure."""
        try:
            from mcp import ClientSession  # type: ignore
            from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore
        except Exception as e:
            LOG.warning("mcp package not available: %r", e)
            return None

        stack = AsyncExitStack()
        try:
            if cfg.transport == "stdio":
                if not cfg.command:
                    LOG.warning("MCP server %s: missing command for stdio", cfg.name)
                    await stack.aclose()
                    return None
                params = StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=dict(cfg.env) if cfg.env else None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif cfg.transport in ("sse", "streamable_http"):
                # Pulled lazily so missing optional deps don't break stdio servers.
                try:
                    if cfg.transport == "sse":
                        from mcp.client.sse import sse_client  # type: ignore
                    else:
                        from mcp.client.streamable_http import streamablehttp_client as sse_client  # type: ignore
                except Exception as e:
                    LOG.warning("MCP %s transport %s unavailable: %r", cfg.name, cfg.transport, e)
                    await stack.aclose()
                    return None
                ctx = sse_client(cfg.url or "", headers=cfg.headers or None)
                if cfg.transport == "streamable_http":
                    read, write, _ = await stack.enter_async_context(ctx)
                else:
                    read, write = await stack.enter_async_context(ctx)
            else:
                LOG.warning("MCP server %s: unsupported transport %s", cfg.name, cfg.transport)
                await stack.aclose()
                return None

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[cfg.name] = (session, stack)
            LOG.info("MCP server %s connected (%s)", cfg.name, cfg.transport)
            return session
        except Exception as e:
            LOG.warning("MCP server %s connect failed: %r", cfg.name, e)
            await stack.aclose()
            return None

    async def _ensure_session(self, name: str) -> Optional[Any]:
        async with self._lock:
            entry = self._sessions.get(name)
            if entry is not None:
                return entry[0]
            cfg = get_config_manager().get_mcp_server(name)
            if cfg is None or not cfg.enabled:
                return None
            return await self._connect(cfg)

    # ----- tools ------------------------------------------------------------

    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """Return aggregated OpenAI tool dicts across every enabled server."""
        out: List[Dict[str, Any]] = []
        for cfg in get_config_manager().list_mcp_servers():
            if not cfg.enabled:
                continue
            session = await self._ensure_session(cfg.name)
            if session is None:
                continue
            try:
                resp = await session.list_tools()
                tools = getattr(resp, "tools", []) or []
                converted: List[Dict[str, Any]] = []
                for t in tools:
                    name = getattr(t, "name", None) or t.get("name", "")  # type: ignore[union-attr]
                    if not name:
                        continue
                    desc = (
                        getattr(t, "description", None)
                        or (t.get("description", "") if isinstance(t, dict) else "")
                    )
                    schema = (
                        getattr(t, "inputSchema", None)
                        or (t.get("inputSchema", {}) if isinstance(t, dict) else {})
                        or {"type": "object"}
                    )
                    converted.append(
                        {
                            "type": "function",
                            "function": {
                                "name": f"{cfg.name}__{name}",
                                "description": desc,
                                "parameters": schema,
                            },
                        }
                    )
                self._tools_cache[cfg.name] = converted
                out.extend(converted)
            except Exception as e:
                LOG.warning("list_tools on %s failed: %r", cfg.name, e)
        return out

    async def call_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatch a tool call. ``qualified_name`` is ``"<server>__<tool>"``."""
        if "__" not in qualified_name:
            raise ValueError(f"Tool name must be qualified <server>__<tool>: {qualified_name!r}")
        server, tool = qualified_name.split("__", 1)
        session = await self._ensure_session(server)
        if session is None:
            raise RuntimeError(f"MCP server not connected: {server!r}")
        timeout = get_config_manager().get_mcp_tool_call_timeout()
        return await asyncio.wait_for(session.call_tool(tool, arguments), timeout=timeout)

    # ----- lifecycle --------------------------------------------------------

    async def invalidate(self, name: str) -> None:
        async with self._lock:
            entry = self._sessions.pop(name, None)
            self._tools_cache.pop(name, None)
        if entry is None:
            return
        _, stack = entry
        try:
            await stack.aclose()
        except Exception as e:  # pragma: no cover
            LOG.warning("Error closing MCP session %s: %r", name, e)

    async def shutdown(self) -> None:
        async with self._lock:
            items = list(self._sessions.items())
            self._sessions.clear()
            self._tools_cache.clear()
        for name, (_session, stack) in items:
            try:
                await stack.aclose()
            except Exception as e:  # pragma: no cover
                LOG.warning("Error closing MCP session %s: %r", name, e)

    async def health(self) -> Dict[str, Dict[str, Any]]:
        """Report connection state for every configured server."""
        out: Dict[str, Dict[str, Any]] = {}
        for cfg in get_config_manager().list_mcp_servers():
            connected = cfg.name in self._sessions
            out[cfg.name] = {
                "enabled": cfg.enabled,
                "transport": cfg.transport,
                "connected": connected,
                "tools": len(self._tools_cache.get(cfg.name, [])),
            }
        return out


_singleton: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _singleton
    if _singleton is None:
        _singleton = MCPManager()
    return _singleton
