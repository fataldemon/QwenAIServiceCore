"""Runtime configuration manager.

Handles two JSON config files in ``config/``:

* ``providers.json`` -- LLM providers (OpenAI-compatible HTTP endpoints).
* ``mcp_servers.json`` -- MCP servers and tool-call policy.

Both files are loaded eagerly at startup and can be reloaded / mutated through
the admin API. All file operations are atomic (write-then-rename) and guarded
by an in-process :class:`asyncio.Lock` so that concurrent admin operations
remain consistent.

If the JSON file is missing, the matching ``*.example.json`` is copied as the
initial content (so a fresh checkout boots without any manual setup).

Design note: we intentionally keep the schema permissive (extra keys allowed)
because providers may have provider-specific configuration knobs
(e.g. ``extra_body.mm_processor_kwargs`` for vLLM, ``response_format`` for
others). Validation focuses on the keys our code actually depends on.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
PROVIDERS_FILE = os.path.join(CONFIG_DIR, "providers.json")
PROVIDERS_EXAMPLE_FILE = os.path.join(CONFIG_DIR, "providers.example.json")
MCP_FILE = os.path.join(CONFIG_DIR, "mcp_servers.json")
MCP_EXAMPLE_FILE = os.path.join(CONFIG_DIR, "mcp_servers.example.json")


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    type: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)
    request_timeout: float = 600.0
    stream_chunk_timeout: float = 60.0
    prefetch_media: bool = False
    description: str = ""
    # Free-form bag for provider-specific fields we do not interpret directly.
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ProviderConfig":
        known = {
            "type",
            "base_url",
            "api_key",
            "model",
            "supports_vision",
            "supports_audio",
            "supports_video",
            "extra_body",
            "request_timeout",
            "stream_chunk_timeout",
            "prefetch_media",
            "description",
        }
        kwargs = {k: data[k] for k in known if k in data}
        raw = {k: v for k, v in data.items() if k not in known}
        return cls(name=name, raw=raw, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "supports_vision": self.supports_vision,
            "supports_audio": self.supports_audio,
            "supports_video": self.supports_video,
            "extra_body": self.extra_body,
            "request_timeout": self.request_timeout,
            "stream_chunk_timeout": self.stream_chunk_timeout,
            "prefetch_media": self.prefetch_media,
            "description": self.description,
        }
        d.update(self.raw)
        return d


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server entry."""

    name: str
    enabled: bool = False
    transport: str = "stdio"  # one of: stdio / sse / streamable_http
    # stdio
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    # remote (sse / streamable_http)
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=name,
            enabled=bool(data.get("enabled", False)),
            transport=str(data.get("transport", "stdio")),
            command=data.get("command"),
            args=list(data.get("args", []) or []),
            env=dict(data.get("env", {}) or {}),
            url=data.get("url"),
            headers=dict(data.get("headers", {}) or {}),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "enabled": self.enabled,
            "transport": self.transport,
            "description": self.description,
        }
        if self.transport == "stdio":
            d["command"] = self.command
            d["args"] = self.args
            d["env"] = self.env
        else:
            d["url"] = self.url
            d["headers"] = self.headers
        return d


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically: write to a temp file then ``os.replace``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _ensure_file_from_example(path: str, example_path: str) -> None:
    if os.path.exists(path):
        return
    if os.path.exists(example_path):
        shutil.copyfile(example_path, path)


# ----------------------------------------------------------------------------
# ConfigManager
# ----------------------------------------------------------------------------


class ConfigManager:
    """Thread-safe singleton-style configuration manager.

    Use :func:`get_config_manager` to obtain the process-wide instance.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._providers: Dict[str, ProviderConfig] = {}
        self._active_provider: str = ""
        self._active_character: str = ""
        self._mcp_servers: Dict[str, MCPServerConfig] = {}
        self._mcp_tool_call_mode: str = "passthrough"
        self._mcp_tool_call_timeout: float = 30.0
        self._mcp_max_tool_rounds: int = 5
        # Synchronous load so that the manager is ready immediately after
        # construction. We avoid touching the asyncio lock here because
        # construction is expected during single-threaded startup.
        self._reload_providers_sync()
        self._reload_mcp_sync()

    # ----- providers ---------------------------------------------------------

    def _reload_providers_sync(self) -> None:
        _ensure_file_from_example(PROVIDERS_FILE, PROVIDERS_EXAMPLE_FILE)
        if not os.path.exists(PROVIDERS_FILE):
            self._providers = {}
            self._active_provider = ""
            self._active_character = ""
            return
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        providers = data.get("providers", {}) or {}
        self._providers = {
            name: ProviderConfig.from_dict(name, cfg) for name, cfg in providers.items()
        }
        active = data.get("active", "") or ""
        if active and active in self._providers:
            self._active_provider = active
        elif self._providers:
            # Fall back to the first one alphabetically -- deterministic.
            self._active_provider = sorted(self._providers.keys())[0]
        else:
            self._active_provider = ""
        self._active_character = str(data.get("active_character", ""))

    def _dump_providers_sync(self) -> None:
        data = {
            "active": self._active_provider,
            "active_character": self._active_character,
            "providers": {name: p.to_dict() for name, p in self._providers.items()},
        }
        _atomic_write_json(PROVIDERS_FILE, data)

    def list_providers(self) -> List[ProviderConfig]:
        return [deepcopy(p) for p in self._providers.values()]

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        p = self._providers.get(name)
        return deepcopy(p) if p is not None else None

    def get_active_provider(self) -> Optional[ProviderConfig]:
        if not self._active_provider:
            return None
        return self.get_provider(self._active_provider)

    def get_active_provider_name(self) -> str:
        return self._active_provider

    async def upsert_provider(self, name: str, config: Dict[str, Any]) -> ProviderConfig:
        async with self._lock:
            self._providers[name] = ProviderConfig.from_dict(name, config)
            if not self._active_provider:
                self._active_provider = name
            self._dump_providers_sync()
            return deepcopy(self._providers[name])

    async def delete_provider(self, name: str) -> bool:
        async with self._lock:
            if name not in self._providers:
                return False
            del self._providers[name]
            if self._active_provider == name:
                self._active_provider = (
                    sorted(self._providers.keys())[0] if self._providers else ""
                )
            self._dump_providers_sync()
            return True

    async def activate_provider(self, name: str) -> bool:
        async with self._lock:
            if name not in self._providers:
                return False
            self._active_provider = name
            self._dump_providers_sync()
            return True

    # ----- mcp ---------------------------------------------------------------

    def _reload_mcp_sync(self) -> None:
        _ensure_file_from_example(MCP_FILE, MCP_EXAMPLE_FILE)
        if not os.path.exists(MCP_FILE):
            self._mcp_servers = {}
            self._mcp_tool_call_mode = "passthrough"
            self._mcp_tool_call_timeout = 30.0
            self._mcp_max_tool_rounds = 5
            return
        with open(MCP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        servers = data.get("servers", {}) or {}
        self._mcp_servers = {
            name: MCPServerConfig.from_dict(name, cfg) for name, cfg in servers.items()
        }
        self._mcp_tool_call_mode = str(data.get("tool_call_mode", "passthrough"))
        try:
            self._mcp_tool_call_timeout = float(data.get("tool_call_timeout", 30))
        except (TypeError, ValueError):
            self._mcp_tool_call_timeout = 30.0
        try:
            self._mcp_max_tool_rounds = int(data.get("max_tool_rounds", 5))
        except (TypeError, ValueError):
            self._mcp_max_tool_rounds = 5

    def _dump_mcp_sync(self) -> None:
        data = {
            "servers": {name: s.to_dict() for name, s in self._mcp_servers.items()},
            "tool_call_mode": self._mcp_tool_call_mode,
            "tool_call_timeout": self._mcp_tool_call_timeout,
            "max_tool_rounds": self._mcp_max_tool_rounds,
        }
        _atomic_write_json(MCP_FILE, data)

    def list_mcp_servers(self) -> List[MCPServerConfig]:
        return [deepcopy(s) for s in self._mcp_servers.values()]

    def get_mcp_server(self, name: str) -> Optional[MCPServerConfig]:
        s = self._mcp_servers.get(name)
        return deepcopy(s) if s is not None else None

    def get_mcp_tool_call_mode(self) -> str:
        return self._mcp_tool_call_mode

    def get_mcp_tool_call_timeout(self) -> float:
        return self._mcp_tool_call_timeout

    def get_mcp_max_tool_rounds(self) -> int:
        return self._mcp_max_tool_rounds

    async def set_mcp_max_tool_rounds(self, rounds: int) -> None:
        async with self._lock:
            self._mcp_max_tool_rounds = max(1, min(20, int(rounds)))
            self._dump_mcp_sync()

    async def upsert_mcp_server(
        self, name: str, config: Dict[str, Any]
    ) -> MCPServerConfig:
        async with self._lock:
            self._mcp_servers[name] = MCPServerConfig.from_dict(name, config)
            self._dump_mcp_sync()
            return deepcopy(self._mcp_servers[name])

    async def delete_mcp_server(self, name: str) -> bool:
        async with self._lock:
            if name not in self._mcp_servers:
                return False
            del self._mcp_servers[name]
            self._dump_mcp_sync()
            return True

    async def set_mcp_tool_call_mode(self, mode: str) -> None:
        async with self._lock:
            if mode not in ("passthrough", "server_side"):
                raise ValueError(
                    "tool_call_mode must be 'passthrough' or 'server_side'"
                )
            self._mcp_tool_call_mode = mode
            self._dump_mcp_sync()

    # ----- active character --------------------------------------------------

    def get_active_character(self) -> str:
        """Return the name of the currently active character (folder name)."""
        return self._active_character

    async def set_active_character(self, character: str) -> None:
        """Set the active character by folder name (e.g. ``tendou_arisu``).

        The value is stored in ``providers.json`` under an ``active_character``
        key so it survives restarts. Passing an empty string clears the setting.
        """
        async with self._lock:
            self._active_character = character
            # Persist alongside provider data for simplicity.
            data = {
                "active": self._active_provider,
                "active_character": self._active_character,
                "providers": {name: p.to_dict() for name, p in self._providers.items()},
            }
            _atomic_write_json(PROVIDERS_FILE, data)


_singleton: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Return the process-wide :class:`ConfigManager` instance."""

    global _singleton
    if _singleton is None:
        _singleton = ConfigManager()
    return _singleton
