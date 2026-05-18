"""Backend registry.

A tiny indirection that maps the ``type`` field in a :class:`ProviderConfig`
to a concrete :class:`LLMBackend` implementation. Backends are cached per
provider name so HTTP connection pools are reused. When a provider is updated
or removed via the admin API, callers should ``invalidate(name)`` so the next
request rebuilds it.

This is intentionally synchronous: ``ConfigManager`` is loaded eagerly at
startup, so resolving the active provider is just a dict lookup, not an
``await``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from core.config_manager import ProviderConfig, get_config_manager

from .base import LLMBackend
from .openai_compatible import OpenAICompatibleBackend

LOG = logging.getLogger(__name__)

_backends: Dict[str, LLMBackend] = {}
_lock = asyncio.Lock()


def _build_backend(cfg: ProviderConfig) -> LLMBackend:
    t = (cfg.type or "openai_compatible").lower()
    if t == "openai_compatible":
        return OpenAICompatibleBackend(cfg)
    # ``inprocess_vllm`` is intentionally not implemented on the ``dev`` branch
    # -- see docs/MAINTENANCE.md for the rationale (we now rely on
    # ``vllm serve`` as an external process).
    raise ValueError(f"Unsupported backend type: {cfg.type!r}")


def get_backend(provider_name: Optional[str] = None) -> LLMBackend:
    """Get (or lazily build) the backend for ``provider_name``.

    When ``provider_name`` is ``None`` the active provider from
    :class:`ConfigManager` is used.
    """
    cm = get_config_manager()
    if provider_name is None:
        cfg = cm.get_active_provider()
        if cfg is None:
            raise RuntimeError(
                "No active provider configured. Edit config/providers.json "
                "or use the /admin UI to add one."
            )
    else:
        cfg = cm.get_provider(provider_name)
        if cfg is None:
            raise KeyError(f"Unknown provider: {provider_name!r}")

    cached = _backends.get(cfg.name)
    if cached is not None:
        return cached
    backend = _build_backend(cfg)
    _backends[cfg.name] = backend
    return backend


async def invalidate(provider_name: str) -> None:
    """Drop the cached backend for ``provider_name`` and close its client."""
    async with _lock:
        backend = _backends.pop(provider_name, None)
        if backend is None:
            return
        close = getattr(backend, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception as e:  # pragma: no cover -- best-effort cleanup
                LOG.warning("Error while closing backend %s: %r", provider_name, e)


async def invalidate_all() -> None:
    """Drop all cached backends. Useful during graceful shutdown."""
    async with _lock:
        items = list(_backends.items())
        _backends.clear()
    for name, backend in items:
        close = getattr(backend, "aclose", None)
        if close is None:
            continue
        try:
            await close()
        except Exception as e:  # pragma: no cover
            LOG.warning("Error while closing backend %s: %r", name, e)
