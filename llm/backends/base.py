"""Abstract base for LLM backends.

A backend converts a list of OpenAI-style chat messages plus sampling
parameters into a text completion, and optionally streams tokens. It is
deliberately **stateless across requests**: any per-request context (such as
the request id, used by ``abort``) is passed explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class StreamChunk:
    """A single streamed delta.

    ``text`` is the *incremental* text since the previous chunk. ``reasoning``
    carries the chain-of-thought delta when the provider exposes it (e.g.
    ``reasoning_content`` from DeepSeek / DashScope reasoning models, or
    Qwen3's ``<think>`` content) -- empty otherwise.

    ``function_calls`` is the (possibly partial) list of tool calls assembled
    so far; only emitted when the backend can produce them. ``finish_reason``
    is non-empty only on the terminal chunk.
    """

    text: str = ""
    reasoning: str = ""
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None
    """The upstream's raw JSON data object for this chunk (one per SSE frame).
    Forwarded to the log and Request Monitor."""


@dataclass
class GenerationResult:
    """The aggregated outcome of a generation call."""

    text: str = ""
    reasoning: str = ""
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None
    raw_events: List[Dict[str, Any]] = field(default_factory=list)
    """Complete list of raw upstream JSON data objects for the full generation.
    Populated by backends during ``generate()`` and logged to the vLLM
    request log (``logs/vllm_request_log.jsonl``)."""


class LLMBackend(ABC):
    """Common interface for every backend implementation."""

    @abstractmethod
    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Run a non-streaming completion."""

    @abstractmethod
    async def generate_stream(
        self,
        *,
        messages: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run a streaming completion.

        Returns an async iterator of :class:`StreamChunk`. The reason this is
        declared ``async def`` (rather than a sync method returning an async
        iterator) is to give implementations a place to do any one-time setup
        before the first chunk is produced -- e.g. registering an abort
        handle, opening an HTTP stream, ``await``-ing rate limiters.
        Callers iterate as::

            it = await backend.generate_stream(...)
            async for chunk in it:
                ...
        """

    async def abort(self, request_id: str) -> None:
        """Best-effort abort. Default implementation is a no-op.

        For HTTP backends this typically cancels the in-flight stream by
        closing the response; for in-process engines it talks to the engine
        directly. Concrete backends override this when they can do better.
        """
