"""OpenAI-compatible HTTP backend.

This backend talks to any service that implements the OpenAI ``/v1/chat/
completions`` shape -- ``vllm serve``, DashScope's OpenAI-compatible mode,
DeepSeek, OpenAI proper, SiliconFlow, etc.

Streaming is exposed as an :class:`AsyncIterator` of :class:`StreamChunk`.
We parse SSE incrementally and surface:

* ``delta.content`` (the standard token stream),
* ``delta.reasoning_content`` (DeepSeek / DashScope reasoning),
* ``delta.tool_calls`` (assembled across deltas using the OpenAI index/id rules),
* ``usage`` block on the terminal chunk when the upstream supports it.

The backend also implements *cooperative abort*: registering an ``asyncio.Event``
for a given ``request_id`` while the stream is running, so that
:meth:`abort` can flip the event and the iterator stops on the next chunk.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ..backends.base import GenerationResult, LLMBackend, StreamChunk
from core.config_manager import ProviderConfig


class OpenAICompatibleBackend(LLMBackend):
    """A backend that talks OpenAI-compatible HTTP/SSE."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=config.stream_chunk_timeout or 60.0,
                write=10.0,
                pool=None,
            ),
        )
        # Active streams keyed by request_id -- used by ``abort``.
        self._abort_events: Dict[str, asyncio.Event] = {}

    # ----- helpers -----------------------------------------------------------

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _build_url(self) -> str:
        base = self._config.base_url.rstrip("/")
        # The OpenAI spec uses ``/chat/completions``; the base_url is expected
        # to already contain the ``/v1`` suffix per provider convention.
        return f"{base}/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _build_payload(
        self,
        *,
        messages: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": stream,
        }
        if stream:
            # vLLM and others include usage in the final chunk when asked.
            payload["stream_options"] = {"include_usage": True}
        # Sampling knobs -- only forward those that are not ``None``.
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
        ):
            v = sampling.get(k)
            if v is not None:
                payload[k] = v
        if tools:
            # vLLM's ``functions`` schema is the legacy one; modern OpenAI uses
            # ``tools=[{"type":"function","function":...}]``. To maximise
            # compatibility we forward BOTH when present.
            normalized_tools: List[Dict[str, Any]] = []
            for t in tools:
                if isinstance(t, dict) and t.get("type") == "function":
                    normalized_tools.append(t)
                elif isinstance(t, dict) and "name" in t:
                    normalized_tools.append({"type": "function", "function": t})
            if normalized_tools:
                payload["tools"] = normalized_tools

        # Merge provider-level extra body and per-request extra body, the
        # per-request one wins on key conflicts.
        merged_extra: Dict[str, Any] = {}
        merged_extra.update(self._config.extra_body or {})
        if extra_body:
            merged_extra.update(extra_body)
        for k, v in merged_extra.items():
            payload.setdefault(k, v)
        return payload

    # ----- non-streaming -----------------------------------------------------

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Run the upstream as a *stream* internally and aggregate.

        The legacy in-process vLLM backend exposed a non-streaming HTTP
        contract whose body could nevertheless be **interrupted mid-flight**
        — its implementation iterated over an internal generator and the
        caller (``engine.abort(...)``) would terminate that iterator. The
        client then received a normal :class:`ChatCompletionResponse` whose
        ``content`` was the partial text and whose ``finish_reason`` was
        ``"abort"``.

        To keep that contract working unchanged, we deliberately do **not**
        issue a one-shot ``stream=false`` POST here. Instead we open an SSE
        stream and accumulate the chunks until either the upstream sends
        ``[DONE]`` or someone flips our abort event via :meth:`abort`.
        On abort, whatever text has been produced so far is returned with
        ``finish_reason="abort"`` — exactly the legacy shape.
        """
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        last_raw: Optional[Dict[str, Any]] = None

        it = await self.generate_stream(
            messages=messages,
            sampling=sampling,
            tools=tools,
            request_id=request_id,
            extra_body=extra_body,
        )
        async for chunk in it:
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            # generate_stream emits the *cumulative* function_calls list on
            # every tool-call delta, so the latest non-empty list wins.
            if chunk.function_calls:
                function_calls = chunk.function_calls
            if chunk.raw is not None:
                last_raw = chunk.raw
                usage = (chunk.raw or {}).get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
                # ``error`` / ``abort`` are terminal, but we still want to
                # return whatever text we have so the front-end sees the
                # partial answer just like the legacy implementation did.
                break

        return GenerationResult(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            function_calls=function_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=last_raw,
        )

    # ----- streaming ---------------------------------------------------------

    async def generate_stream(
        self,
        *,
        messages: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(
            messages=messages,
            sampling=sampling,
            tools=tools,
            stream=True,
            extra_body=extra_body,
        )
        abort_event = asyncio.Event()
        if request_id:
            self._abort_events[request_id] = abort_event

        async def _iterator() -> AsyncIterator[StreamChunk]:
            try:
                async with self._client.stream(
                    "POST",
                    self._build_url(),
                    headers={**self._build_headers(), "Accept": "text/event-stream"},
                    json=payload,
                    timeout=self._config.request_timeout or 600.0,
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        yield StreamChunk(
                            text=f"[backend http {resp.status_code}] {body[:500]}",
                            finish_reason="error",
                        )
                        return

                    # Accumulator state for tool calls (OpenAI streams them as
                    # incremental fragments keyed by ``index``).
                    tool_buf: Dict[int, Dict[str, Any]] = {}
                    final_finish_reason = ""
                    last_chunk_time = time.monotonic()
                    chunk_timeout = self._config.stream_chunk_timeout or 60.0

                    async for raw_line in resp.aiter_lines():
                        if abort_event.is_set():
                            yield StreamChunk(finish_reason="abort")
                            return
                        if not raw_line:
                            # Heartbeat / inter-event blank line.
                            if time.monotonic() - last_chunk_time > chunk_timeout:
                                yield StreamChunk(
                                    text="[stream timeout]",
                                    finish_reason="error",
                                )
                                return
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        last_chunk_time = time.monotonic()
                        choices = event.get("choices") or []
                        if not choices:
                            # ``usage``-only frame from include_usage. Emit it
                            # as a no-op chunk so the caller can stash usage.
                            yield StreamChunk(raw=event)
                            continue
                        ch = choices[0]
                        delta = ch.get("delta") or {}
                        text_delta = str(delta.get("content") or "")
                        reasoning_delta = str(delta.get("reasoning_content") or "")
                        # Tool-call assembly
                        for tc in delta.get("tool_calls", []) or []:
                            idx = int(tc.get("index", 0))
                            slot = tool_buf.setdefault(
                                idx,
                                {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                        finish_reason = ch.get("finish_reason") or ""
                        if finish_reason:
                            final_finish_reason = finish_reason
                        yield StreamChunk(
                            text=text_delta,
                            reasoning=reasoning_delta,
                            function_calls=_tool_calls_to_function_calls(
                                list(tool_buf.values())
                            )
                            if tool_buf
                            else [],
                            finish_reason=finish_reason,
                            raw=event,
                        )
                    if not final_finish_reason:
                        # Stream ended without a finish_reason -- emit one.
                        yield StreamChunk(finish_reason="stop")
            except httpx.HTTPError as e:
                yield StreamChunk(text=f"[backend error] {e!r}", finish_reason="error")
            finally:
                if request_id:
                    self._abort_events.pop(request_id, None)

        return _iterator()

    # ----- lifecycle ---------------------------------------------------------

    async def abort(self, request_id: str) -> None:
        ev = self._abort_events.get(request_id)
        if ev is not None:
            ev.set()

    async def aclose(self) -> None:
        with suppress(Exception):
            await self._client.aclose()


def _tool_calls_to_function_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI ``tool_calls`` items into our internal function_call dicts.

    The internal shape used by the rest of the service mirrors the legacy
    Qwen function-call format: ``{"name": str, "arguments": str_or_dict, "id": str}``.
    Arguments are kept as the raw string the model emitted, since callers may
    want to parse them themselves (or surface them to the front-end as-is).
    """
    out: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        out.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            }
        )
    return out
