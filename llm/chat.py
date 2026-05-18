"""High-level chat orchestration.

This module exposes the two entry points the FastAPI layer used to call on
the legacy in-process branch:

* :func:`chat`            -- analysis-style completions (mirrors the old
  ``/assistant/v1/chat/completions`` route).
* :func:`chat_on_setting` -- character-aware completion with embedding
  augmentation (mirrors ``/v1/chat/completions`` + WebSocket).

Both functions now:

1. Pull the active :class:`LLMBackend` from :mod:`llm.backends.registry`.
2. Run the unified content normalizer so that legacy ``[image,file=...]``
   placeholders and OpenAI-style content arrays produce equivalent prompts.
3. Inject knowledge-base retrieval (``setting`` + ``knowledge``) into the
   system prompt, exactly as the legacy code did.
4. Surface MCP-discovered tools when ``mcp_tool_call_mode == "server_side"``.
5. Support both non-streaming and streaming completion; the streaming variant
   is exposed via :func:`chat_on_setting_stream` returning an async iterator
   of ``ChatCompletionResponse(chunk)`` objects.

The signatures of :func:`chat` and :func:`chat_on_setting` were widened so
the FastAPI layer can keep the same handler body. The ``engine`` /
``autoProcessor`` / ``active_lora_path`` parameters from the legacy signature
are accepted and ignored -- this keeps a smaller diff in ``main.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from core.config_manager import get_config_manager
from core.content_normalizer import (
    expand_gif_parts,
    has_media,
    normalize_content,
    to_openai_content,
)
from embedding.embedding import (
    add_knowledge,
    check_emotion,
    find_material_by_index,
    process_embedding,
    remove_reference_url,
)
from models.base import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaMessage,
)
from template import REPLY_INSTRUCTION, SETTING

from .backends import get_backend
from .backends.base import GenerationResult, StreamChunk

LOG = logging.getLogger(__name__)

# In-flight requests, keyed by ``abort_id`` (set by the client). Used by
# :func:`abort_request` to flip the cooperative abort flag on the backend.
_active_requests: Dict[str, Tuple[str, str]] = {}  # abort_id -> (provider_name, request_id)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sampling_from_request(req: ChatCompletionRequest, max_tokens: int) -> Dict[str, Any]:
    s: Dict[str, Any] = {"max_tokens": max_tokens}
    for k in ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty"):
        v = getattr(req, k, None)
        if v is not None:
            s[k] = v
    if req.stop:
        s["stop"] = req.stop
    return s


# ---------------------------------------------------------------------------
# Message preparation
# ---------------------------------------------------------------------------


def _provider_supports_media(provider_cfg) -> Tuple[bool, bool, bool]:
    return (
        bool(provider_cfg.supports_vision),
        bool(provider_cfg.supports_audio),
        bool(provider_cfg.supports_video),
    )


def _prepare_messages(
    raw_messages: List[ChatMessage],
    *,
    provider_cfg,
    system_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Turn the request's pydantic messages into upstream payload dicts.

    ``system_prefix``, if non-empty, is prepended as a system message at index
    0. Any incoming ``role="system"`` message is concatenated to it.
    """
    supports_vision, supports_audio, supports_video = _provider_supports_media(provider_cfg)
    prefetch = bool(provider_cfg.prefetch_media)

    system_parts: List[str] = []
    if system_prefix:
        system_parts.append(system_prefix)
    converted: List[Dict[str, Any]] = []
    for m in raw_messages:
        if m.role == "system":
            # Merge into the system prefix so we never send duplicate system
            # messages (some providers reject that).
            if isinstance(m.content, str):
                system_parts.append(m.content)
            continue

        parts = normalize_content(m.content)
        parts = expand_gif_parts(parts)
        # Filter unsupported modalities so we never silently get a 400 from a
        # text-only provider just because the client sent an image.
        filtered = []
        for p in parts:
            if p.kind == "image" and not supports_vision:
                filtered.append(p.__class__(kind="text", text="[图片已省略]"))
            elif p.kind == "audio" and not supports_audio:
                filtered.append(p.__class__(kind="text", text="[音频已省略]"))
            elif p.kind == "video" and not supports_video:
                filtered.append(p.__class__(kind="text", text="[视频已省略]"))
            else:
                filtered.append(p)
        content_payload = to_openai_content(filtered, prefetch_files=prefetch)
        msg: Dict[str, Any] = {"role": m.role, "content": content_payload}
        if m.function_call:
            msg["function_call"] = m.function_call
        converted.append(msg)

    out: List[Dict[str, Any]] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend(converted)
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


_EMOTION_RE = re.compile(r"【([^】]+)】")


def _split_thought_and_answer(text: str) -> Tuple[str, str]:
    """Split a Qwen3-style ``<think>...</think>...`` payload."""
    if "<think>" in text and "</think>" in text:
        m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return "", text


def _postprocess_answer(answer: str, character: str) -> str:
    """Mirror the legacy reply pipeline: snap emotion + strip control tokens."""
    answer = answer.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    if not answer:
        return answer
    m = _EMOTION_RE.search(answer)
    if m:
        snapped = check_emotion(f"【{m.group(1)}】", character)
        answer = answer.replace(m.group(0), snapped, 1)
    return answer


# ---------------------------------------------------------------------------
# Tool gathering
# ---------------------------------------------------------------------------


async def _gather_tools(req: ChatCompletionRequest) -> Optional[List[Dict[str, Any]]]:
    """Combine request ``functions`` + MCP server tools (if server-side mode)."""
    tools: List[Dict[str, Any]] = []
    if req.functions:
        # Legacy Qwen format -- forward as-is; the backend normalizes shape.
        tools.extend(req.functions)
    cm = get_config_manager()
    if cm.get_mcp_tool_call_mode() == "server_side":
        try:
            from core.mcp_manager import get_mcp_manager

            mm = get_mcp_manager()
            mcp_tools = await mm.list_all_tools()
            tools.extend(mcp_tools)
        except Exception as e:
            LOG.warning("Failed to gather MCP tools: %r", e)
    return tools or None


# ---------------------------------------------------------------------------
# Public: non-streaming chat()
# ---------------------------------------------------------------------------


async def chat(
    *,
    request: ChatCompletionRequest,
    max_tokens: int,
    engine: Any = None,  # legacy: ignored
    autoProcessor: Any = None,  # legacy: ignored
) -> ChatCompletionResponseChoice:
    """Analysis-style completion: no embedding augmentation, no character framing."""
    backend = get_backend()
    provider_cfg = backend.config  # type: ignore[attr-defined]
    messages = _prepare_messages(request.messages, provider_cfg=provider_cfg)
    tools = await _gather_tools(request)

    request_id = request.request_id or str(uuid.uuid4())
    if request.abort_id:
        _active_requests[request.abort_id] = (provider_cfg.name, request_id)

    try:
        result = await backend.generate(
            messages=messages,
            sampling=_sampling_from_request(request, max_tokens),
            tools=tools,
            request_id=request_id,
        )
    finally:
        if request.abort_id:
            _active_requests.pop(request.abort_id, None)

    thought, answer = _split_thought_and_answer(result.text)
    if result.reasoning and not thought:
        thought = result.reasoning
    return ChatCompletionResponseChoice(
        index=0,
        thought=thought,
        embedding_list=[],
        message=ChatMessage(role="assistant", content=answer),
        finish_reason=_map_finish_reason(result.finish_reason),
    )


# ---------------------------------------------------------------------------
# Public: chat_on_setting()
# ---------------------------------------------------------------------------


def _last_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            parts = normalize_content(m.content)
            return "".join(p.text or "" for p in parts if p.kind == "text").strip()
    return ""


async def chat_on_setting(
    *,
    request: ChatCompletionRequest,
    max_tokens: int,
    index: int,
    engine: Any = None,
    autoProcessor: Any = None,
    active_lora_path: str = "",
) -> ChatCompletionResponseChoice:
    """Character chat with embedding-based knowledge augmentation."""
    backend = get_backend()
    provider_cfg = backend.config  # type: ignore[attr-defined]

    # Special "type" branches (1 = summarise to memory, 2 = long-term memory).
    rtype = request.type or 0
    if rtype == 1:
        # Treat the last assistant message as the new knowledge to remember.
        for m in request.messages:
            if m.role == "assistant" and isinstance(m.content, str):
                add_knowledge(m.content, request.character or "")
        return ChatCompletionResponseChoice(
            index=index,
            thought="",
            embedding_list=[],
            message=ChatMessage(role="assistant", content="ok"),
            finish_reason="stop",
        )

    # Build embedding-augmented system prompt.
    user_text = _last_user_text(request.messages)
    embeddings_text = ""
    embedding_index_list: List[int] = list(request.embeddings_buffer or [])
    if request.on_embedding and user_text:
        try:
            embeddings_text, embedding_index_list = process_embedding(
                content=remove_reference_url(user_text),
                top_k=5,
                character=request.character or "",
                client_buffer=embedding_index_list,
                max_length=8,
                client_information=request.information or "",
            )
        except Exception as e:
            LOG.warning("process_embedding failed: %r", e)
            embeddings_text = ""
    system_prefix = SETTING.format(embeddings=embeddings_text) + REPLY_INSTRUCTION

    messages = _prepare_messages(
        request.messages, provider_cfg=provider_cfg, system_prefix=system_prefix
    )
    tools = await _gather_tools(request)

    request_id = request.request_id or str(uuid.uuid4())
    if request.abort_id:
        _active_requests[request.abort_id] = (provider_cfg.name, request_id)

    try:
        result = await backend.generate(
            messages=messages,
            sampling=_sampling_from_request(request, max_tokens),
            tools=tools,
            request_id=request_id,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)}}
            if request.enable_thinking is not None
            else None,
        )
    finally:
        if request.abort_id:
            _active_requests.pop(request.abort_id, None)

    thought, answer = _split_thought_and_answer(result.text)
    if result.reasoning and not thought:
        thought = result.reasoning
    answer = _postprocess_answer(answer, request.character or "")
    return ChatCompletionResponseChoice(
        index=index,
        thought=thought,
        embedding_list=embedding_index_list,
        message=ChatMessage(role="assistant", content=answer),
        finish_reason=_map_finish_reason(result.finish_reason),
    )


# ---------------------------------------------------------------------------
# Streaming variants
# ---------------------------------------------------------------------------


async def chat_on_setting_stream(
    *,
    request: ChatCompletionRequest,
    max_tokens: int,
    index: int,
) -> AsyncIterator[ChatCompletionResponse]:
    """Yield :class:`ChatCompletionResponse` chunks (``object="chat.completion.chunk"``).

    The first chunk carries ``delta.role="assistant"``; subsequent chunks
    carry incremental text; the terminal chunk carries ``finish_reason``.
    """
    backend = get_backend()
    provider_cfg = backend.config  # type: ignore[attr-defined]

    user_text = _last_user_text(request.messages)
    embeddings_text = ""
    embedding_index_list: List[int] = list(request.embeddings_buffer or [])
    if request.on_embedding and user_text:
        try:
            embeddings_text, embedding_index_list = process_embedding(
                content=remove_reference_url(user_text),
                top_k=5,
                character=request.character or "",
                client_buffer=embedding_index_list,
                max_length=8,
                client_information=request.information or "",
            )
        except Exception as e:
            LOG.warning("process_embedding failed: %r", e)

    system_prefix = SETTING.format(embeddings=embeddings_text) + REPLY_INSTRUCTION
    messages = _prepare_messages(
        request.messages, provider_cfg=provider_cfg, system_prefix=system_prefix
    )
    tools = await _gather_tools(request)
    request_id = request.request_id or str(uuid.uuid4())
    if request.abort_id:
        _active_requests[request.abort_id] = (provider_cfg.name, request_id)

    # Emit the opening role chunk.
    yield ChatCompletionResponse(
        model=request.model,
        object="chat.completion.chunk",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )

    try:
        it = await backend.generate_stream(
            messages=messages,
            sampling=_sampling_from_request(request, max_tokens),
            tools=tools,
            request_id=request_id,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)}}
            if request.enable_thinking is not None
            else None,
        )
        async for chunk in it:  # type: StreamChunk
            if not chunk.text and not chunk.finish_reason:
                continue
            yield ChatCompletionResponse(
                model=request.model,
                object="chat.completion.chunk",
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(content=chunk.text or None),
                        finish_reason=(
                            "length" if chunk.finish_reason == "length"
                            else ("stop" if chunk.finish_reason else None)
                        ),
                    )
                ],
            )
    finally:
        if request.abort_id:
            _active_requests.pop(request.abort_id, None)


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------


async def abort_request(abort_id: str) -> bool:
    """Cooperatively abort the in-flight request with ``abort_id``."""
    entry = _active_requests.get(abort_id)
    if entry is None:
        return False
    provider_name, request_id = entry
    backend = get_backend(provider_name)
    try:
        await backend.abort(request_id)
        return True
    except Exception as e:
        LOG.warning("abort failed for %s: %r", abort_id, e)
        return False


def _map_finish_reason(reason: str) -> str:
    if reason in ("stop", "length", "function_call", "abort", "error"):
        return reason
    if reason == "tool_calls":
        return "function_call"
    if reason == "":
        return "stop"
    return "stop"
