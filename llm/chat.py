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
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from core.config_manager import get_config_manager
from core.content_normalizer import (
    has_media,
    normalize_content,
    to_openai_content,
)
from core.persona_manager import Persona, get_persona_manager
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
# NOTE: Per-character SETTING / REPLY_INSTRUCTION used to live in
# ``template.py`` as hard-coded strings tied to the "天童爱丽丝" persona.
# They now come from ``embedding/<character>/persona.json`` via
# :mod:`core.persona_manager`. ``template.py`` no longer exports those.

from .backends import get_backend
from .backends.base import GenerationResult, StreamChunk

LOG = logging.getLogger(__name__)

# In-flight requests, keyed by ``abort_id`` (set by the client). Used by
# :func:`abort_request` to flip the cooperative abort flag on the backend.
_active_requests: Dict[str, Tuple[str, str]] = {}  # abort_id -> (provider_name, request_id)

# Chat log file path (rotated on restart; max ~10 MB before truncation).
_CHAT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_CHAT_LOG_FILE = os.path.join(_CHAT_LOG_DIR, "chat_log.jsonl")
_CHAT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_VLLM_REQUEST_LOG_FILE = os.path.join(_CHAT_LOG_DIR, "vllm_request_log.jsonl")


def _append_chat_log(entry: Dict[str, Any]) -> None:
    """Append one JSON line to the chat log file.

    The file is truncated to roughly ``_CHAT_LOG_MAX_BYTES`` when it grows
    beyond that limit (simple truncation -- we drop the oldest entries).
    """
    try:
        os.makedirs(_CHAT_LOG_DIR, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        # Check size and truncate if needed before appending.
        if os.path.isfile(_CHAT_LOG_FILE):
            try:
                size = os.path.getsize(_CHAT_LOG_FILE)
                if size > _CHAT_LOG_MAX_BYTES:
                    # Keep roughly the last 75% of the file.
                    with open(_CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    keep = int(len(lines) * 0.75)
                    with open(_CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                        f.writelines(lines[-keep:])
            except OSError:
                pass
        with open(_CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        LOG.debug("Failed to write chat log: %r", e)


def _append_vllm_request_log(entry: Dict[str, Any]) -> None:
    """Append one JSON line to the vLLM request log file."""
    try:
        os.makedirs(_CHAT_LOG_DIR, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(_VLLM_REQUEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        LOG.debug("Failed to write vLLM request log: %r", e)


def truncate_vllm_request_log() -> None:
    """Truncate the vLLM request log file to zero bytes."""
    try:
        os.makedirs(_CHAT_LOG_DIR, exist_ok=True)
        with open(_VLLM_REQUEST_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        LOG.debug("Failed to truncate vLLM request log: %r", e)


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

    Legacy ``role="function"`` messages are converted to ``role="tool"`` with
    content wrapped in ``<tool_response>...</tool_response>`` tags, matching
    the Qwen3.6 chat template expectations. Legacy ``function_call`` on
    assistant messages is converted to the ``tool_calls`` format.
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

        # --- Legacy function-call support (Qwen3.6 template uses ``tool`` role) ---
        if m.role == "function":
            # Convert legacy ``function`` role to ``tool`` role with
            # ``<tool_response>`` wrapper expected by Qwen3.6 chat template.
            text_content = ""
            if isinstance(m.content, str):
                text_content = m.content
            elif isinstance(m.content, list):
                text_parts = []
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                text_content = " ".join(text_parts)
            converted.append({
                "role": "tool",
                "content": f"<tool_response>\n{text_content}\n</tool_response>",
            })
            continue

        parts = normalize_content(m.content)

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
        # Convert legacy ``function_call`` to ``tool_calls`` (Qwen3.6 format).
        if m.function_call:
            if "tool_calls" not in msg:
                msg["tool_calls"] = []
            msg["tool_calls"].append({
                "type": "function",
                "function": {
                    "name": m.function_call.get("name", ""),
                    "arguments": m.function_call.get("arguments", ""),
                },
            })
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        converted.append(msg)

    out: List[Dict[str, Any]] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend(converted)
    return out


# ---------------------------------------------------------------------------
# Persona-driven system prompt
# ---------------------------------------------------------------------------


def _build_persona_system_prefix(character: str, embeddings_text: str) -> str:
    """Build the system prompt prefix from the character's ``persona.json``.

    If the character has no persona on disk we return an empty string -- the
    request becomes a generic completion without any character framing.
    This is intentional: the legacy hard-coded Alice prompt is gone, every
    persona is now data the operator can edit at runtime.

    ``setting`` may contain a ``{embeddings}`` placeholder; if it does, we
    fill it with the retrieved knowledge text. If it doesn't, the retrieved
    knowledge is appended verbatim at the end of the prefix.
    """
    persona = get_persona_manager().get_persona(character)
    if persona is None:
        return ""
    setting = persona.setting or ""
    if setting:
        if "{embeddings}" in setting:
            try:
                setting = setting.format(embeddings=embeddings_text)
            except (KeyError, IndexError):
                # Malformed format spec -- fall back to literal + suffix so
                # the user at least gets *something* useful.
                setting = setting + "\n" + (embeddings_text or "")
        elif embeddings_text:
            setting = setting + "\n" + embeddings_text
    elif embeddings_text:
        setting = embeddings_text
    return setting + (persona.reply_instruction or "")


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
    # Qwen chat_template injects <think>, model outputs only </think>
    if "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[0].strip(), parts[1].strip()
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


async def _execute_mcp_tool(
    mm,  # MCPManager
    tool_name: str,
    arguments: str,
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Execute an MCP tool and append tool_call + tool_result to messages.

    Returns a log entry dict if successful, None on failure.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
    except json.JSONDecodeError:
        args = {}
    try:
        tool_result = await mm.call_tool(tool_name, args)
    except Exception as e:
        LOG.warning("MCP tool %s failed: %r", tool_name, e)
        tool_result = f"Error: {e!r}"
    result_str = json.dumps(tool_result, ensure_ascii=False) if not isinstance(tool_result, str) else tool_result
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": tool_name, "arguments": arguments},
        }],
    })
    messages.append({
        "role": "tool",
        "content": f"<tool_response>\n{result_str}\n</tool_response>",
    })
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "mcp_tool_execution",
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result_str[:2000],
    }


async def _gather_tools(req: ChatCompletionRequest) -> Tuple[Optional[List[Dict[str, Any]]], Set[str]]:
    """Combine request ``functions`` + MCP server tools + Skill virtual tools.

    Returns ``(tools, mcp_tool_names)`` where ``mcp_tool_names`` is the set of
    tool names that originate from MCP servers (used for server_side routing).
    """
    tools: List[Dict[str, Any]] = []
    mcp_names: Set[str] = set()
    if req.functions:
        tools.extend(req.functions)
    cm = get_config_manager()
    if cm.get_mcp_tool_call_mode() == "server_side":
        try:
            from core.mcp_manager import get_mcp_manager

            mm = get_mcp_manager()
            mcp_tools = await mm.list_all_tools()
            for t in mcp_tools:
                name = (t.get("function") or {}).get("name", "")
                if name:
                    mcp_names.add(name)
            tools.extend(mcp_tools)
        except Exception as e:
            LOG.warning("Failed to gather MCP tools: %r", e)
    # Always advertise skill virtual tools so the LLM can ``read_skill`` / ``list_skills``.
    try:
        from core.skill_manager import get_skill_manager
        tools.extend(get_skill_manager().virtual_tools())
    except Exception as e:
        LOG.warning("Failed to gather skill tools: %r", e)
    return tools or None, mcp_names


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
    tools, mcp_names = await _gather_tools(request)

    request_id = request.request_id or str(uuid.uuid4())

    # Legacy abort: in the main-branch protocol, abort_id IS the request_id to cancel
    if request.abort_id:
        LOG.info("Legacy abort via chat request for abort_id=%r", request.abort_id)
        await backend.abort(request.abort_id)  # direct: abort_id == request_id

    if request.abort_id:
        _active_requests[request.abort_id] = (provider_cfg.name, request_id)

    rtype = request.type or 0
    extra_body = None
    if rtype == 1:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        result = await backend.generate(
            messages=messages,
            sampling=_sampling_from_request(request, max_tokens),
            tools=tools,
            request_id=request_id,
            extra_body=extra_body,
        )
    finally:
        if request.abort_id:
            _active_requests.pop(request.abort_id, None)

    thought, answer = _split_thought_and_answer(result.text)
    if result.reasoning and not thought:
        thought = result.reasoning

    if rtype == 1:
        add_knowledge(content=answer, character="_shared")
        return ChatCompletionResponseChoice(
            index=0,
            thought="",
            embedding_list=[],
            message=ChatMessage(role="assistant", content=answer),
            finish_reason="stop",
        )

    if result.function_calls:
        fc = result.function_calls[0]
        return ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            embedding_list=[],
            message=ChatMessage(
                role="assistant",
                content=answer or "",
                function_call={"name": fc.get("name", ""), "arguments": fc.get("arguments", "")},
            ),
            finish_reason="function_call",
        )
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
                add_knowledge(m.content, "_shared")
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
    system_prefix = _build_persona_system_prefix(
        request.character or "", embeddings_text
    )

    messages = _prepare_messages(
        request.messages, provider_cfg=provider_cfg, system_prefix=system_prefix
    )
    tools, mcp_names = await _gather_tools(request)
    sampling = _sampling_from_request(request, max_tokens)
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)}}
        if request.enable_thinking is not None
        else None
    )

    request_ts = datetime.now(timezone.utc).isoformat()

    # Legacy abort: in the main-branch protocol, abort_id IS the request_id to cancel
    if request.abort_id:
        LOG.info("Legacy abort via chat request for abort_id=%r", request.abort_id)
        await backend.abort(request.abort_id)  # direct: abort_id == request_id

    request_id = request.request_id or str(uuid.uuid4())
    if request.abort_id:
        _active_requests[request.abort_id] = (provider_cfg.name, request_id)

    max_rounds = get_config_manager().get_mcp_max_tool_rounds()
    result = None
    thought = ""
    answer = ""
    round_num = 0
    async def _mcp_execute():
        nonlocal result, thought, answer, round_num
        round_num += 1
        try:
            result = await backend.generate(
                messages=messages,
                sampling=sampling,
                tools=tools,
                request_id=request_id,
                extra_body=extra_body,
            )
        finally:
            if request.abort_id and result is None:
                _active_requests.pop(request.abort_id, None)
        thought, answer = _split_thought_and_answer(result.text)
        if result.reasoning and not thought:
            thought = result.reasoning
        answer = _postprocess_answer(answer, request.character or "")
        _append_vllm_request_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "character": request.character or "",
            "provider": provider_cfg.name,
            "model": provider_cfg.model,
            "base_url": provider_cfg.base_url,
            "type": f"mcp_round/{round_num}",
            "request": {
                "messages": messages,
                "sampling": sampling,
                "tools": tools,
                "extra_body": extra_body,
            },
            "response": {
                "finish_reason": result.finish_reason,
                "function_calls": result.function_calls or None,
                "tokens": {
                    "prompt": result.prompt_tokens,
                    "completion": result.completion_tokens,
                },
                "answer": answer,
                "thought": thought,
            },
        })

    async def _mcp_loop() -> Optional[ChatCompletionResponseChoice]:
        nonlocal result, thought, answer
        for _round in range(max_rounds):
            await _mcp_execute()
            if not result.function_calls:
                return ChatCompletionResponseChoice(
                    index=index,
                    thought=thought,
                    embedding_list=embedding_index_list,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason=_map_finish_reason(result.finish_reason),
                )
            executed_any = False
            for fc in result.function_calls:
                name = fc.get("name", "")
                if name in mcp_names:
                    from core.mcp_manager import get_mcp_manager
                    mm = get_mcp_manager()
                    log_entry = await _execute_mcp_tool(mm, name, fc.get("arguments", "{}"), messages)
                    if log_entry is not None:
                        _append_vllm_request_log(log_entry)
                    executed_any = True
                else:
                    # Frontend or Skill function — return to caller.
                    fc = result.function_calls[0]
                    return ChatCompletionResponseChoice(
                        index=index,
                        thought=thought,
                        embedding_list=embedding_index_list,
                        message=ChatMessage(
                            role="assistant",
                            content=answer or "",
                            function_call={
                                "name": fc.get("name", ""),
                                "arguments": fc.get("arguments", ""),
                            },
                        ),
                        finish_reason="function_call",
                    )
            if not executed_any:
                break  # no MCP tools to execute (shouldn't happen due to name check)
        # Max rounds exhausted — return whatever text we have.
        return ChatCompletionResponseChoice(
            index=index,
            thought=thought,
            embedding_list=embedding_index_list,
            message=ChatMessage(role="assistant", content=answer),
            finish_reason=_map_finish_reason(result.finish_reason if result else "stop"),
        )

    # --- Run tool loop -------------------------------------------------------
    choice = await _mcp_loop()
    if request.abort_id:
        _active_requests.pop(request.abort_id, None)

    # Log the conversation turn to the chat log file.
    try:
        log_entry = {
            "ts": request_ts,
            "character": request.character or "",
            "user": user_text,
            "assistant": answer,
            "thought": thought,
            "finish_reason": result.finish_reason,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
            },
        }
        _append_chat_log(log_entry)
    except Exception:
        pass  # Logging failure must never break the response.

    return choice


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

    system_prefix = _build_persona_system_prefix(
        request.character or "", embeddings_text
    )
    messages = _prepare_messages(
        request.messages, provider_cfg=provider_cfg, system_prefix=system_prefix
    )
    tools, mcp_names = await _gather_tools(request)
    sampling = _sampling_from_request(request, max_tokens)
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)}}
        if request.enable_thinking is not None
        else None
    )

    request_ts = datetime.now(timezone.utc).isoformat()

    request_id = request.request_id or str(uuid.uuid4())
    # Legacy abort: in the main-branch protocol, abort_id IS the request_id to cancel
    if request.abort_id:
        LOG.info("Legacy abort via chat request for abort_id=%r", request.abort_id)
        await backend.abort(request.abort_id)  # direct: abort_id == request_id
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

    collected_text: List[str] = []
    collected_function_calls: List[Dict[str, Any]] = []

    try:
        it = await backend.generate_stream(
            messages=messages,
            sampling=sampling,
            tools=tools,
            request_id=request_id,
            extra_body=extra_body,
        )
        async for chunk in it:  # type: StreamChunk
            if chunk.text:
                collected_text.append(chunk.text)
            if chunk.function_calls:
                collected_function_calls = chunk.function_calls
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
                            _map_finish_reason(chunk.finish_reason)
                            if chunk.finish_reason
                            else None
                        ),
                    )
                ],
            )
    finally:
        if request.abort_id:
            _active_requests.pop(request.abort_id, None)

    # Log the full conversation turn after streaming ends.
    try:
        full_answer = "".join(collected_text).strip()
        thought, clean_answer = _split_thought_and_answer(full_answer)
        if not thought:
            thought = ""
        log_entry = {
            "ts": request_ts,
            "character": request.character or "",
            "user": user_text,
            "assistant": clean_answer or full_answer,
            "thought": thought,
            "finish_reason": "stop",
            "tokens": {},
        }
        _append_chat_log(log_entry)
        _append_vllm_request_log({
            "ts": request_ts,
            "character": request.character or "",
            "provider": provider_cfg.name,
            "model": provider_cfg.model,
            "base_url": provider_cfg.base_url,
            "type": "streaming",
            "request": {
                "messages": messages,
                "sampling": sampling,
                "tools": tools,
                "extra_body": extra_body,
            },
            "response": {
                "finish_reason": "stop",
                "function_calls": collected_function_calls or None,
                "answer": clean_answer or full_answer,
                "thought": thought,
            },
        })
    except Exception:
        pass  # Logging failure must never break the response.


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