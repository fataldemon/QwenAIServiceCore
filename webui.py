"""Gradio admin UI mounted at ``/admin``.

This is a thin presentation layer over the JSON admin API exposed by
:mod:`admin.routes`. We deliberately do **not** hit the HTTP API from the UI
-- since the UI runs in the same process as the FastAPI app, talking to the
Python managers directly is simpler and avoids needing a self-targeted HTTP
client just to render a form.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr  # type: ignore

from core.config_manager import get_config_manager
from core.mcp_manager import get_mcp_manager
from core.persona_manager import get_persona_manager
from core.skill_manager import get_skill_manager
from llm.backends.registry import invalidate as invalidate_backend

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def capture_main_loop() -> None:
    """Store the FastAPI event loop so Gradio thread callbacks can dispatch to it."""
    global _MAIN_LOOP
    try:
        _MAIN_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        pass

_EMBEDDING_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "embedding"
)
_CHAT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "chat_log.jsonl"
)
_VLLM_REQUEST_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "vllm_request_log.jsonl"
)

_KNOWLEDGE_SUBJECTS = ["setting", "knowledge", "expression"]

_CUSTOM_CSS = """
#vllm-log-display textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    background: #1a1a2e !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
#vllm-log-display label {
    color: #58a6ff !important;
}
footer { visibility: hidden !important; }
"""


def _run(coro):
    """Synchronously execute an async function from a Gradio callback."""
    global _MAIN_LOOP
    if _MAIN_LOOP is not None and _MAIN_LOOP.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, _MAIN_LOOP)
        return future.result()
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


def _active_provider_markdown() -> str:
    cm = get_config_manager()
    active_char = cm.get_active_character()
    char_info = f" | Character: `{active_char}`" if active_char else ""
    return (
        f"**Active Provider:** `{cm.get_active_provider_name() or '(none)'}`"
        f"{char_info}"
    )


def _provider_choices() -> List[str]:
    return [p.name for p in get_config_manager().list_providers()]


def _refresh_providers() -> Tuple[List[List[Any]], str, gr.update]:
    cm = get_config_manager()
    active_name = cm.get_active_provider_name()
    rows = []
    for p in cm.list_providers():
        marker = "✓" if p.name == active_name else ""
        rows.append(
            [
                marker,
                p.name,
                p.type,
                p.model,
                p.base_url,
                p.supports_vision,
                p.supports_audio,
                p.supports_video,
                p.prefetch_media,
                p.description,
            ]
        )
    return (
        rows,
        _active_provider_markdown(),
        gr.update(choices=_provider_choices(), value=active_name),
    )


def _provider_table_select(evt: gr.SelectData, table) -> Tuple:
    if evt.index is None:
        return ("", "", "", "", "", "", "", "", "", "", "")
    row_idx = evt.index[0]
    try:
        import pandas as pd
        if isinstance(table, pd.DataFrame):
            if row_idx >= len(table):
                return ("", "", "", "", "", "", "", "", "", "", "")
            row = table.iloc[row_idx].tolist()
        else:
            if row_idx >= len(table):
                return ("", "", "", "", "", "", "", "", "", "", "")
            row = table[row_idx]
    except Exception:
        return ("", "", "", "", "", "", "", "", "", "", "")
    return (
        row[1] if len(row) > 1 else "",
        row[4] if len(row) > 4 else "",
        row[3] if len(row) > 3 else "",
        "",
        row[2] if len(row) > 2 else "",
        bool(row[5]) if len(row) > 5 else False,
        bool(row[6]) if len(row) > 6 else False,
        bool(row[7]) if len(row) > 7 else False,
        bool(row[8]) if len(row) > 8 else False,
        "",
        row[9] if len(row) > 9 else "",
    )


def _save_provider(
    name: str,
    base_url: str,
    api_key: str,
    model: str,
    ptype: str,
    supports_vision: bool,
    supports_audio: bool,
    supports_video: bool,
    prefetch_media: bool,
    extra_body_json: str,
    description: str,
) -> Tuple[List[List[Any]], str, gr.update, str]:
    if not name.strip():
        rows, active, radio = _refresh_providers()
        return rows, active, radio, "✗ name is required"
    try:
        extra_body = json.loads(extra_body_json) if extra_body_json.strip() else {}
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a JSON object")
    except Exception as e:
        rows, active, radio = _refresh_providers()
        return rows, active, radio, f"✗ bad extra_body JSON: {e}"
    body = {
        "type": ptype.strip() or "openai_compatible",
        "base_url": base_url.strip(),
        "api_key": api_key.strip(),
        "model": model.strip(),
        "supports_vision": bool(supports_vision),
        "supports_audio": bool(supports_audio),
        "supports_video": bool(supports_video),
        "prefetch_media": bool(prefetch_media),
        "extra_body": extra_body,
        "description": description.strip(),
    }
    try:
        _run(get_config_manager().upsert_provider(name.strip(), body))
        _run(invalidate_backend(name.strip()))
    except Exception as e:
        rows, active, radio = _refresh_providers()
        return rows, active, radio, f"✗ {e}"
    rows, active, radio = _refresh_providers()
    return rows, active, radio, f"✓ saved {name.strip()}"


def _activate_provider(name: str) -> Tuple[List[List[Any]], str, gr.update, str]:
    name = (name or "").strip()
    if not name:
        rows, active, radio = _refresh_providers()
        return rows, active, radio, "✗ name is required"
    ok = _run(get_config_manager().activate_provider(name))
    rows, active, radio = _refresh_providers()
    return rows, active, radio, ("✓ activated" if ok else "✗ unknown provider")


def _delete_provider(name: str) -> Tuple[List[List[Any]], str, gr.update, str]:
    name = name.strip()
    if not name:
        rows, active, radio = _refresh_providers()
        return rows, active, radio, "✗ name is required"
    ok = _run(get_config_manager().delete_provider(name))
    if ok:
        _run(invalidate_backend(name))
    rows, active, radio = _refresh_providers()
    return rows, active, radio, ("✓ deleted" if ok else "✗ unknown provider")


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------


def _refresh_mcp() -> Tuple[List[List[Any]], str]:
    cm = get_config_manager()
    rows = []
    health = _run(get_mcp_manager().health())
    for s in cm.list_mcp_servers():
        h = health.get(s.name, {})
        rows.append(
            [
                s.name,
                s.enabled,
                s.transport,
                s.command or s.url or "",
                h.get("connected", False),
                h.get("tools", 0),
                s.description,
            ]
        )
    return rows, f"Mode: `{cm.get_mcp_tool_call_mode()}` | Timeout: {cm.get_mcp_tool_call_timeout()}s"


def _save_mcp(
    name: str,
    enabled: bool,
    transport: str,
    command: str,
    args_text: str,
    url: str,
    headers_json: str,
    description: str,
) -> Tuple[List[List[Any]], str, str]:
    if not name.strip():
        rows, info = _refresh_mcp()
        return rows, info, "✗ name is required"
    try:
        headers = json.loads(headers_json) if headers_json.strip() else {}
        if not isinstance(headers, dict):
            raise ValueError("headers must be a JSON object")
    except Exception as e:
        rows, info = _refresh_mcp()
        return rows, info, f"✗ bad headers JSON: {e}"
    args = [a for a in (args_text or "").splitlines() if a.strip()]
    body = {
        "enabled": bool(enabled),
        "transport": transport,
        "command": command.strip() or None,
        "args": args,
        "url": url.strip() or None,
        "headers": headers,
        "description": description.strip(),
    }
    try:
        _run(get_config_manager().upsert_mcp_server(name.strip(), body))
        _run(get_mcp_manager().invalidate(name.strip()))
    except Exception as e:
        rows, info = _refresh_mcp()
        return rows, info, f"✗ {e}"
    rows, info = _refresh_mcp()
    return rows, info, f"✓ saved {name.strip()}"


def _delete_mcp(name: str) -> Tuple[List[List[Any]], str, str]:
    name = name.strip()
    if not name:
        rows, info = _refresh_mcp()
        return rows, info, "✗ name is required"
    ok = _run(get_config_manager().delete_mcp_server(name))
    if ok:
        _run(get_mcp_manager().invalidate(name))
    rows, info = _refresh_mcp()
    return rows, info, ("✓ deleted" if ok else "✗ unknown server")


def _set_mcp_mode(mode: str) -> Tuple[List[List[Any]], str, str]:
    try:
        _run(get_config_manager().set_mcp_tool_call_mode(mode))
    except Exception as e:
        rows, info = _refresh_mcp()
        return rows, info, f"✗ {e}"
    rows, info = _refresh_mcp()
    return rows, info, f"✓ mode = {mode}"


# ---------------------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------------------


def _refresh_skills() -> Tuple[List[List[Any]], str]:
    sm = get_skill_manager()
    rows = [
        [s["name"], s.get("version", ""), s.get("auto_inject", False), s.get("description", "")]
        for s in sm.list_skills()
    ]
    return rows, f"{len(rows)} skill(s) loaded"


def _reload_skills() -> Tuple[List[List[Any]], str]:
    get_skill_manager().reload()
    return _refresh_skills()


def _read_skill(name: str) -> str:
    if not name.strip():
        return ""
    body = get_skill_manager().read_skill(name.strip())
    return body or "(skill not found)"


# ---------------------------------------------------------------------------
# Persona / Character helpers
# ---------------------------------------------------------------------------


def _persona_choices() -> List[str]:
    return [p.character for p in get_persona_manager().list_personas()]


def _refresh_personas() -> Tuple[List[List[Any]], gr.update, gr.update]:
    pm = get_persona_manager()
    active = get_config_manager().get_active_character()
    rows = []
    for p in pm.list_personas():
        rows.append(
            [
                p.character,
                p.display_name,
                bool(p.setting),
                bool(p.reply_instruction),
                bool(p.image_setting),
            ]
        )
    choices = [p.character for p in pm.list_personas()]
    return rows, gr.update(choices=choices), gr.update(choices=choices, value=active)


def _load_persona(character: str) -> Tuple:
    character = (character or "").strip()
    if not character:
        return tuple("" for _ in range(9))
    p = get_persona_manager().get_persona(character)
    ac = get_config_manager().get_active_character()
    is_active = ac == character
    active_label = " **[active]**" if is_active else ""
    if p is None:
        return ("", "", "", "", "", "", "", "", f"✗ no persona.json for `{character}` (will create on save){active_label}")
    return (
        p.display_name,
        p.setting,
        p.reply_instruction,
        p.image_setting,
        str(p.max_chat_len or ""),
        str(p.max_analysis_len or ""),
        str(p.max_quick_reply or ""),
        str(p.default_temperature or ""),
        f"✓ loaded `{character}`{active_label}",
    )


def _load_persona_and_set_active(character: str) -> Tuple:
    character = (character or "").strip()
    if not character:
        return tuple("" for _ in range(10))
    p = get_persona_manager().get_persona(character)
    try:
        _run(get_config_manager().set_active_character(character))
    except Exception:
        pass
    if p is None:
        return ("", "", "", "", "", "", "", "", f"✗ no persona.json for `{character}`", character)
    return (
        p.display_name,
        p.setting,
        p.reply_instruction,
        p.image_setting,
        str(p.max_chat_len or ""),
        str(p.max_analysis_len or ""),
        str(p.max_quick_reply or ""),
        str(p.default_temperature or ""),
        f"✓ loaded & activated `{character}`",
        character,
    )


def _save_persona(
    character: str,
    display_name: str,
    setting: str,
    reply_instruction: str,
    image_setting: str,
    max_chat_len: str,
    max_analysis_len: str,
    max_quick_reply: str,
    default_temperature: str,
) -> Tuple[List[List[Any]], gr.update, gr.update, str]:
    character = (character or "").strip()
    if not character:
        rows, dd, radio = _refresh_personas()
        return rows, dd, radio, "✗ character name is required"
    body = {
        "display_name": display_name,
        "setting": setting,
        "reply_instruction": reply_instruction,
        "image_setting": image_setting,
    }
    for key, val in [
        ("max_chat_len", max_chat_len),
        ("max_analysis_len", max_analysis_len),
        ("max_quick_reply", max_quick_reply),
        ("default_temperature", default_temperature),
    ]:
        s = val.strip() if isinstance(val, str) else ""
        if s:
            try:
                if key == "default_temperature":
                    body[key] = float(s)
                else:
                    body[key] = int(s)
            except ValueError:
                pass
    try:
        _run(get_persona_manager().upsert_persona(character, body))
    except Exception as e:
        rows, dd, radio = _refresh_personas()
        return rows, dd, radio, f"✗ {e}"
    rows, dd, radio = _refresh_personas()
    return rows, dd, radio, f"✓ saved `{character}`"


def _delete_persona(character: str) -> Tuple[List[List[Any]], gr.update, gr.update, str]:
    character = (character or "").strip()
    if not character:
        rows, dd, radio = _refresh_personas()
        return rows, dd, radio, "✗ character name is required"
    ok = _run(get_persona_manager().delete_persona(character))
    rows, dd, radio = _refresh_personas()
    return rows, dd, radio, ("✓ deleted" if ok else "✗ unknown character")


def _preview_persona(character: str, user_text: str) -> str:
    character = (character or "").strip()
    if not character:
        return "(pick a character first)"
    from llm.chat import _build_persona_system_prefix
    from embedding.embedding import process_embedding, remove_reference_url

    embeddings_text = ""
    user_text = (user_text or "").strip()
    if user_text:
        try:
            embeddings_text, _ = process_embedding(
                content=remove_reference_url(user_text),
                top_k=5,
                character=character,
                client_buffer=[],
                max_length=8,
                client_information="",
            )
        except Exception as e:
            return f"(process_embedding failed: {e!r})\n\n" + _build_persona_system_prefix(character, "")
    return _build_persona_system_prefix(character, embeddings_text)


# ---------------------------------------------------------------------------
# Knowledge Base helpers
# ---------------------------------------------------------------------------


def _kb_character_choices() -> List[str]:
    if not os.path.isdir(_EMBEDDING_ROOT):
        return []
    return sorted(
        d for d in os.listdir(_EMBEDDING_ROOT)
        if os.path.isdir(os.path.join(_EMBEDDING_ROOT, d))
        and not d.startswith("__")
    )


def _kb_refresh_choices() -> Tuple[gr.update, gr.update, str, str]:
    choices = _kb_character_choices()
    return (
        gr.update(choices=choices),
        gr.update(choices=_KNOWLEDGE_SUBJECTS),
        "",
        "",
    )


def _kb_load_files(character: str, subject: str) -> Tuple[gr.update, str, str]:
    if not character or not subject:
        return gr.update(choices=[], value=None), "", f"Select both character and subject."
    subject_dir = os.path.join(_EMBEDDING_ROOT, character, subject)
    if not os.path.isdir(subject_dir):
        return gr.update(choices=[], value=None), "", f"No `{subject}` directory for `{character}`."
    mem_files = sorted(f for f in os.listdir(subject_dir) if f.endswith(".mem"))
    if not mem_files:
        return gr.update(choices=[], value=None), "", f"No `.mem` files in `{character}/{subject}`."
    return gr.update(choices=mem_files, value=mem_files[0]), mem_files[0] if mem_files else "", f"{len(mem_files)} file(s)."


def _kb_read_file(character: str, subject: str, filename: str) -> str:
    if not character or not subject or not filename:
        return ""
    filepath = os.path.join(_EMBEDDING_ROOT, character, subject, filename)
    if not os.path.isfile(filepath):
        return f"(file not found: {filename})"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"(read error: {e})"


def _kb_save_file(character: str, subject: str, filename: str, content: str) -> str:
    if not character or not subject or not filename:
        return "✗ character, subject and filename required."
    filepath = os.path.join(_EMBEDDING_ROOT, character, subject, filename)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✓ saved `{filename}`"
    except Exception as e:
        return f"✗ {e}"


def _kb_new_file(character: str, subject: str, filename: str) -> Tuple[str, str]:
    if not character or not subject or not filename:
        return "", "✗ filename is required."
    if not filename.endswith(".mem"):
        filename = filename + ".mem"
    filepath = os.path.join(_EMBEDDING_ROOT, character, subject, filename)
    if os.path.isfile(filepath):
        return "", f"✗ `{filename}` already exists."
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content, f"✓ created `{filename}`"
    except Exception as e:
        return "", f"✗ {e}"


def _kb_rebuild_index(character: str, subject: str) -> str:
    if not character or not subject:
        return "✗ character and subject required."
    from embedding.embedding import generate_vector
    try:
        result = generate_vector(character, subject)
        if result == "success":
            return f"✓ index rebuilt for `{character}/{subject}`"
        elif result == "empty":
            return f"⭘ no content, index removed for `{character}/{subject}`"
        else:
            return f"✗ rebuild failed: {result}"
    except Exception as e:
        return f"✗ rebuild error: {e}"


def _kb_index_status(character: str, subject: str) -> str:
    if not character or not subject:
        return "Select a character and subject."
    from embedding.data_store import load_materials, index_path
    p = index_path(character, subject)
    if not os.path.exists(p):
        return "No index file."
    try:
        import faiss  # type: ignore
        idx = faiss.read_index(p)
        n_total = int(idx.ntotal)
    except Exception:
        n_total = 0
    materials = load_materials(character, subject)
    n_materials = len(materials) if materials else 0
    return f"Index: {n_total} vectors | Materials: {n_materials} rows | File: `{os.path.basename(p)}`"


# ---------------------------------------------------------------------------
# Chat Logs helpers (kept for reference; new UI uses vLLM request log)
# ---------------------------------------------------------------------------


def _read_chat_logs(limit: int = 200) -> List[List[Any]]:
    if not os.path.isfile(_CHAT_LOG_FILE):
        return []
    try:
        with open(_CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    rows: List[List[Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append([
            entry.get("ts", "")[-23:-7] if entry.get("ts") else "",
            entry.get("character", ""),
            entry.get("user", "")[:80],
            entry.get("assistant", "")[:120],
            entry.get("thought", "")[:60] if entry.get("thought") else "",
            entry.get("finish_reason", ""),
            entry.get("tokens", {}).get("prompt", ""),
            entry.get("tokens", {}).get("completion", ""),
        ])
    return rows[::-1]


def _refresh_chat_logs() -> Tuple[List[List[Any]], str]:
    rows = _read_chat_logs(200)
    return rows, f"{len(rows)} log entries (newest first)"


def _filter_chat_logs(character: str) -> Tuple[List[List[Any]], str]:
    all_rows = _read_chat_logs(2000)
    if not character:
        return all_rows[:200], f"{min(len(all_rows), 200)} log entries"
    filtered = [r for r in all_rows if r[1] == character]
    return filtered[:200], f"{len(filtered)} entries for `{character}`"


# ---------------------------------------------------------------------------
# vLLM Request Log helpers (real-time console viewer)
# ---------------------------------------------------------------------------


def _sanitize_request_for_display(req: dict) -> dict:
    """Deep copy the request dict, replace base64 data URIs with short placeholders."""
    req = copy.deepcopy(req)
    for msg in req.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                for media_key in ("image_url", "video_url", "audio_url"):
                    media = part.get(media_key) or {}
                    if isinstance(media, dict):
                        url = media.get("url", "")
                        if isinstance(url, str) and url.startswith("data:"):
                            media["url"] = f"[base64 {media_key.lstrip('_')}, {len(url)} chars]"
        elif isinstance(content, str) and content.startswith("data:"):
            msg["content"] = f"[base64 data, {len(content)} chars]"
    return req


def _format_vllm_request_log() -> str:
    """Read the vLLM request log and return a formatted console-style string."""
    if not os.path.isfile(_VLLM_REQUEST_LOG_FILE):
        return "(no request log yet — send a chat request to see entries)"

    SEP = "─" * 80

    try:
        with open(_VLLM_REQUEST_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"(error reading log: {e})"

    if not lines:
        return "(request log is empty)"

    parts: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = entry.get("ts", "")[:19].replace("T", " ")
        character = entry.get("character", "")
        provider = entry.get("provider", "")
        model = entry.get("model", "")
        base_url = entry.get("base_url", "")
        req_type = entry.get("type", "")
        req = entry.get("request") or {}
        resp = entry.get("response") or {}

        parts.append(SEP)
        parts.append(
            f"[{ts}]  character={character}  provider={provider}  "
            f"model={model}  type={req_type}"
        )
        parts.append(SEP)

        parts.append(f">>> REQUEST  ({base_url}/chat/completions)")
        parts.append(json.dumps(_sanitize_request_for_display(req), ensure_ascii=False, indent=2))

        parts.append(SEP)
        parts.append("<<< RESPONSE")
        resp_lines = []
        finish = resp.get("finish_reason", "")
        tokens = resp.get("tokens") or {}
        if tokens:
            resp_lines.append(
                f"prompt_tokens={tokens.get('prompt','?')}  "
                f"completion_tokens={tokens.get('completion','?')}  "
                f"finish_reason={finish}"
            )
        else:
            resp_lines.append(f"finish_reason={finish}")
        answer = resp.get("answer", "")
        thought = resp.get("thought", "")
        if thought:
            resp_lines.append(f"thought: {thought[:500]}")
        if answer:
            resp_lines.append(f"answer: {answer[:1000]}")
        parts.extend(resp_lines)
        parts.append(SEP)

    # Reverse so newest appears at the top of the text box.
    parts.reverse()
    return "\n".join(parts)


_last_log_mtime = 0.0


def _refresh_vllm_log_display() -> str:
    global _last_log_mtime
    if os.path.isfile(_VLLM_REQUEST_LOG_FILE):
        try:
            mtime = os.path.getmtime(_VLLM_REQUEST_LOG_FILE)
            if mtime == _last_log_mtime:
                return gr.update()
            _last_log_mtime = mtime
        except OSError:
            pass
    else:
        _last_log_mtime = 0.0
    return _format_vllm_request_log()


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------


def build_admin_ui() -> "gr.Blocks":
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    with gr.Blocks(title="QwenAIServiceCore Admin", theme=theme, css=_CUSTOM_CSS) as ui:
        gr.Markdown("# QwenAIServiceCore Admin")
        gr.Markdown(
            "Manage LLM providers, MCP servers, characters (persona + knowledge base), "
            "skills, and monitor real-time vLLM requests."
        )

        # ---------- Providers ----------
        with gr.Tab("LLM Providers"):
            prov_status = gr.Markdown()
            with gr.Row():
                prov_radio = gr.Radio(
                    choices=_provider_choices(),
                    label="Active Provider (select to activate)",
                    interactive=True,
                )
            prov_table = gr.Dataframe(
                headers=[
                    "active", "name", "type", "model", "base_url",
                    "vision", "audio", "video", "prefetch", "description",
                ],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                p_name = gr.Textbox(label="name")
                p_type = gr.Textbox(label="type", value="openai_compatible")
                p_base_url = gr.Textbox(label="base_url (e.g. http://localhost:8000/v1)")
                p_model = gr.Textbox(label="model")
            with gr.Row():
                p_api_key = gr.Textbox(label="api_key", type="password")
                p_description = gr.Textbox(label="description")
            with gr.Row():
                p_v = gr.Checkbox(label="supports vision", value=True)
                p_a = gr.Checkbox(label="supports audio")
                p_video = gr.Checkbox(label="supports video")
                p_pre = gr.Checkbox(label="prefetch media URLs", value=True)
            p_extra = gr.Textbox(
                label="extra_body (JSON merged into every request)",
                placeholder='e.g. {"mm_processor_kwargs": {"fps": 2}}',
                lines=3,
            )
            with gr.Row():
                p_save = gr.Button("Save / Update", variant="primary")
                p_delete = gr.Button("Delete by name", variant="stop")
                p_refresh = gr.Button("Refresh")
            p_message = gr.Markdown()

            prov_table.select(
                _provider_table_select,
                [prov_table],
                [p_name, p_base_url, p_model, p_api_key, p_type,
                 p_v, p_a, p_video, p_pre, p_extra, p_description],
            )

            prov_radio.change(
                _activate_provider,
                [prov_radio],
                [prov_table, prov_status, prov_radio, p_message],
            )

            p_save.click(
                _save_provider,
                [p_name, p_base_url, p_api_key, p_model, p_type,
                 p_v, p_a, p_video, p_pre, p_extra, p_description],
                [prov_table, prov_status, prov_radio, p_message],
            )
            p_delete.click(
                _delete_provider, [p_name],
                [prov_table, prov_status, prov_radio, p_message],
            )
            p_refresh.click(_refresh_providers, None, [prov_table, prov_status, prov_radio])
            ui.load(_refresh_providers, None, [prov_table, prov_status, prov_radio])

        # ---------- MCP ----------
        with gr.Tab("MCP Servers"):
            mcp_status = gr.Markdown()
            mcp_table = gr.Dataframe(
                headers=["name", "enabled", "transport", "command/url",
                         "connected", "tools", "description"],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                m_name = gr.Textbox(label="name")
                m_enabled = gr.Checkbox(label="enabled")
                m_transport = gr.Dropdown(
                    choices=["stdio", "sse", "streamable_http"],
                    value="stdio", label="transport",
                )
            with gr.Row():
                m_command = gr.Textbox(label="command (stdio only)")
                m_url = gr.Textbox(label="url (sse / streamable_http)")
            m_args = gr.Textbox(label="args (one per line, stdio only)", lines=3)
            m_headers = gr.Textbox(label="headers (JSON, remote only)", lines=2, placeholder="{}")
            m_description = gr.Textbox(label="description")
            with gr.Row():
                m_save = gr.Button("Save / Update", variant="primary")
                m_delete = gr.Button("Delete by name", variant="stop")
                m_refresh = gr.Button("Refresh")
            mode_dropdown = gr.Dropdown(
                choices=["passthrough", "server_side"],
                value="passthrough",
                label="tool_call_mode",
            )
            m_set_mode = gr.Button("Apply mode")
            m_message = gr.Markdown()

            m_save.click(
                _save_mcp,
                [m_name, m_enabled, m_transport, m_command, m_args, m_url,
                 m_headers, m_description],
                [mcp_table, mcp_status, m_message],
            )
            m_delete.click(_delete_mcp, [m_name], [mcp_table, mcp_status, m_message])
            m_refresh.click(_refresh_mcp, None, [mcp_table, mcp_status])
            m_set_mode.click(_set_mcp_mode, [mode_dropdown], [mcp_table, mcp_status, m_message])
            ui.load(_refresh_mcp, None, [mcp_table, mcp_status])

        # ---------- Skills ----------
        with gr.Tab("Skills"):
            sk_status = gr.Markdown()
            sk_table = gr.Dataframe(
                headers=["name", "version", "auto_inject", "description"],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                sk_name = gr.Textbox(label="name to preview")
                sk_view = gr.Button("View body")
                sk_reload = gr.Button("Reload from disk", variant="primary")
            sk_body = gr.Code(label="SKILL.md body", language="markdown", lines=15)

            sk_view.click(_read_skill, [sk_name], [sk_body])
            sk_reload.click(_reload_skills, None, [sk_table, sk_status])
            ui.load(_refresh_skills, None, [sk_table, sk_status])

        # ---------- Characters (Persona + Knowledge Base) ----------
        with gr.Tab("Characters"):
            gr.Markdown(
                "Manage per-character persona settings and knowledge base files. "
                "Persona config at `embedding/<character>/persona.json`; "
                "knowledge `.mem` files at `embedding/<character>/<subject>/`."
            )

            # --- Persona Section ---
            gr.Markdown("### Persona Configuration")
            pe_status = gr.Markdown()
            pe_table = gr.Dataframe(
                headers=["character", "display_name", "has_setting",
                         "has_reply_instruction", "has_image_setting"],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                pc_radio = gr.Radio(
                    choices=_persona_choices(),
                    label="Active Character (select to load & activate)",
                    interactive=True,
                )
                pe_refresh = gr.Button("Refresh")
            with gr.Row():
                pe_character = gr.Textbox(
                    label="character (folder name under embedding/)",
                    placeholder="e.g. tendou_arisu",
                )
                pe_display_name = gr.Textbox(label="display_name")
            pe_setting = gr.Textbox(
                label="setting (system prompt; may include {embeddings})",
                lines=12,
            )
            pe_reply_instruction = gr.Textbox(
                label="reply_instruction (appended after setting)",
                lines=4,
            )
            pe_image_setting = gr.Textbox(
                label="image_setting (optional figure framing)",
                lines=4,
            )
            with gr.Row():
                pe_max_chat_len = gr.Textbox(label="max_chat_len", placeholder="e.g. 15000")
                pe_max_analysis_len = gr.Textbox(label="max_analysis_len", placeholder="e.g. 6000")
                pe_max_quick_reply = gr.Textbox(label="max_quick_reply", placeholder="e.g. 600")
                pe_default_temperature = gr.Textbox(label="default_temperature", placeholder="e.g. 0.7")
            with gr.Row():
                pe_save = gr.Button("Save / Update", variant="primary")
                pe_delete = gr.Button("Delete by name", variant="stop")
            with gr.Accordion("Preview rendered system prompt", open=False):
                pe_preview_input = gr.Textbox(
                    label="simulated user message (used to call process_embedding)",
                    lines=2,
                )
                pe_preview_btn = gr.Button("Render preview", variant="primary")
                pe_preview_out = gr.Code(label="rendered system prompt", lines=20)
            pe_message = gr.Markdown()

            pc_radio.change(
                _load_persona_and_set_active,
                [pc_radio],
                [pe_display_name, pe_setting, pe_reply_instruction,
                 pe_image_setting, pe_max_chat_len, pe_max_analysis_len,
                 pe_max_quick_reply, pe_default_temperature,
                 pe_message, pe_character],
            )

            pe_save.click(
                _save_persona,
                [pe_character, pe_display_name, pe_setting, pe_reply_instruction,
                 pe_image_setting, pe_max_chat_len, pe_max_analysis_len,
                 pe_max_quick_reply, pe_default_temperature],
                [pe_table, pc_radio, pc_radio, pe_message],
            )
            pe_delete.click(
                _delete_persona, [pe_character],
                [pe_table, pc_radio, pc_radio, pe_message],
            )
            pe_refresh.click(_refresh_personas, None, [pe_table, pc_radio, pc_radio])
            pe_preview_btn.click(
                _preview_persona, [pe_character, pe_preview_input], [pe_preview_out],
            )

            # --- Knowledge Base Section ---
            gr.Markdown("### Knowledge Base")
            kb_status = gr.Markdown("Select a character and subject.")
            with gr.Row():
                kb_character = gr.Dropdown(
                    choices=_kb_character_choices(),
                    label="Character",
                )
                kb_subject = gr.Dropdown(
                    choices=_KNOWLEDGE_SUBJECTS,
                    value="setting",
                    label="Subject (knowledge type)",
                )
                kb_refresh_list = gr.Button("Refresh file list")
            kb_file_list = gr.Dropdown(
                choices=[],
                label=".mem file",
                interactive=True,
            )
            with gr.Row():
                kb_new_filename = gr.Textbox(
                    label="New file name (e.g. new_chat.mem)",
                    placeholder=".mem extension auto-added",
                )
                kb_new_btn = gr.Button("Create new file")
            kb_content = gr.Code(
                label="File content",
                language="markdown",
                lines=25,
            )
            with gr.Row():
                kb_save = gr.Button("Save file", variant="primary")
                kb_delete_file = gr.Button("Delete file", variant="stop")
            with gr.Row():
                kb_rebuild = gr.Button("Rebuild FAISS index")
                kb_index_status_btn = gr.Button("Show index status")
            kb_index_info = gr.Markdown()
            kb_action_msg = gr.Markdown()

            kb_character.change(
                _kb_load_files, [kb_character, kb_subject],
                [kb_file_list, kb_file_list, kb_status],
            )
            kb_subject.change(
                _kb_load_files, [kb_character, kb_subject],
                [kb_file_list, kb_file_list, kb_status],
            )
            kb_refresh_list.click(
                _kb_load_files, [kb_character, kb_subject],
                [kb_file_list, kb_file_list, kb_status],
            )
            kb_file_list.change(
                _kb_read_file, [kb_character, kb_subject, kb_file_list], [kb_content],
            )
            kb_save.click(
                _kb_save_file,
                [kb_character, kb_subject, kb_file_list, kb_content],
                [kb_action_msg],
            )
            kb_new_btn.click(
                _kb_new_file,
                [kb_character, kb_subject, kb_new_filename],
                [kb_content, kb_action_msg],
            ).then(
                _kb_load_files, [kb_character, kb_subject],
                [kb_file_list, kb_file_list, kb_status],
            )
            kb_rebuild.click(
                _kb_rebuild_index, [kb_character, kb_subject], [kb_action_msg],
            )
            kb_index_status_btn.click(
                _kb_index_status, [kb_character, kb_subject], [kb_index_info],
            )

            ui.load(_refresh_personas, None, [pe_table, pc_radio, pc_radio])
            ui.load(_kb_refresh_choices, None, [kb_character, kb_subject, kb_status, kb_content])

        # ---------- Conversation Logs (table) ----------
        with gr.Tab("Conversation Logs"):
            gr.Markdown(
                "Recent chat conversation logs from `logs/chat_log.jsonl`. "
                "Newest entries appear first."
            )
            log_status = gr.Markdown()
            log_filter = gr.Dropdown(
                choices=[""] + _persona_choices(),
                label="Filter by character (empty = show all)",
                value="",
            )
            with gr.Row():
                log_refresh = gr.Button("Refresh")
            log_table = gr.Dataframe(
                headers=["time", "character", "user", "assistant",
                         "thought", "finish", "prompt_tk", "completion_tk"],
                interactive=False,
                wrap=True,
            )

            log_filter.change(_filter_chat_logs, [log_filter], [log_table, log_status])
            log_refresh.click(_refresh_chat_logs, None, [log_table, log_status])
            ui.load(_refresh_chat_logs, None, [log_table, log_status])

        # ---------- Request Monitor (real-time vLLM request log) ----------
        with gr.Tab("Request Monitor"):
            gr.Markdown(
                "Real-time console showing the full vLLM request payloads "
                "(including system prompts) and responses. "
                "Log file: `logs/vllm_request_log.jsonl` (cleared on each startup). "
                "Newest entries appear at the top."
            )
            with gr.Row():
                log_interval = gr.Slider(
                    0.5, 30, value=3, step=0.5,
                    label="Refresh interval (seconds)",
                )
                log_force = gr.Button("Refresh now")
            monitor_status = gr.Markdown()
            monitor_display = gr.Textbox(
                label="vLLM Request Log",
                lines=35,
                max_lines=200,
                interactive=False,
                elem_id="vllm-log-display",
                autoscroll=False,
                value="(waiting for requests — auto-refreshes every few seconds)",
            )
            timer = gr.Timer(value=3, active=True)

            timer.tick(_refresh_vllm_log_display, None, [monitor_display])
            log_force.click(
                _format_vllm_request_log, None, [monitor_display],
            )
            log_interval.change(
                lambda val: gr.Timer(value=float(val), active=True),
                [log_interval], [timer],
            )

    return ui


if __name__ == "__main__":  # pragma: no cover -- standalone dev mode
    build_admin_ui().launch()
