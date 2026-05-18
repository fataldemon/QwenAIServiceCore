"""Gradio admin UI mounted at ``/admin``.

This is a thin presentation layer over the JSON admin API exposed by
:mod:`admin.routes`. We deliberately do **not** hit the HTTP API from the UI
-- since the UI runs in the same process as the FastAPI app, talking to the
Python managers directly is simpler and avoids needing a self-targeted HTTP
client just to render a form.

The UI is intentionally minimal: one tab per administrable subject
(providers / MCP servers / skills). Anything more ambitious belongs in a
proper SPA which the existing chat front-end already provides for the chat
side; this surface is purely for operators.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple

import gradio as gr  # type: ignore

from core.config_manager import get_config_manager
from core.mcp_manager import get_mcp_manager
from core.skill_manager import get_skill_manager
from llm.backends.registry import invalidate as invalidate_backend


# ---------------------------------------------------------------------------
# Async helpers (gradio callbacks are sync; we bridge to asyncio with run())
# ---------------------------------------------------------------------------


def _run(coro):
    """Synchronously execute an async function from a Gradio callback."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside FastAPI's loop; schedule and wait.
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


def _refresh_providers() -> Tuple[List[List[Any]], str]:
    cm = get_config_manager()
    rows = []
    for p in cm.list_providers():
        marker = "✓" if p.name == cm.get_active_provider_name() else ""
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
                p.description,
            ]
        )
    return rows, f"Active: {cm.get_active_provider_name() or '(none)'}"


def _save_provider(
    name: str,
    base_url: str,
    api_key: str,
    model: str,
    supports_vision: bool,
    supports_audio: bool,
    supports_video: bool,
    prefetch_media: bool,
    extra_body_json: str,
    description: str,
) -> Tuple[List[List[Any]], str, str]:
    if not name.strip():
        rows, active = _refresh_providers()
        return rows, active, "✗ name is required"
    try:
        extra_body = json.loads(extra_body_json) if extra_body_json.strip() else {}
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a JSON object")
    except Exception as e:
        rows, active = _refresh_providers()
        return rows, active, f"✗ bad extra_body JSON: {e}"

    body = {
        "type": "openai_compatible",
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
        rows, active = _refresh_providers()
        return rows, active, f"✗ {e}"
    rows, active = _refresh_providers()
    return rows, active, f"✓ saved {name.strip()}"


def _activate_provider(name: str) -> Tuple[List[List[Any]], str, str]:
    name = name.strip()
    if not name:
        rows, active = _refresh_providers()
        return rows, active, "✗ name is required"
    ok = _run(get_config_manager().activate_provider(name))
    rows, active = _refresh_providers()
    return rows, active, ("✓ activated" if ok else "✗ unknown provider")


def _delete_provider(name: str) -> Tuple[List[List[Any]], str, str]:
    name = name.strip()
    if not name:
        rows, active = _refresh_providers()
        return rows, active, "✗ name is required"
    ok = _run(get_config_manager().delete_provider(name))
    if ok:
        _run(invalidate_backend(name))
    rows, active = _refresh_providers()
    return rows, active, ("✓ deleted" if ok else "✗ unknown provider")


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
    return rows, f"Mode: {cm.get_mcp_tool_call_mode()} | Timeout: {cm.get_mcp_tool_call_timeout()}s"


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
# Build the UI
# ---------------------------------------------------------------------------


def build_admin_ui() -> "gr.Blocks":
    with gr.Blocks(title="QwenAIServiceCore Admin") as ui:
        gr.Markdown("# QwenAIServiceCore Admin")
        gr.Markdown(
            "Manage LLM providers, MCP servers and skill modules. "
            "Changes take effect immediately for new requests; in-flight "
            "requests keep their current backend."
        )

        # ---------- Providers ----------
        with gr.Tab("LLM Providers"):
            prov_status = gr.Markdown()
            prov_table = gr.Dataframe(
                headers=[
                    "active",
                    "name",
                    "type",
                    "model",
                    "base_url",
                    "vision",
                    "audio",
                    "video",
                    "description",
                ],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                p_name = gr.Textbox(label="name")
                p_base_url = gr.Textbox(label="base_url (e.g. http://localhost:8000/v1)")
                p_model = gr.Textbox(label="model")
            with gr.Row():
                p_api_key = gr.Textbox(label="api_key", type="password")
                p_description = gr.Textbox(label="description")
            with gr.Row():
                p_v = gr.Checkbox(label="supports vision")
                p_a = gr.Checkbox(label="supports audio")
                p_video = gr.Checkbox(label="supports video")
                p_pre = gr.Checkbox(label="prefetch local files as data: URIs")
            p_extra = gr.Textbox(
                label="extra_body (JSON merged into every request)",
                placeholder='e.g. {"mm_processor_kwargs": {"fps": 2}}',
                lines=3,
            )
            with gr.Row():
                p_save = gr.Button("Save / Update")
                p_activate = gr.Button("Activate by name")
                p_delete = gr.Button("Delete by name", variant="stop")
                p_refresh = gr.Button("Refresh")
            p_message = gr.Markdown()

            p_save.click(
                _save_provider,
                [p_name, p_base_url, p_api_key, p_model, p_v, p_a, p_video, p_pre, p_extra, p_description],
                [prov_table, prov_status, p_message],
            )
            p_activate.click(_activate_provider, [p_name], [prov_table, prov_status, p_message])
            p_delete.click(_delete_provider, [p_name], [prov_table, prov_status, p_message])
            p_refresh.click(lambda: _refresh_providers(), None, [prov_table, prov_status])
            ui.load(lambda: _refresh_providers(), None, [prov_table, prov_status])

        # ---------- MCP ----------
        with gr.Tab("MCP Servers"):
            mcp_status = gr.Markdown()
            mcp_table = gr.Dataframe(
                headers=["name", "enabled", "transport", "command/url", "connected", "tools", "description"],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                m_name = gr.Textbox(label="name")
                m_enabled = gr.Checkbox(label="enabled")
                m_transport = gr.Dropdown(
                    choices=["stdio", "sse", "streamable_http"], value="stdio", label="transport"
                )
            with gr.Row():
                m_command = gr.Textbox(label="command (stdio only)")
                m_url = gr.Textbox(label="url (sse / streamable_http)")
            m_args = gr.Textbox(label="args (one per line, stdio only)", lines=3)
            m_headers = gr.Textbox(label="headers (JSON, remote only)", lines=2, placeholder="{}")
            m_description = gr.Textbox(label="description")
            with gr.Row():
                m_save = gr.Button("Save / Update")
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
                [m_name, m_enabled, m_transport, m_command, m_args, m_url, m_headers, m_description],
                [mcp_table, mcp_status, m_message],
            )
            m_delete.click(_delete_mcp, [m_name], [mcp_table, mcp_status, m_message])
            m_refresh.click(lambda: _refresh_mcp(), None, [mcp_table, mcp_status])
            m_set_mode.click(_set_mcp_mode, [mode_dropdown], [mcp_table, mcp_status, m_message])
            ui.load(lambda: _refresh_mcp(), None, [mcp_table, mcp_status])

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
                sk_reload = gr.Button("Reload from disk")
            sk_body = gr.Code(label="SKILL.md body", language="markdown", lines=15)

            sk_view.click(_read_skill, [sk_name], [sk_body])
            sk_reload.click(lambda: _reload_skills(), None, [sk_table, sk_status])
            ui.load(lambda: _refresh_skills(), None, [sk_table, sk_status])

    return ui


if __name__ == "__main__":  # pragma: no cover -- standalone dev mode
    build_admin_ui().launch()
