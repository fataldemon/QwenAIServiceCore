"""Admin REST routes.

Every route under ``/admin/api/*`` is meant to be called from the Gradio UI
(or any other admin client). Routes are intentionally kept JSON-in / JSON-out
with very little input validation -- :mod:`core.config_manager` is the
source of truth for schema invariants, and any error there bubbles up here
as a 400.

There is no authentication on these routes beyond the optional HTTP Basic
middleware in ``main.py``. If you expose ``/admin`` publicly, **set
``--api-auth``**.

Endpoints
---------

Providers:

* ``GET    /admin/api/providers``                  -- list
* ``GET    /admin/api/providers/{name}``           -- detail
* ``PUT    /admin/api/providers/{name}``           -- upsert
* ``DELETE /admin/api/providers/{name}``           -- remove
* ``POST   /admin/api/providers/{name}/activate``  -- make active

MCP:

* ``GET    /admin/api/mcp/servers``                -- list
* ``GET    /admin/api/mcp/servers/{name}``         -- detail
* ``PUT    /admin/api/mcp/servers/{name}``         -- upsert
* ``DELETE /admin/api/mcp/servers/{name}``         -- remove
* ``POST   /admin/api/mcp/mode``                   -- set passthrough/server_side
* ``GET    /admin/api/mcp/health``                 -- connection states

Skills:

* ``GET    /admin/api/skills``                     -- list
* ``GET    /admin/api/skills/{name}``              -- body
* ``POST   /admin/api/skills/reload``              -- rescan disk

Personas (per-character system prompt):

* ``GET    /admin/api/personas``                   -- list characters with persona.json
* ``GET    /admin/api/personas/{character}``       -- detail
* ``PUT    /admin/api/personas/{character}``       -- upsert
* ``DELETE /admin/api/personas/{character}``       -- remove
* ``POST   /admin/api/personas/{character}/preview`` -- render the full system
  prompt that ``chat_on_setting`` would inject (handy when editing in the UI).
  Body: ``{"user_text": "...", "information": "..."}``; both optional.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from core.config_manager import get_config_manager
from core.mcp_manager import get_mcp_manager
from core.persona_manager import get_persona_manager
from core.skill_manager import get_skill_manager
from llm.backends.registry import invalidate as invalidate_backend


def register_admin_routes(app: FastAPI) -> None:
    """Attach every ``/admin/api/*`` route to ``app``."""

    # ------------------- providers -------------------

    @app.get("/admin/api/providers")
    async def list_providers():
        cm = get_config_manager()
        return {
            "active": cm.get_active_provider_name(),
            "providers": [p.to_dict() | {"name": p.name} for p in cm.list_providers()],
        }

    @app.get("/admin/api/providers/{name}")
    async def get_provider(name: str):
        p = get_config_manager().get_provider(name)
        if p is None:
            raise HTTPException(404, "provider not found")
        return p.to_dict() | {"name": p.name}

    @app.put("/admin/api/providers/{name}")
    async def upsert_provider(name: str, body: Dict[str, Any]):
        try:
            cm = get_config_manager()
            cfg = await cm.upsert_provider(name, body)
            # Drop any cached HTTP client so the new config takes effect on
            # the next request.
            await invalidate_backend(name)
            return cfg.to_dict() | {"name": cfg.name}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete("/admin/api/providers/{name}")
    async def delete_provider(name: str):
        ok = await get_config_manager().delete_provider(name)
        if not ok:
            raise HTTPException(404, "provider not found")
        await invalidate_backend(name)
        return {"ok": True}

    @app.post("/admin/api/providers/{name}/activate")
    async def activate_provider(name: str):
        ok = await get_config_manager().activate_provider(name)
        if not ok:
            raise HTTPException(404, "provider not found")
        return {"ok": True, "active": name}

    # ------------------- mcp -------------------

    @app.get("/admin/api/mcp/servers")
    async def list_mcp_servers():
        cm = get_config_manager()
        return {
            "tool_call_mode": cm.get_mcp_tool_call_mode(),
            "tool_call_timeout": cm.get_mcp_tool_call_timeout(),
            "servers": [s.to_dict() | {"name": s.name} for s in cm.list_mcp_servers()],
        }

    @app.get("/admin/api/mcp/servers/{name}")
    async def get_mcp_server(name: str):
        s = get_config_manager().get_mcp_server(name)
        if s is None:
            raise HTTPException(404, "mcp server not found")
        return s.to_dict() | {"name": s.name}

    @app.put("/admin/api/mcp/servers/{name}")
    async def upsert_mcp_server(name: str, body: Dict[str, Any]):
        try:
            s = await get_config_manager().upsert_mcp_server(name, body)
            # Force a reconnect on next use so toggling ``enabled`` takes
            # effect immediately.
            await get_mcp_manager().invalidate(name)
            return s.to_dict() | {"name": s.name}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete("/admin/api/mcp/servers/{name}")
    async def delete_mcp_server(name: str):
        ok = await get_config_manager().delete_mcp_server(name)
        if not ok:
            raise HTTPException(404, "mcp server not found")
        await get_mcp_manager().invalidate(name)
        return {"ok": True}

    @app.post("/admin/api/mcp/mode")
    async def set_mcp_mode(body: Dict[str, Any]):
        try:
            await get_config_manager().set_mcp_tool_call_mode(body.get("mode", ""))
            return {"ok": True, "mode": body.get("mode")}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/admin/api/mcp/health")
    async def mcp_health():
        return await get_mcp_manager().health()

    # ------------------- skills -------------------

    @app.get("/admin/api/skills")
    async def list_skills():
        return {"skills": get_skill_manager().list_skills()}

    @app.get("/admin/api/skills/{name}")
    async def read_skill(name: str):
        body = get_skill_manager().read_skill(name)
        if body is None:
            raise HTTPException(404, "skill not found")
        return {"name": name, "body": body}

    @app.post("/admin/api/skills/reload")
    async def reload_skills():
        get_skill_manager().reload()
        return {"ok": True, "skills": get_skill_manager().list_skills()}

    # ------------------- personas -------------------

    @app.get("/admin/api/personas")
    async def list_personas():
        pm = get_persona_manager()
        return {
            "personas": [
                {"character": p.character, **p.to_dict()}
                for p in pm.list_personas()
            ],
        }

    @app.get("/admin/api/personas/{character}")
    async def get_persona(character: str):
        p = get_persona_manager().get_persona(character)
        if p is None:
            raise HTTPException(404, "persona not found")
        return {"character": p.character, **p.to_dict()}

    @app.put("/admin/api/personas/{character}")
    async def upsert_persona(character: str, body: Dict[str, Any]):
        try:
            p = await get_persona_manager().upsert_persona(character, body)
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"character": p.character, **p.to_dict()}

    @app.delete("/admin/api/personas/{character}")
    async def delete_persona(character: str):
        ok = await get_persona_manager().delete_persona(character)
        if not ok:
            raise HTTPException(404, "persona not found")
        return {"ok": True}

    @app.post("/admin/api/personas/{character}/preview")
    async def preview_persona(character: str, body: Dict[str, Any]):
        """Render the full system prompt that would be injected.

        This is a *dry run*: it calls ``process_embedding`` with whatever
        ``user_text`` the caller supplies (or an empty string), splices the
        result into the persona's ``setting``/``reply_instruction``, and
        returns the final text. No LLM call is made.
        """
        from llm.chat import _build_persona_system_prefix
        from embedding.embedding import process_embedding, remove_reference_url

        user_text = (body.get("user_text") or "").strip()
        information = body.get("information") or ""
        embeddings_text = ""
        if user_text:
            try:
                embeddings_text, _ = process_embedding(
                    content=remove_reference_url(user_text),
                    top_k=5,
                    character=character,
                    client_buffer=[],
                    max_length=8,
                    client_information=information,
                )
            except Exception as e:
                # Fall through with empty embeddings so the user still
                # sees the literal persona text in the preview.
                return {
                    "character": character,
                    "system_prompt": _build_persona_system_prefix(character, ""),
                    "embeddings_text": "",
                    "warning": f"process_embedding failed: {e!r}",
                }
        return {
            "character": character,
            "system_prompt": _build_persona_system_prefix(
                character, embeddings_text
            ),
            "embeddings_text": embeddings_text,
        }
