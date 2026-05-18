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
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from core.config_manager import get_config_manager
from core.mcp_manager import get_mcp_manager
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
