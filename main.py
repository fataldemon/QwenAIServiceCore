"""FastAPI application entry point.

Compared to the legacy ``main.py`` (which booted an in-process vLLM
``AsyncLLMEngine``), this version:

* talks to an OpenAI-compatible upstream over HTTP (see :mod:`llm.backends`);
* exposes streaming completions via SSE under the same ``/v1/chat/completions``
  route the existing front-end already uses (the route is now stream-aware
  via the ``stream`` field already declared in :class:`ChatCompletionRequest`);
* keeps every legacy route shape intact so the existing front-end keeps
  working without any code change:

    - ``POST /assistant/v1/chat/completions`` (analysis)
    - ``POST /v1/chat/completions``           (chat, now stream-capable)
    - ``WS   /ws/{ws_mode}``                  (quick-reply)

* adds an admin surface for runtime configuration:

    - ``GET  /admin``                          -- Gradio UI (mounted)
    - ``GET  /admin/api/providers``            -- list providers
    - ``PUT  /admin/api/providers/{name}``     -- upsert
    - ``DELETE /admin/api/providers/{name}``   -- delete
    - ``POST /admin/api/providers/{name}/activate``
    - ``GET  /admin/api/mcp/servers`` + PUT/DELETE/health
    - ``POST /admin/api/abort/{abort_id}``     -- cooperative abort

* migrates legacy embedding pickles on startup automatically (no-op when
  already migrated).

The old in-process vLLM bootstrap (``vllm_start_engine``) is preserved as
dead code in :mod:`llm.local_llm_manage` for documentation purposes only -- it
is never imported here. Run an external ``vllm serve`` instead and point a
provider at it via :class:`ConfigManager` (default fixture in
``config/providers.example.json``).
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from admin.routes import register_admin_routes
from core.config_manager import get_config_manager
from core.mcp_manager import get_mcp_manager
from core.persona_manager import get_persona_manager
from core.skill_manager import get_skill_manager
from embedding.migrate import migrate_all
from llm.chat import (
    abort_request,
    chat,
    chat_on_setting,
    chat_on_setting_stream,
)
from models.base import ChatCompletionRequest, ChatCompletionResponse
from template import (
    LEGACY_ALICE_IMAGE_SETTING,
    LEGACY_ALICE_REPLY_INSTRUCTION,
    LEGACY_ALICE_SETTING,
    _get_args,
    max_analysis_len,
    max_chat_len,
    max_quick_reply,
)
from utils.websocketutils import WebsocketManager

LOG = logging.getLogger(__name__)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Keeps the legacy HTTP Basic auth behaviour bit-for-bit identical."""

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get("Authorization")
        if authorization:
            try:
                _schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass
        return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager singletons -- catches obvious config errors on boot.
    get_config_manager()
    get_skill_manager()
    persona_manager = get_persona_manager()
    # Seed the legacy "Tendou Arisu" persona on first boot of an
    # upgraded deployment so existing front-ends keep getting the same
    # character behaviour without manual file authoring. Idempotent.
    try:
        persona_manager.seed_legacy_alice_if_missing(
            character="tendou_arisu",
            display_name="天童爱丽丝",
            setting=LEGACY_ALICE_SETTING,
            reply_instruction=LEGACY_ALICE_REPLY_INSTRUCTION,
            image_setting=LEGACY_ALICE_IMAGE_SETTING,
        )
    except Exception as e:  # pragma: no cover
        LOG.warning("Persona seed failed: %r", e)
    # Best-effort migration; never fatal.
    try:
        summary = migrate_all()
        if summary:
            LOG.info("Embedding migration summary: %s", summary)
    except Exception as e:  # pragma: no cover -- best-effort
        LOG.warning("Embedding migration failed: %r", e)
    yield
    # Graceful shutdown: close MCP sessions + backend HTTP clients.
    try:
        await get_mcp_manager().shutdown()
    except Exception as e:  # pragma: no cover
        LOG.warning("MCP shutdown failed: %r", e)
    try:
        from llm.backends.registry import invalidate_all

        await invalidate_all()
    except Exception as e:  # pragma: no cover
        LOG.warning("Backend shutdown failed: %r", e)


app = FastAPI(lifespan=lifespan)
websocket_manager = WebsocketManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Legacy completion routes (unchanged contract for the existing front-end)
# ---------------------------------------------------------------------------


@app.post("/assistant/v1/chat/completions", response_model=ChatCompletionResponse)
async def completion_without_lora(request: ChatCompletionRequest):
    choice = await chat(request=request, max_tokens=max_analysis_len)
    return ChatCompletionResponse(
        model=request.model, choices=[choice], object="chat.completion"
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Non-streaming response by default; streams SSE when ``stream=true``.

    The streaming format follows OpenAI's convention: each SSE ``data:`` line
    carries a ``ChatCompletionResponse`` with ``object="chat.completion.chunk"``,
    terminated by ``data: [DONE]``. This is what every modern OpenAI client
    library and front-end already knows how to consume.
    """
    if request.stream:
        async def _gen():
            try:
                async for resp in chat_on_setting_stream(
                    request=request, max_tokens=max_chat_len, index=0
                ):
                    yield {"data": resp.json()}
            except Exception as e:
                LOG.exception("streaming failure: %r", e)
                yield {"data": '{"error":"stream_failure"}'}
            yield {"data": "[DONE]"}

        return EventSourceResponse(_gen())

    choice = await chat_on_setting(
        request=request, max_tokens=max_chat_len, index=0
    )
    resp = ChatCompletionResponse(
        model=request.model, choices=[choice], object="chat.completion"
    )
    # Keep the WebSocket broadcast contract intact for legacy front-ends.
    await websocket_manager.broadcast(resp.json())
    return resp


@app.websocket("/ws/{ws_mode}")
async def websocket_endpoint(ws_mode: str, websocket: WebSocket):
    """Quick-reply WebSocket -- same JSON contract as before."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json(mode=ws_mode)
            LOG.debug("ws data received: %s", data)
            try:
                request = ChatCompletionRequest.parse_obj(data)
                choice = await chat_on_setting(
                    request=request, max_tokens=max_quick_reply, index=1
                )
                resp = ChatCompletionResponse(
                    model=request.model, choices=[choice], object="chat.completion"
                )
                await websocket_manager.send_message_to_client(resp.json(), websocket)
            except ValidationError as e:
                LOG.warning("ws validation error: %s", e.json())
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.post("/admin/api/abort/{abort_id}")
async def admin_abort(abort_id: str):
    ok = await abort_request(abort_id)
    if not ok:
        raise HTTPException(status_code=404, detail="abort_id not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Admin: REST + Gradio mount
# ---------------------------------------------------------------------------

register_admin_routes(app)

try:
    from webui import build_admin_ui  # type: ignore
    import gradio as gr  # type: ignore

    _admin_ui = build_admin_ui()
    app = gr.mount_gradio_app(app, _admin_ui, path="/admin")
except Exception as e:  # pragma: no cover -- gradio is optional
    LOG.warning("Gradio admin UI not mounted: %r", e)


if __name__ == "__main__":
    args = _get_args()
    if args.api_auth:
        user, _, pwd = args.api_auth.partition(":")
        app.add_middleware(BasicAuthMiddleware, username=user, password=pwd)
    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
