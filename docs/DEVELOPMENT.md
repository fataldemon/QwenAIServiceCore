# Development Guide

> Audience: **code maintainers** of this gateway. For *operating* a running
> deployment (rollback, adding providers, restarting safely) see
> [`MAINTENANCE.md`](MAINTENANCE.md), which is intentionally separate from
> this document.

This guide explains how the code is organised, how a request travels
through it, and **where to look when you want to change a specific
behaviour** — the last part being the bulk of the document.

---

## 1. Repository layout

```
QwenAIServiceCore/
├── main.py                    FastAPI app, route declarations, lifespan.
├── webui.py                   Gradio admin UI mounted at /admin.
│                              (LLM Providers, MCP Servers, Skills,
│                              Characters, Shared Knowledge,
│                              Request Monitor with raw SSE viewer).
├── template.py                Static config (CLI flags, length budgets) +
│                              one shared system fragment (REACT_INSTRUCTION).
│                              Persona strings only live here as
│                              LEGACY_ALICE_* seed material.
├── admin/
│   └── routes.py              JSON REST under /admin/api/*.
│
├── core/
│   ├── config_manager.py      Provider + MCP + global config; file-backed
│   │                          (config/*.json) with atomic write + lock.
│   ├── content_normalizer.py  Multi-modal content parsing, GIF expansion,
│   │                          OpenAI <-> legacy placeholder bridging.
│   ├── mcp_manager.py         MCP session pool, list_tools(), call_tool().
│   ├── persona_manager.py     Per-character persona.json registry; provides
│   │                          the system prompt that chat_on_setting uses.
│   └── skill_manager.py       Skill discovery (skills/<name>/SKILL.md),
│                              auto-injection, virtual tool surface.
│
├── llm/
│   ├── chat.py                ★ The orchestration layer. Everything the
│   │                          HTTP / WS handlers eventually call.
│   ├── backends/
│   │   ├── base.py            LLMBackend ABC + GenerationResult / StreamChunk.
│   │   ├── registry.py        get_backend(name) cache; invalidate(name).
│   │   └── openai_compatible.py
│   │                          The one concrete backend today. SSE-only —
│   │                          even non-streaming requests aggregate from a
│   │                          stream so abort works mid-flight.
│   └── local_llm_manage.py    [DEAD CODE] kept for reference; the previous
│                              in-process vLLM bootstrap.
│
├── embedding/
│   ├── embedding.py           Public API: process_embedding(),
│   │                          add_knowledge(), check_emotion().
│   ├── data_store.py          materials.jsonl + FAISS index + binaries/.
│   └── migrate.py             One-time .pkl -> jsonl migration at boot.
│
├── models/
│   └── base.py                All pydantic schemas (request/response).
│
├── utils/                     Legacy helpers, kept for compatibility.
├── config/                    Runtime JSON config (git-ignored).
├── skills/<name>/SKILL.md     One folder per skill.
└── docs/                      This documentation.
```

A few stable assumptions you can rely on:

* **`llm/chat.py` is the only file that calls backends.** Routes don't.
  Routes call `chat.py` functions.
* **`core/config_manager.py` is the only file that touches
  `config/*.json`.** Everything else asks the manager.
* **`embedding/data_store.py` is the only file that touches
  `materials.jsonl`.** Higher layers go through `embedding/embedding.py`.

---

## 2. Request data flow

### `POST /v1/chat/completions` (non-streaming)

```
main.py: create_chat_completion(request)
  └─> llm.chat.chat_on_setting(request, max_tokens, index=0)
        ├─> embedding.process_embedding(...)           # knowledge retrieval
        ├─> core.persona_manager.get_persona(character)
        │       └─> assembled into the system prompt prefix
        ├─> _prepare_messages(...)                     # normalise content,
        │       └─> core.content_normalizer.*          # strip unsupported
        │                                              # modalities,
        │                                              # expand GIFs
        ├─> _gather_tools(req)                         # functions + MCP
        ├─> llm.backends.registry.get_backend()
        │     └─> OpenAICompatibleBackend.generate(...)
        │           └─> internally generate_stream(...) aggregator
        └─> _postprocess_answer(text, character)       # 【emotion】 snap
```

### `POST /v1/chat/completions` with `stream=true`

```
main.py: create_chat_completion(request)
  └─> sse_starlette.EventSourceResponse(
        llm.chat.chat_on_setting_stream(request, max_tokens, index=0)
      )
        ├─> embedding + persona + content normalize (same as above)
        ├─> backend.generate_stream(...)               # AsyncIterator[StreamChunk]
        └─> yields ChatCompletionResponse(chunk, ...)
```

### `WS /ws/{ws_mode}`

```
main.py: websocket_endpoint
  └─> for each inbound JSON:
        └─> chat_on_setting(request, max_tokens=max_quick_reply, index=1)
              -- same as the non-streaming path, smaller token budget
```

### Cooperative abort

```
front-end -> POST /admin/api/abort/<abort_id>
              └─> main.py: admin_abort
                    └─> llm.chat.abort_request(abort_id)
                          └─> backend.abort(request_id)
                                └─> sets asyncio.Event
                                      └─> generate_stream loop checks event
                                            └─> yields StreamChunk(finish_reason="abort")
                                                  └─> generate() aggregator
                                                        breaks and returns
                                                        the partial text
```

This is **why `generate()` is implemented as an aggregator over the
stream**, not a plain `stream=false` POST: it preserves the legacy
contract whereby aborting a non-streaming request still returns a normal
`ChatCompletionResponse` whose `content` is the partial answer and whose
`finish_reason` is `"abort"`.

---

## 3. "I want to change X — where do I look?"

| Goal | File(s) | Notes |
|---|---|---|
| Add a new LLM provider that **is** OpenAI-compatible | `/admin/api/providers` UI or `config/providers.json` | No code changes. |
| Add a new LLM provider that is **not** OpenAI-compatible (e.g. Anthropic native, Bedrock) | New `llm/backends/<name>.py` implementing `LLMBackend`; teach `llm/backends/registry.py::_build_backend` to dispatch on `ProviderConfig.type`; extend `ProviderConfig` if you need new fields. | Keep `OpenAICompatibleBackend` as the reference shape. |
| Change default sampling (temperature, top_p) | `llm/chat.py::_sampling_from_request` | Per-request fields always win over server defaults. |
| Change the system-prompt template / knowledge-base injection style | `core/persona_manager.py` (schema) + `llm/chat.py::_build_persona_system_prefix` (assembly logic) | The strings themselves live in `embedding/<character>/persona.json`, not in code. |
| Add or edit a character persona | `/admin` → **Personas** tab, **or** edit `embedding/<character>/persona.json` directly. | Hot-reloaded on the next request — no restart needed. |
| Change the `【emotion】` snapping logic | `llm/chat.py::_postprocess_answer` (parser) + `embedding/embedding.py::check_emotion` (lookup) | The two are decoupled; the parser only knows the syntax. |
| Change how `<think>...</think>` is split out | `llm/chat.py::_split_thought_and_answer` | |
| Support a new modality, or change GIF handling | `core/content_normalizer.py` | `expand_gif_parts` is the GIF code path; add new `MediaPart` kinds beside it. |
| Knowledge-base retrieval: top_k, max_length | `llm/chat.py::chat_on_setting` (and `chat_on_setting_stream`) — passes them to `process_embedding(top_k=..., max_length=...)` | |
| Change the on-disk shape of knowledge entries | `embedding/data_store.py` (schema) + `embedding/migrate.py` (add a new migration step) | Bump `SCHEMA_VERSION`. |
| Add an `/admin/api/...` endpoint | `admin/routes.py` only. Put the implementation in `core/` or `llm/`. | Routes are intentionally a thin shell over manager methods. |
| Add a tab / form to the admin Gradio UI | `webui.py` | The UI talks to the Python managers directly (no self-HTTP). |
| Tweak abort semantics (e.g. also send an upstream `DELETE`) | `llm/backends/openai_compatible.py::abort` | Don't touch the event registration; that's how the public contract works. |
| Add another HTTP route under `/v1/...` or `/assistant/v1/...` | `main.py` — declare the route, delegate to `llm.chat`. | Routes never call backends directly. |
| Skill auto-injection rules | `core/skill_manager.py::matches_user_text` | `_gather_tools` and `chat_on_setting` consume the result. |
| Add a new MCP transport (e.g. WebSocket) | `core/mcp_manager.py::_open_session` (dispatch) and a small helper alongside. | |
| Add a runtime-toggleable flag | `core/config_manager.py` (add field + persistence) + admin route + UI control. **Don't** use `os.environ` for these. | |
| Change the WebSocket protocol | `main.py::websocket_endpoint` + `utils/websocketutils.py` | |
| Strip control tokens like `<|endoftext|>` | `llm/chat.py::_postprocess_answer` | Already strips a couple — add more there. |
| Add a field to the request/response log format | `llm/chat.py::_append_vllm_request_log` | The log entry dict is assembled at the call site — add your key there. Also update the monitor formatter in `webui.py::_format_vllm_request_log`. |
| Surface a new field from the upstream SSE stream | `llm/backends/openai_compatible.py::generate_stream` (parse into `StreamChunk`), then `generate()` accumulates into `GenerationResult.raw_events`. | The `raw_events` field flows to the log and the Request Monitor. |
| Change how image paths in ``image_setting`` are resolved | `llm/chat.py::_resolve_media_paths` | Resolves relative paths against `embedding/<character>/image/`. |

If something genuinely doesn't fit any of the rows above, prefer adding
a small module under `core/` and importing it from `llm/chat.py` rather
than fattening `main.py` or any backend.

---

## 4. Conventions and anti-patterns

* **No business logic in route handlers.** `main.py` and
  `admin/routes.py` should be ≤ 5 lines per endpoint, delegating to a
  manager or `llm/chat.py`.
* **Backends never read config files directly.** They receive a
  `ProviderConfig` from the registry. This keeps swap-out trivial.
* **Do not bypass `_active_requests`.** The cooperative abort guarantees
  ride on it. If you add a new completion entry point, register the
  abort_id at the start and pop it in `finally`.
* **New runtime flags go through `core/config_manager.py`.** No
  environment variables, no module-level globals.
* **Schema lives in `models/base.py`.** Anything that crosses the HTTP
  boundary should be a pydantic model declared there.
* **Persona prompts are data, not code.** Do not re-introduce hard-coded
  Alice/Yuzu/etc. strings in `template.py` or `llm/chat.py`. Edit
  `embedding/<character>/persona.json` (directly or via the admin UI).
* **The legacy in-process `llm/local_llm_manage.py` is frozen.** Do not
  add new behaviour there. It exists only as a reference for what the
  legacy `main` branch did.
* **Every backend must populate `raw_events` on `GenerationResult`** and
  `raw` on each `StreamChunk`. The contract is that `raw` is the
  upstream's raw SSE data (JSON object), and `raw_events` is the
  complete list of all such objects for the full generation. This is
  consumed by the Request Monitor and `logs/vllm_request_log.jsonl`.

---

## 5. Manual smoke tests

A quick local round-trip without writing a unit test suite:

```bash
# 1) Start an upstream model (any OpenAI-compatible server).
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --host 127.0.0.1

# 2) Start the gateway.
python main.py --server-port 8000 --server-name 127.0.0.1

# 3) Plain completion.
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any",
        "messages": [{"role":"user","content":"Hello"}],
        "character": "tendou_arisu"
      }' | jq

# 4) Streaming.
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
        "model": "any",
        "stream": true,
        "messages": [{"role":"user","content":"Say a long sentence."}],
        "character": "tendou_arisu"
      }'

# 5) Abort. Terminal A:
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any",
        "messages": [{"role":"user","content":"Write a 1000-word essay."}],
        "character": "tendou_arisu",
        "abort_id": "smoke1"
      }'
# Terminal B (within a second of A):
curl -X POST http://127.0.0.1:8000/admin/api/abort/smoke1
# Expected in A: a normal ChatCompletionResponse with partial content
# and finish_reason="abort".

# 6) Persona preview.
curl -s -X POST http://127.0.0.1:8000/admin/api/personas/tendou_arisu/preview \
  -H "Content-Type: application/json" \
  -d '{"user_text":"老师你好"}' | jq -r .system_prompt | head -n 40
```

---

## 6. Where things deliberately *don't* live

To save time the next time you go looking:

* **There is no in-process LLM.** Nothing in this repo loads model
  weights. If you need to change generation parameters, you change
  what's *sent* to the upstream, not the upstream itself.
* **There is no `tokenizers` use at request time.** Length budgets are
  in tokens but enforced by the upstream; we only pass `max_tokens`.
* **There is no authentication framework.** The optional HTTP Basic
  middleware in `main.py` is the entire story. If you need OAuth /
  JWT / API-key headers, wrap the FastAPI app at the deployment level
  (nginx, traefik, etc.) — do not add it ad-hoc to individual routes.
* **There is no Celery / Redis / background queue.** Everything is in
  one process, one event loop. The cooperative-abort design depends on
  that.
* **There is no front-end in this repo.** The "front-end" mentioned in
  comments and docs is an external SPA that calls these HTTP / WS
  endpoints; this repo is the backend only.
