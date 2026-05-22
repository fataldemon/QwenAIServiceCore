# QwenAIServiceCore — dev branch

> 中文文档见 [`docs/zh/README.md`](docs/zh/README.md)。

A FastAPI gateway that fronts an OpenAI-compatible LLM service (typically a
local `vllm serve` process, or any remote provider such as DashScope /
DeepSeek / OpenAI) and adds character-aware features that the existing
front-end already depends on:

* per-character knowledge-base retrieval (FAISS over `sentence-transformers`)
* emotion snapping (`【...】` → the closest known expression)
* multimodal input normalisation (legacy `[image,file=...]` placeholders **and**
  OpenAI-style content arrays produce the same upstream payload)
* runtime configurable LLM providers (no service restart)
* MCP tool integration (passthrough or server-side)
* skill modules under `skills/<name>/SKILL.md`
* **streaming** completions over SSE under the existing `/v1/chat/completions`
  route — set `stream=true` in the request body
* cooperative abort (set `abort_id` on the request, POST it to
  `/admin/api/abort/{abort_id}`)

## What changed compared to `main`

* The in-process vLLM `AsyncLLMEngine` is no longer started. Run
  `vllm serve <model>` separately and add a provider that points at it.
* `llm/local_llm_manage.py` is kept as dead reference code only; the live
  code path lives in `llm/chat.py` + `llm/backends/`.
* Image / audio / video / GIF placeholders are now handled in
  `core/content_normalizer.py`. Most media is forwarded to vLLM by reference;
  GIFs are expanded locally to image frames via Pillow because vLLM does not
  treat `.gif` as a video container.
* Knowledge-base on-disk format is now `materials.jsonl` + `binaries/` (see
  `docs/EMBEDDING_DATA_SCHEMA.md`). Existing pickle data is migrated
  automatically on first boot — original `*.pkl` files are renamed to
  `*.pkl.bak` so you can roll back at any time.
* All routes the existing front-end calls are unchanged in shape:
  `POST /assistant/v1/chat/completions`, `POST /v1/chat/completions`,
  `WS /ws/{ws_mode}`. The only difference is that `/v1/chat/completions` now
  also accepts `stream=true`.

## Quick start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Start the upstream model (in a separate terminal)
#    Example: a multimodal Qwen on localhost:8001
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --host 0.0.0.0

# 3) Edit config/providers.json (auto-created from providers.example.json
#    on first run) so that the active provider points at your vLLM,
#    or use the admin UI at /admin after step 4.

# 4) Run the gateway
python main.py --server-port 8000 --server-name 0.0.0.0
```

Open <http://localhost:8000/admin> for the Gradio admin UI (LLM providers,
MCP servers, skills, characters, and a **Request Monitor** with live
request/response telemetry and raw SSE event viewer). The JSON admin API
is at `/admin/api/*` — see `docs/API.md`.

## Streaming

Set `stream=true` in the request body to `/v1/chat/completions`. The response
is an `text/event-stream` whose `data:` frames are
`ChatCompletionResponse(object="chat.completion.chunk")` JSON documents,
terminated by `data: [DONE]`. Existing non-streaming clients keep working
unchanged (the default is `stream=false`).

## Configuration files

* `config/providers.json` — LLM providers and the active one.
* `config/mcp_servers.json` — MCP servers + tool-call mode
  (`passthrough` or `server_side`).

Both files are auto-seeded from their `*.example.json` siblings on first
boot. They contain API keys and are git-ignored.

## Documentation

* [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — developer guide:
  repository layout, request data flow, and a "I want to change X — where
  do I look?" cheat sheet. **Start here if you're going to modify code.**
* [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) — operational checklist:
  rolling back to the legacy in-process branch, adding providers, etc.
* [`docs/API.md`](docs/API.md) — HTTP API reference (completion + admin +
  persona schema).
* [`docs/MCP_AND_SKILLS.md`](docs/MCP_AND_SKILLS.md) — how MCP servers and
  skills plug in.
* [`docs/EMBEDDING_DATA_SCHEMA.md`](docs/EMBEDDING_DATA_SCHEMA.md) — the
  new `materials.jsonl` schema and migration from pickles.
