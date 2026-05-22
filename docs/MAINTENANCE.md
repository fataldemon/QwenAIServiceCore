# Maintenance Notes

## Branch layout

* `main` — legacy single-process build with in-process vLLM. Production
  baseline; touch only for hotfixes.
* `dev` — current branch. External `vllm serve` + multi-provider gateway +
  admin UI + streaming. All new work lands here.

To roll back to the legacy behaviour, check out `main` and restart. Note
that the legacy branch reads `materials.pkl` / `tags_map.pkl` /
`paragraphs.pkl`, which the `dev` branch renames to `*.pkl.bak` on first
boot. Rename them back if you need legacy mode and were running `dev` first:

```bash
cd embedding/<character>/<subject>/vector
for f in *.pkl.bak; do mv "$f" "${f%.bak}"; done
```

## Running upstream LLMs

The gateway is purely a client. Start the upstream model out-of-band:

```bash
# Local vLLM, multimodal Qwen
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 --max-model-len 32768

# Local vLLM, text-only DeepSeek
vllm serve deepseek-ai/DeepSeek-V3 --port 8002
```

The gateway is configured per-provider via `config/providers.json` (or the
admin UI). Switching the active provider takes effect immediately — no
gateway restart required.

## Embedding model

`sentence-transformers` is loaded lazily on first knowledge-base operation.
Override the device with `EMBEDDING_DEVICE=cpu` if you don't have a GPU on
the gateway host; the default is `auto` (CUDA if available, else CPU).

## Logs

The gateway writes two structured JSONL logs under ``logs/``:

* ``logs/chat_log.jsonl`` — lightweight conversation records (user text,
  assistant reply, thought, finish_reason, timestamps). Auto-truncated at
  ~10 MB (oldest 25% dropped).
* ``logs/vllm_request_log.jsonl`` — full request/response telemetry.
  Cleared on every startup. Each entry contains: character, provider,
  model, base_url, the complete messages array, sampling parameters,
  tools, extra_body, the generated answer, thought, raw SSE events
  (``raw_events``), function_calls, and token counts. This is what the
  **Request Monitor** tab in ``/admin`` displays.

Both files are human-readable JSONL (one JSON object per line).

The default `uvicorn` logging is fine for development. For production set
`LOG_LEVEL=INFO` and direct stdout/stderr to your usual log shipper. The
admin REST API logs every `upsert_*` / `delete_*` / `set_*` at `INFO`.

## Common problems

| Symptom | Likely cause | Fix |
|--|--|--|
| `503` from `/v1/chat/completions` | active provider URL unreachable | hit `GET /admin/api/providers`, fix base_url, or `POST /admin/api/providers/<other>/activate` |
| MCP server stays disconnected | `mcp` package missing, or bad command | `pip install mcp`; check `command`+`args` in the admin UI |
| Streaming hangs in front-end | front-end not parsing SSE | confirm `Content-Type: text/event-stream` is reaching the client; some reverse proxies buffer SSE — disable buffering |
| Knowledge results stale after editing `*.mem` | index not rebuilt | call `generate_vector(character, subject)` in a Python shell, or POST a `type=1` chat request which calls `add_knowledge` |

## Single-process assumption

`ConfigManager`, `MCPManager` and `SkillManager` are process-singletons.
Run the gateway with `--workers 1` (the default). If you ever scale out,
externalise the config files (NFS / object store) and the MCP pool will need
to become per-worker.
