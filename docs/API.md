# HTTP API

## Completion routes

These are kept compatible with the legacy `main` branch. Only `stream=true`
on `/v1/chat/completions` is new.

### `POST /assistant/v1/chat/completions`

Analysis-style completion. No character framing, no embedding augmentation.
Request body is :class:`models.base.ChatCompletionRequest`. Response is
:class:`ChatCompletionResponse` with one :class:`ChatCompletionResponseChoice`.

### `POST /v1/chat/completions`

Character chat. Adds:

* knowledge-base retrieval (`setting` + `knowledge`) injected into the
  system prompt;
* emotion snapping on the assistant reply;
* WebSocket broadcast of the result to every `/ws/*` listener (legacy).

Set `stream=true` to switch to SSE. Streaming response format:

```
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"created":...}
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}],"created":...}
...
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"created":...}
data: [DONE]
```

### `WS /ws/{ws_mode}`

`ws_mode` ∈ `{"text", "binary"}` — controls how the client sends JSON. The
server echoes one `ChatCompletionResponse` per inbound request (quick-reply
budget, `max_quick_reply=600`).

### Request fields specific to this server

In addition to the OpenAI-style fields, :class:`ChatCompletionRequest`
accepts:

| Field | Meaning |
|--|--|
| `information` | Extra context string appended to the embedding system prompt. |
| `embeddings_buffer` | List of row ids — the front-end's view of the LRU buffer. The server merges this with fresh hits and returns the result in `choices[0].embedding_list`. |
| `on_embedding` | Set `false` to skip knowledge-base retrieval entirely. |
| `character` | Character folder under `embedding/`. Defaults to `tendou_arisu`. |
| `type` | `0` chat (default), `1` save assistant message as knowledge (then return `"ok"`), `2` long-term memory (currently treated as `0`). |
| `request_id` | Idempotency key; forwarded to the upstream. |
| `abort_id` | Client-chosen key. POST to `/admin/api/abort/{abort_id}` to cancel. Works on **both** streaming and non-streaming routes — see "Cooperative abort" below. |
| `enable_thinking` | Forwarded as `chat_template_kwargs.enable_thinking` to upstreams that understand it (Qwen). |

## Cooperative abort

The legacy `main` branch supported aborting an in-flight non-streaming
request and returning the partial answer that had been generated so far.
The `dev` branch preserves that contract.

Behaviour:

* Non-streaming (`POST /assistant/v1/chat/completions`,
  `POST /v1/chat/completions` without `stream=true`, `WS /ws/{*}`):
  upon `POST /admin/api/abort/{abort_id}` the in-flight call short-circuits
  and the original HTTP / WebSocket request **still returns a normal
  `ChatCompletionResponse`** whose `content` is whatever tokens had been
  produced and whose `finish_reason` is `"abort"`. Front-ends that rely on
  this behaviour continue to work unchanged.
* Streaming (`/v1/chat/completions` with `stream=true`): the SSE stream
  emits a terminal `data:` frame with `finish_reason="abort"`, followed by
  `data: [DONE]`.

Implementation note: internally the gateway always talks to the upstream
via SSE (even for non-streaming clients) and aggregates the chunks. This
is what makes mid-flight cancellation possible.

## Admin REST

All routes are JSON-in / JSON-out. Errors return 4xx with `{"detail": "..."}`.
See `admin/routes.py` for the full list. Highlights:

```
GET    /admin/api/providers
PUT    /admin/api/providers/<name>      -- body = ProviderConfig dict
POST   /admin/api/providers/<name>/activate
DELETE /admin/api/providers/<name>

GET    /admin/api/mcp/servers
PUT    /admin/api/mcp/servers/<name>    -- body = MCPServerConfig dict
DELETE /admin/api/mcp/servers/<name>
POST   /admin/api/mcp/mode              -- body = {"mode": "passthrough"|"server_side"}
GET    /admin/api/mcp/health

GET    /admin/api/skills
GET    /admin/api/skills/<name>
POST   /admin/api/skills/reload

GET    /admin/api/personas
GET    /admin/api/personas/<character>
PUT    /admin/api/personas/<character>     -- body = Persona JSON (see schema below)
DELETE /admin/api/personas/<character>
POST   /admin/api/personas/<character>/preview
                                          -- body = {"user_text": "...", "information": "..."}

POST   /admin/api/abort/<abort_id>
```

## Admin Gradio UI

The Gradio interface at `/admin` has these tabs:

| Tab | Purpose |
|-----|---------|
| **LLM Providers** | Add, edit, activate, delete OpenAI-compatible providers. |
| **MCP Servers** | Configure MCP server connections (stdio or SSE). |
| **Skills** | Discovered skills from `skills/<name>/SKILL.md`, with reload. |
| **Characters** | Persona editor (system prompt, image_setting, reply_instruction) + knowledge-base file browser. |
| **Shared Knowledge** | Cross-character knowledge base. |
| **Request Monitor** | Real-time terminal-style viewer of `logs/vllm_request_log.jsonl`. Shows full request payloads, sampling params, tools, response text, token counts, and raw SSE events (collapsible). Configurable refresh interval. |

## Persona config schema

Per-character system prompt; stored at
``embedding/<character>/persona.json``. All fields except
``display_name`` are optional.

```json
{
  "display_name":        "天童爱丽丝",
  "setting":             "你是爱丽丝……\n{embeddings}",
  "reply_instruction":   "\n回答规范：……",
  "image_setting":       "**你的形象设定**：\n……",
  "max_chat_len":        15000,
  "max_analysis_len":    6000,
  "max_quick_reply":     600,
  "default_temperature": 0.7,
  "default_top_p":       0.9
}
```

* ``setting`` may contain a ``{embeddings}`` placeholder; ``chat_on_setting``
  splices the knowledge-base retrieval result there. If the placeholder is
  absent, retrieved knowledge is appended at the end of ``setting`` instead.
* Characters **without** a ``persona.json`` are treated as having no
  persona — the request becomes a generic completion with no character
  framing. This is intentional: it lets you have a "no role-play" default
  while still keeping multiple characters around.
* The legacy "Tendou Arisu" prompt is seeded into
  ``embedding/tendou_arisu/persona.json`` on first boot (idempotent), so an
  upgraded deployment continues to behave exactly the same.

To add a new character via the admin UI:

1. Open ``/admin`` → **Personas** tab.
2. Type the folder name (e.g. ``some_new_one``), fill the fields,
   click **Save / Update**. The standard subfolders
   (``setting/``, ``knowledge/``, ``expression/``) are created automatically.
3. Drop ``.mem`` knowledge / settings files into those subfolders, then
   restart the service to let the embedding loader pick them up.

## Provider config schema

```json
{
  "type": "openai_compatible",
  "base_url": "http://localhost:8001/v1",
  "api_key": "EMPTY",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "supports_vision": true,
  "supports_audio": false,
  "supports_video": false,
  "prefetch_media": false,
  "extra_body": {"mm_processor_kwargs": {"fps": 2}},
  "description": "local vLLM"
}
```

`prefetch_media` makes the gateway download referenced files and embed them
as `data:` URIs — useful when the upstream cannot reach the file URL but
your gateway can. Off by default to keep payloads small.

## MCP server config schema

```json
{
  "enabled": true,
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"],
  "env": {"FOO": "bar"},
  "description": "filesystem (read-only)"
}
```

For remote transports:

```json
{
  "enabled": true,
  "transport": "sse",
  "url": "https://mcp.example.com/sse",
  "headers": {"Authorization": "Bearer ..."}
}
```
