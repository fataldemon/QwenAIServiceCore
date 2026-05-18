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
| `abort_id` | Client-chosen key. POST to `/admin/api/abort/{abort_id}` to cancel. |
| `enable_thinking` | Forwarded as `chat_template_kwargs.enable_thinking` to upstreams that understand it (Qwen). |

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

POST   /admin/api/abort/<abort_id>
```

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
