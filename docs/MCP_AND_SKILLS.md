# MCP and Skills

## MCP tool-call modes

Set globally via `POST /admin/api/mcp/mode`:

* **`passthrough`** (default) — the gateway never lists MCP tools to the
  model. If the front-end is itself an MCP client (and the front-end is
  what manages the user's MCP servers), it should keep doing what it does
  today: discover tools, call them, and feed the results back to the model
  in subsequent turns. The gateway is a pure LLM proxy in this mode.
* **`server_side`** — the gateway aggregates `tools/list` across every
  enabled MCP server and injects them into the upstream call. When the
  model emits a `tool_calls` field, the gateway is expected to:

  1. parse the qualified name `<server>__<tool>`;
  2. call `MCPManager.call_tool(...)`;
  3. push the result back to the model as a `role="tool"` message;
  4. continue the loop until the model returns a plain answer.

  The current `dev` branch implements steps 1–4 partially — it advertises
  tools and surfaces the `function_call`/`tool_calls` in
  `ChatCompletionResponseChoice` (`finish_reason="function_call"`). Closing
  the loop end-to-end is a follow-up: the assumption is that most real
  deployments use `passthrough` because the chat front-end already speaks
  MCP. Server-side mode is wired up but expects the client to drive the
  follow-up turn for now.

## Adding an MCP server

Via UI: `/admin` → MCP Servers tab → fill in `name`, `transport`, and
either `command`+`args` (stdio) or `url`+`headers` (sse / streamable_http).

Via REST:

```bash
curl -X PUT http://localhost:8000/admin/api/mcp/servers/filesystem \
  -H "Content-Type: application/json" \
  -d '{
        "enabled": true,
        "transport": "stdio",
        "command": "npx",
        "args": ["-y","@modelcontextprotocol/server-filesystem","/srv/data"],
        "description": "read-only fs"
      }'
```

The connection is opened lazily on the first `tools/list` or `call_tool`,
and the `ClientSession` is cached for the rest of the process lifetime
(or until you re-`PUT` / `DELETE` the server, which invalidates the cache).

Health: `GET /admin/api/mcp/health` reports `{enabled, transport, connected,
tools}` per server.

## Skills

A skill is a folder under `skills/<name>/` containing `SKILL.md`. The file
has YAML front matter followed by a markdown body:

```markdown
---
name: example
description: "Demo skill that explains <think> blocks."
version: "0.1.0"
auto_inject: false
triggers:
  keywords: ["think", "思考"]
  regex:
    - "<think>"
files: []
---
# Example skill

Body markdown. This is what gets returned by `read_skill("example")`.
```

* When `auto_inject: true` and any of `triggers.keywords` /
  `triggers.regex` matches the latest user message, the body is appended
  to the system prompt automatically. **No tool call is needed.**
* Two virtual tools are always available to the model regardless of MCP
  configuration: `list_skills` and `read_skill(name)`. Surface them via
  `SkillManager.virtual_tools()` if you wire up server-side tool-calling.

Skills are loaded once at startup and on `POST /admin/api/skills/reload`.
The body itself is **always** read from disk on demand, so editing
`SKILL.md` takes effect on the next `read_skill` call without a reload.
