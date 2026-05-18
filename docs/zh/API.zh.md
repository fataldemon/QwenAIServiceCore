# HTTP 接口

## 聊天接口

与 `main` 分支完全兼容。新增的只有 `/v1/chat/completions` 上的
`stream=true`。

### `POST /assistant/v1/chat/completions`

分析型补全。不做角色设定注入，也不做知识库召回。请求体为
`models.base.ChatCompletionRequest`，响应为 `ChatCompletionResponse`，包含
一个 `ChatCompletionResponseChoice`。

### `POST /v1/chat/completions`

角色聊天。在分析型补全之上叠加：

* 知识库召回（`setting` + `knowledge`）拼到 system 提示词；
* 对回复里的 `【…】` 做表情吸附；
* 把结果通过 WebSocket 广播给所有 `/ws/*` 监听者（旧逻辑）。

请求体带 `stream=true` 即切换为 SSE：

```
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"created":...}
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}],"created":...}
...
data: {"model":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"created":...}
data: [DONE]
```

### `WS /ws/{ws_mode}`

`ws_mode` 取值 `"text"` 或 `"binary"`，控制客户端发 JSON 的方式。
服务端按每个入站请求回一份 `ChatCompletionResponse`，最大 token 数为
`max_quick_reply=600`。

### 本服务特有的请求字段

除了 OpenAI 风格字段之外，`ChatCompletionRequest` 还支持：

| 字段 | 含义 |
|--|--|
| `information` | 附加在知识库系统提示词后的额外上下文 |
| `embeddings_buffer` | 前端的 LRU 索引视图，服务端会把新的命中合并后通过 `choices[0].embedding_list` 回写 |
| `on_embedding` | `false` 时完全跳过知识库召回 |
| `character` | 对应 `embedding/` 下的角色目录，默认 `tendou_arisu` |
| `type` | `0` 普通聊天（默认）；`1` 把最后一条 assistant 消息保存为新的知识，返回 `"ok"`；`2` 长期记忆（当前等同 `0`） |
| `request_id` | 幂等键，原样转发给上游 |
| `abort_id` | 客户端自定义的取消键，POST 到 `/admin/api/abort/{abort_id}` 可取消 |
| `enable_thinking` | 转给支持该参数的上游（Qwen 系），通过 `chat_template_kwargs.enable_thinking` 传递 |

## Admin REST

JSON in / JSON out，错误以 4xx + `{"detail": "..."}` 返回。
完整路由见 `admin/routes.py`，下面是要点：

```
GET    /admin/api/providers
PUT    /admin/api/providers/<name>      -- body 为 ProviderConfig
POST   /admin/api/providers/<name>/activate
DELETE /admin/api/providers/<name>

GET    /admin/api/mcp/servers
PUT    /admin/api/mcp/servers/<name>    -- body 为 MCPServerConfig
DELETE /admin/api/mcp/servers/<name>
POST   /admin/api/mcp/mode              -- body 为 {"mode": "passthrough"|"server_side"}
GET    /admin/api/mcp/health

GET    /admin/api/skills
GET    /admin/api/skills/<name>
POST   /admin/api/skills/reload

POST   /admin/api/abort/<abort_id>
```

## Provider 配置结构

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
  "description": "本地 vLLM"
}
```

`prefetch_media`：当上游访问不了外链但网关可以时，开启后网关会把媒体
下载下来转成 `data:` URI 一起发上去；默认关闭以减少请求体大小。

## MCP Server 配置结构

```json
{
  "enabled": true,
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"],
  "env": {"FOO": "bar"},
  "description": "只读文件系统"
}
```

远程传输：

```json
{
  "enabled": true,
  "transport": "sse",
  "url": "https://mcp.example.com/sse",
  "headers": {"Authorization": "Bearer ..."}
}
```
