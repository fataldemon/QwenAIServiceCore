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
| `abort_id` | 客户端自定义的取消键，POST 到 `/admin/api/abort/{abort_id}` 可取消。**流式与非流式两条路径都生效**，详见下面"协作式取消"。 |
| `enable_thinking` | 转给支持该参数的上游（Qwen 系），通过 `chat_template_kwargs.enable_thinking` 传递 |

## 协作式取消（abort）

旧 `main` 分支允许在非流式请求"飞行中"取消，并把"取消时已经生成的那一段"
作为普通 `ChatCompletionResponse` 返回给前端。`dev` 分支完整保留了这个契约。

行为：

* 非流式（`POST /assistant/v1/chat/completions`、不带 `stream=true` 的
  `POST /v1/chat/completions`、`WS /ws/{*}`）：当 POST
  `/admin/api/abort/{abort_id}` 进来时，飞行中的请求会就地短路，**原请求
  仍然按正常 `ChatCompletionResponse` 返回**，其中 `content` 是被打断那一
  刻已经产生的文本，`finish_reason` 为 `"abort"`。依赖该行为的前端可以
  原样工作，无需修改。
* 流式（`/v1/chat/completions` 带 `stream=true`）：SSE 流会最后发一帧
  `finish_reason="abort"`，然后 `data: [DONE]`。

实现说明：网关内部**始终用 SSE 与上游通信**（即使对外是非流式的客户端，
也是先 stream、后聚合），这是非流式也能中途取消的前提。

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

GET    /admin/api/personas
GET    /admin/api/personas/<character>
PUT    /admin/api/personas/<character>     -- body 为 Persona JSON（见下面 schema）
DELETE /admin/api/personas/<character>
POST   /admin/api/personas/<character>/preview
                                          -- body = {"user_text": "...", "information": "..."}

POST   /admin/api/abort/<abort_id>
```

## Admin Gradio 管理面板

`/admin` 上的 Gradio 界面包含以下标签页：

| 标签页 | 用途 |
|-----|---------|
| **LLM Providers** | 添加、编辑、激活、删除 OpenAI 兼容的服务商。 |
| **MCP Servers** | 配置 MCP 服务端连接（stdio 或 SSE）。 |
| **Skills** | `skills/<name>/SKILL.md` 形式的 Skill 列表与重载。 |
| **Characters** | Persona 编辑器（system 提示词、image_setting、reply_instruction）+ 知识库文件浏览器。 |
| **Shared Knowledge** | 跨角色共享的知识库。 |
| **Request Monitor** | 实时终端风格查看器，读取 `logs/vllm_request_log.jsonl`。展示完整请求载荷、采样参数、工具定义、响应文本、token 统计，以及可折叠展开的原始 SSE 事件。刷新周期可配置。 |

## Persona（角色人设）配置结构

每个角色一份系统提示词，存放于 ``embedding/<character>/persona.json``。
除 ``display_name`` 外字段均可选。

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

* ``setting`` 中可以包含 ``{embeddings}`` 占位符，``chat_on_setting`` 会
  把知识库召回结果填进去；如果没有占位符，召回结果会自动追加到
  ``setting`` 末尾。
* **没有 `persona.json` 的角色**会被视作"没有人设"——请求退化为普通补全，
  不注入角色框架。这是有意保留的"通用助手"行为。
* 旧版的"天童爱丽丝"提示词在首次启动后会被自动写入
  ``embedding/tendou_arisu/persona.json``（幂等），所以从旧版升级时
  现有角色的表现完全不变，不需要任何人工搬运。

在 admin 页面里新增一个角色：

1. 打开 ``/admin`` → **Personas** Tab；
2. 在 character 一栏填入新的目录名（比如 ``some_new_one``），编辑各
   字段后点 **Save / Update**。``setting/`` / ``knowledge/`` /
   ``expression/`` 三个子目录会自动建出来；
3. 把 ``.mem`` 知识/设定文件丢进对应子目录，然后重启服务让 embedding
   loader 把新文件吃进去。

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
