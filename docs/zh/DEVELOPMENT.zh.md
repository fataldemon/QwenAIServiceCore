# 开发维护文档

> 适用人群：**代码维护者**。如果你要找的是"如何安全部署 / 回滚 / 上下游
> 切换"，请看 [`MAINTENANCE.zh.md`](MAINTENANCE.zh.md)，那是运维向的文档，
> 与本文有意分开。

本文主要讲三件事：项目代码是怎么组织的、一个请求是怎么走完整个链路的，
以及——这部分篇幅最长——**想改某个功能的时候应该去看哪个文件**。

---

## 1. 项目结构

```
QwenAIServiceCore/
├── main.py                    FastAPI app，路由声明，lifespan。
├── webui.py                   挂在 /admin 上的 Gradio 后台 UI，
│                              （LLM Providers, MCP Servers, Skills,
│                              Characters, Shared Knowledge,
│                              Request Monitor 含原始 SSE 事件查看器）。
├── template.py                静态配置（CLI 参数、长度预算）+
│                              一个跟角色无关的提示词片段
│                              (REACT_INSTRUCTION)。
│                              人设字符串只以 LEGACY_ALICE_* 的形式
│                              留在这里，作为首启动的"种子数据"。
├── admin/
│   └── routes.py              /admin/api/* 的 JSON REST。
│
├── core/
│   ├── config_manager.py      Provider + MCP + 全局配置；文件持久化
│   │                          (config/*.json)，带原子写 + 锁。
│   ├── content_normalizer.py  多模态内容解析、GIF 抽帧、OpenAI 风格
│   │                          与旧版占位符之间的双向桥接。
│   ├── mcp_manager.py         MCP 会话池、list_tools()、call_tool()。
│   ├── persona_manager.py     每个角色一份 persona.json 的注册表，
│   │                          chat_on_setting 的 system 提示词来源。
│   └── skill_manager.py       Skill 发现（skills/<name>/SKILL.md）、
│                              自动注入、虚拟工具暴露。
│
├── llm/
│   ├── chat.py                ★ 编排层。HTTP / WS 处理函数最终都
│   │                          落到这里。
│   ├── backends/
│   │   ├── base.py            LLMBackend ABC + GenerationResult /
│   │   │                      StreamChunk。
│   │   ├── registry.py        get_backend(name) 缓存；
│   │   │                      invalidate(name)。
│   │   └── openai_compatible.py
│   │                          目前唯一一个具体实现。SSE-only —
│   │                          连"非流式"也是在流式之上聚合的，
│   │                          这样 abort 能在中途生效。
│   └── local_llm_manage.py    【死代码】保留作参考；原本是
│                              in-process vLLM 的启动逻辑。
│
├── embedding/
│   ├── embedding.py           对外 API：process_embedding()、
│   │                          add_knowledge()、check_emotion()。
│   ├── data_store.py          materials.jsonl + FAISS 索引 + binaries/。
│   └── migrate.py             启动时把旧 .pkl 一次性迁移到 jsonl。
│
├── models/
│   └── base.py                所有 pydantic schema（请求 / 响应）。
│
├── utils/                     旧的辅助函数，留作兼容。
├── config/                    运行时 JSON 配置（git 忽略）。
├── skills/<name>/SKILL.md     一个 skill 一个目录。
└── docs/                      本文档目录。
```

下面这些假设你可以放心依赖：

* **只有 `llm/chat.py` 调用 backends。** 路由层不直接调。
* **只有 `core/config_manager.py` 读写 `config/*.json`。** 其它代码
  通过 manager。
* **只有 `embedding/data_store.py` 读写 `materials.jsonl`。** 上层都
  走 `embedding/embedding.py`。

---

## 2. 请求数据流

### `POST /v1/chat/completions`（非流式）

```
main.py: create_chat_completion(request)
  └─> llm.chat.chat_on_setting(request, max_tokens, index=0)
        ├─> embedding.process_embedding(...)           # 知识库召回
        ├─> core.persona_manager.get_persona(character)
        │       └─> 拼到 system 提示词前缀
        ├─> _prepare_messages(...)                     # 内容归一化、
        │       └─> core.content_normalizer.*          # 剥掉上游不支持
        │                                              # 的模态、GIF 抽帧
        ├─> _gather_tools(req)                         # functions + MCP
        ├─> llm.backends.registry.get_backend()
        │     └─> OpenAICompatibleBackend.generate(...)
        │           └─> 内部走 generate_stream(...) 聚合
        └─> _postprocess_answer(text, character)       # 【表情】吸附
```

### `POST /v1/chat/completions` 带 `stream=true`

```
main.py: create_chat_completion(request)
  └─> sse_starlette.EventSourceResponse(
        llm.chat.chat_on_setting_stream(request, max_tokens, index=0)
      )
        ├─> embedding + persona + 内容归一化（同上）
        ├─> backend.generate_stream(...)               # AsyncIterator[StreamChunk]
        └─> 依次 yield ChatCompletionResponse(chunk, ...)
```

### `WS /ws/{ws_mode}`

```
main.py: websocket_endpoint
  └─> 每条入站 JSON：
        └─> chat_on_setting(request, max_tokens=max_quick_reply, index=1)
              -- 跟非流式一样的路径，token 预算更小
```

### 协作式取消

```
前端 -> POST /admin/api/abort/<abort_id>
       └─> main.py: admin_abort
             └─> llm.chat.abort_request(abort_id)
                   └─> backend.abort(request_id)
                         └─> 触发 asyncio.Event
                               └─> generate_stream 循环检测到，
                                     └─> 发出 StreamChunk(finish_reason="abort")
                                           └─> generate() 聚合器
                                                 break 并返回当前
                                                 累积到的半截文本
```

正是因为这条路径，`generate()` 才**不是**做成"一次性 `stream=false`
POST"，而是先 stream 再聚合——这样非流式请求被中途取消时还能保留旧契约：
返回正常的 `ChatCompletionResponse`，`content` 是半截内容，
`finish_reason` 为 `"abort"`。

---

## 3. "我要改 X，去哪里找？"

| 想做什么 | 改哪里 | 备注 |
|---|---|---|
| 加一个**兼容 OpenAI 协议**的新服务商 | `/admin/api/providers` UI 或直接编辑 `config/providers.json` | 不用改代码。 |
| 加一个**不兼容 OpenAI 协议**的新服务商（例如 Anthropic 原生协议 / Bedrock） | 在 `llm/backends/<name>.py` 新建文件实现 `LLMBackend`；在 `llm/backends/registry.py::_build_backend` 加一个按 `ProviderConfig.type` 分发的分支；如有新字段就改 `ProviderConfig`。 | 参照 `OpenAICompatibleBackend` 的形状。 |
| 改默认采样参数（temperature、top_p） | `llm/chat.py::_sampling_from_request` | 请求里的字段始终覆盖服务端默认值。 |
| 改 system 提示词模板 / 知识库注入方式 | `core/persona_manager.py`（schema）+ `llm/chat.py::_build_persona_system_prefix`（拼装逻辑） | 字符串本身在 `embedding/<character>/persona.json`，不在代码里。 |
| 新建 / 编辑某个角色的人设 | `/admin` → **Personas** Tab，或者直接编辑 `embedding/<character>/persona.json` | 下一次请求就生效，无需重启。 |
| 改 `【表情】` 吸附逻辑 | `llm/chat.py::_postprocess_answer`（解析）+ `embedding/embedding.py::check_emotion`（查表） | 两者解耦，解析层只关心语法。 |
| 改 `<think>...</think>` 切分逻辑 | `llm/chat.py::_split_thought_and_answer` | |
| 支持新模态 / 改 GIF 抽帧策略 | `core/content_normalizer.py` | `expand_gif_parts` 是 GIF 路径；新增模态在旁边加 `MediaPart` 类型。 |
| 知识库召回参数（top_k、max_length） | `llm/chat.py::chat_on_setting` 与 `chat_on_setting_stream` 里调 `process_embedding(...)` 时的参数 | |
| 改 `materials.jsonl` 字段 | `embedding/data_store.py`（schema）+ `embedding/migrate.py`（加一步迁移） | 记得 bump `SCHEMA_VERSION`。 |
| 加一个 `/admin/api/...` 接口 | 只动 `admin/routes.py`；实现放到 `core/` 或 `llm/`。 | 路由层故意做得很薄。 |
| 给 admin Gradio 后台加 Tab / 表单 | `webui.py` | UI 直接调 Python manager（不走自指 HTTP）。 |
| 调 abort 语义（比如顺手向上游发个 DELETE） | `llm/backends/openai_compatible.py::abort` | 不要动 event 注册，那是公开契约的支点。 |
| 新加 `/v1/...` 或 `/assistant/v1/...` 路由 | `main.py`：声明路由，调用 `llm.chat`。 | 路由不直接调 backend。 |
| Skill 自动注入规则 | `core/skill_manager.py::matches_user_text` | `_gather_tools` 和 `chat_on_setting` 消费结果。 |
| 加新的 MCP transport（比如 WebSocket） | `core/mcp_manager.py::_open_session` 加分发，再加一个小辅助类。 | |
| 加一个运行时可切换的开关 | `core/config_manager.py`（加字段 + 持久化）+ admin 路由 + UI。**不要**用 `os.environ`。 | |
| 改 WebSocket 协议 | `main.py::websocket_endpoint` + `utils/websocketutils.py` | |
| 剥掉 `<|endoftext|>` 之类的控制 token | `llm/chat.py::_postprocess_answer` | 已经在剥几个了，按需补。 |
| 给请求/响应日志增加字段 | `llm/chat.py::_append_vllm_request_log` | 日志 dict 在调用现场拼装 — 在那里加就好；同时更新 `webui.py::_format_vllm_request_log` 里的显示逻辑。 |
| 从上游 SSE 流中提取新字段 | `llm/backends/openai_compatible.py::generate_stream`（解析到 `StreamChunk`），`generate()` 汇总到 `GenerationResult.raw_events`。 | `raw_events` 会一路流到日志和 Request Monitor。 |
| 修改 `image_setting` 里图片路径的解析方式 | `llm/chat.py::_resolve_media_paths` | 相对路径会被拼接到 `embedding/<character>/image/` 下。 |

如果某个修改实在套不进上表，优先在 `core/` 下加个小模块，让 `llm/chat.py`
import 一下，而不是把 `main.py` 或某个 backend 越改越胖。

---

## 4. 约定与反模式

* **路由处理器里不写业务逻辑。** `main.py` 和 `admin/routes.py` 的
  每个端点最好不超过 5 行，全部委托给 manager 或 `llm/chat.py`。
* **Backend 不要直接读配置文件。** 它们应当通过 registry 拿到的
  `ProviderConfig`。这样换实现/换上游成本最低。
* **不要绕开 `_active_requests`。** abort 契约就靠它。如果你加一个
  新的"生成入口"，开头要注册 abort_id，`finally` 里 pop。
* **新加的运行时开关走 `core/config_manager.py`。** 不要用环境变量、
  不要用模块级全局变量。
* **跨 HTTP 边界的 schema 一律放 `models/base.py`。**
* **人设是数据，不是代码。** 不要再把 Alice / 柚子之类的字符串硬编码
  回 `template.py` 或 `llm/chat.py`，改 `embedding/<character>/persona.json`
  即可（直接改文件或者走 admin UI）。
* **`llm/local_llm_manage.py` 已冻结。** 不要往里加新行为，它只是
  旧 `main` 分支的代码参考。
* **每个 backend 都必须为 `GenerationResult` 填充 `raw_events`**，
  并为每个 `StreamChunk` 填充 `raw` 字段。约定是 `raw` 为上游的原始
  SSE 数据（JSON 对象），`raw_events` 为本次生成的**完整**事件列表。
  二者最终被 Request Monitor 和 `logs/vllm_request_log.jsonl` 消费。

---

## 5. 本地手测流程

不用搭单测框架，直接 round-trip：

```bash
# 1) 起一个 OpenAI 兼容的上游模型
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --host 127.0.0.1

# 2) 起网关
python main.py --server-port 8000 --server-name 127.0.0.1

# 3) 普通补全
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any",
        "messages": [{"role":"user","content":"老师你好"}],
        "character": "tendou_arisu"
      }' | jq

# 4) 流式
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
        "model": "any",
        "stream": true,
        "messages": [{"role":"user","content":"说一段长一点的话"}],
        "character": "tendou_arisu"
      }'

# 5) 中途取消。A 终端：
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any",
        "messages": [{"role":"user","content":"写一篇1000字作文"}],
        "character": "tendou_arisu",
        "abort_id": "smoke1"
      }'
# B 终端（A 还没返回时）：
curl -X POST http://127.0.0.1:8000/admin/api/abort/smoke1
# 期望：A 那边返回一个正常的 ChatCompletionResponse，
# content 是半截内容，finish_reason="abort"。

# 6) 角色提示词预览
curl -s -X POST http://127.0.0.1:8000/admin/api/personas/tendou_arisu/preview \
  -H "Content-Type: application/json" \
  -d '{"user_text":"老师你好"}' | jq -r .system_prompt | head -n 40
```

---

## 6. 这里**不存在**的东西

下次找东西之前先看看下面这几条，可以省你不少时间：

* **没有 in-process 大模型。** 本仓库不加载任何模型权重。要改"模型本身"
  的生成参数，是改**发给上游**的请求，不是改上游。
* **请求路径里没有 tokenizer。** 长度预算是 token 数没错，但全靠上游
  执行，我们只传 `max_tokens`。
* **没有完整的认证框架。** 整个仓库的认证就一个 HTTP Basic 中间件
  （`main.py`）。要 OAuth / JWT / API-key header 这类东西，建议在部署层
  （nginx / traefik）包一层，不要往单个路由里硬塞。
* **没有 Celery / Redis / 后台队列。** 整个服务就一个进程一个 event
  loop。协作式 abort 的设计就依赖这一点。
* **本仓库里没有前端。** 文档和注释里讲到的"前端"，指的是另一个外部
  SPA，它通过这些 HTTP / WS 接口调用后端；本仓库只负责后端。
