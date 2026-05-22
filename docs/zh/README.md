# QwenAIServiceCore — dev 分支（中文文档）

本目录提供与英文文档一一对应的中文版本，方便国内运维查阅。如果中英文表述
出现分歧，**以英文版本为准**（`README.md` 与 `docs/*.md`），中文版仅供阅读
辅助。

* [`MAINTENANCE.zh.md`](MAINTENANCE.zh.md) — 运维手册（回滚、升级、常见问题）
* [`API.zh.md`](API.zh.md) — HTTP 接口参考（聊天接口 + Admin 接口）
* [`MCP_AND_SKILLS.zh.md`](MCP_AND_SKILLS.zh.md) — MCP 与 Skill 模块说明
* [`EMBEDDING_DATA_SCHEMA.zh.md`](EMBEDDING_DATA_SCHEMA.zh.md) — 知识库数据结构

---

## 项目简介

QwenAIServiceCore 是一个 FastAPI 网关，对接 OpenAI 兼容的大模型推理服务
（通常是本地的 `vllm serve` 进程，或 DashScope / DeepSeek / OpenAI 等远端
服务商），同时保留了现有前端依赖的角色化能力：

* 按角色（character）的向量知识库检索（FAISS + `sentence-transformers`）
* 表情吸附（`【…】` → 最接近的已知表情）
* 多模态输入归一化（旧的 `[image,file=…]` 占位符与 OpenAI 风格的
  content 数组生成相同的上游请求）
* 运行时可热切换的多 Provider（不需要重启服务）
* MCP 工具集成（passthrough 或 server_side）
* `skills/<name>/SKILL.md` 形式的 Skill 模块
* **流式输出**：在请求体里加 `stream=true`，`/v1/chat/completions` 即返回
  SSE
* 协作式取消：请求里带 `abort_id`，调用
  `POST /admin/api/abort/{abort_id}` 即可取消

## 与 `main` 分支的差异

* 不再在进程内启动 vLLM `AsyncLLMEngine`。请单独运行 `vllm serve <model>`，
  然后在 Provider 配置里指向它即可。
* `llm/local_llm_manage.py` 仅保留作为参考代码，运行时不再使用，新的代码
  路径在 `llm/chat.py` + `llm/backends/`。
* 图像 / 音频 / 视频 / GIF 占位符统一在 `core/content_normalizer.py` 处理。
  大多数媒体以 URL 直接转发给 vLLM；GIF 由于 vLLM 不认 `.gif` 容器，会在
  网关侧用 Pillow 抽帧成 image 列表。
* 知识库的磁盘格式由 pkl 切换为 `materials.jsonl` + `binaries/`，
  详见 [`EMBEDDING_DATA_SCHEMA.zh.md`](EMBEDDING_DATA_SCHEMA.zh.md)。
  旧 pkl 数据会在首次启动时自动迁移，原文件改名为 `*.pkl.bak`，可随时回滚。
* 现有前端使用的全部路由形状保持不变：
  `POST /assistant/v1/chat/completions`、`POST /v1/chat/completions`、
  `WS /ws/{ws_mode}`。仅 `/v1/chat/completions` 多支持了 `stream=true`。

## 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 另开一个终端启动上游模型（示例：多模态 Qwen 在 8001 端口）
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --host 0.0.0.0

# 3) 首次启动会自动从 providers.example.json 生成 config/providers.json，
#    把激活 Provider 指向你的 vLLM；也可以等启动后在 /admin 页面里改。

# 4) 启动网关
python main.py --server-port 8000 --server-name 0.0.0.0
```

浏览器打开 <http://localhost:8000/admin> 进入 Gradio 管理面板
（LLM Providers / MCP Servers / Skills / Characters 五个 Tab，
外加 **Request Monitor** 可实时查看请求遥测和原始 SSE 事件）。
JSON 形式的管理接口在 `/admin/api/*`，详见
[`API.zh.md`](API.zh.md)。

## 流式输出

在 `/v1/chat/completions` 的请求体里加 `stream=true` 即可。响应为
`text/event-stream`，每条 `data:` 是一份
`ChatCompletionResponse(object="chat.completion.chunk")` 的 JSON，
最后以 `data: [DONE]` 结束。原有的非流式客户端无需修改（默认仍为
`stream=false`）。

## 配置文件

* `config/providers.json` — LLM Provider 列表与激活项
* `config/mcp_servers.json` — MCP Server 列表与工具调用模式
  （`passthrough` / `server_side`）

两个文件首次启动时会从同名的 `*.example.json` 拷贝生成，内含 API Key，
已在 `.gitignore` 中。

## 文档索引

* [`DEVELOPMENT.zh.md`](DEVELOPMENT.zh.md) — 开发维护文档：项目结构、
  请求数据流、以及"我要改 X 应该去看哪里"的速查表。
  **要改代码的话先看这一篇。**
* [`MAINTENANCE.zh.md`](MAINTENANCE.zh.md) — 运维清单：如何回滚到旧
  in-process 版本、如何安全地切上下游、加 Provider 等。
* [`API.zh.md`](API.zh.md) — HTTP API 参考（聊天 + 管理 + Persona schema）。
* [`MCP_AND_SKILLS.zh.md`](MCP_AND_SKILLS.zh.md) — MCP 服务器与 Skill
  模块的接入方式（如对应英文版存在）。
* [`../EMBEDDING_DATA_SCHEMA.md`](../EMBEDDING_DATA_SCHEMA.md) — 新版
  `materials.jsonl` 数据结构以及从旧 pickle 的迁移说明。
