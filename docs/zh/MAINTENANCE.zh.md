# 运维手册

## 分支策略

* `main` — 旧版本，使用进程内 vLLM，生产基线，仅修热修。
* `dev` — 当前分支。使用外部 `vllm serve` + 多 Provider 网关 + 管理面板 +
  流式输出。新功能都在这条分支上。

如果需要回退到旧实现，切回 `main` 重启即可。注意：`dev` 分支会在首次启动
时把 `materials.pkl` / `tags_map.pkl` / `paragraphs.pkl` 改名为
`*.pkl.bak`；如需让旧实现继续工作，请把它们改名回去：

```bash
cd embedding/<character>/<subject>/vector
for f in *.pkl.bak; do mv "$f" "${f%.bak}"; done
```

## 启动上游 LLM

网关本身不部署模型，请单独启动上游：

```bash
# 本地 vLLM，多模态 Qwen
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 --max-model-len 32768

# 本地 vLLM，纯文本 DeepSeek
vllm serve deepseek-ai/DeepSeek-V3 --port 8002
```

Provider 在 `config/providers.json` 或 `/admin` 页面里配置。切换激活的
Provider **立即生效**，不需要重启网关。

## Embedding 模型

`sentence-transformers` 在第一次执行知识库操作时才会加载。如果网关机器
没有 GPU，设置 `EMBEDDING_DEVICE=cpu`；默认是 `auto`（有 CUDA 用 CUDA，
否则用 CPU）。

## 日志

网关写入两份结构化的 JSONL 日志，均位于 ``logs/`` 目录下：

* ``logs/chat_log.jsonl`` — 精简的对话记录（用户输入、助手回复、思考过程、
  finish_reason、时间戳）。文件大小超过 ~10 MB 时自动截断（丢弃最早 25% 的行）。
* ``logs/vllm_request_log.jsonl`` — 完整的请求/响应遥测数据。
  每次启动清空。每条记录包含：角色名、provider、模型名、base_url、完整的
  messages 数组、采样参数、工具定义、extra_body、生成的回答、思考过程、
  原始 SSE 事件（`raw_events`）、工具调用、token 统计。**Request Monitor**
  标签页（`/admin`）正是读取此文件来展示的。

两份文件都是人类可读的 JSONL 格式（每行一个 JSON 对象）。

开发期使用 uvicorn 默认日志即可。生产环境建议 `LOG_LEVEL=INFO`，把
stdout/stderr 接到现有日志收集系统。`/admin/api/` 的所有写操作
（upsert / delete / set 等）默认 INFO 级别。

## 常见问题

| 现象 | 可能原因 | 处理 |
|--|--|--|
| `/v1/chat/completions` 返回 503 | 激活 Provider 的 base_url 不通 | `GET /admin/api/providers`，改 base_url，或 `POST /admin/api/providers/<name>/activate` 切换 |
| MCP Server 始终未连接 | 缺少 `mcp` 包，或 command 错误 | `pip install mcp`；在管理面板里核对 command / args |
| 前端流式接收卡住 | 前端没处理 SSE | 确认响应头里有 `Content-Type: text/event-stream` 到达前端；某些反向代理会缓冲 SSE，需要关掉缓冲 |
| 改完 `*.mem` 检索没变 | 索引未重建 | 在 Python 里调用 `generate_vector(character, subject)`，或者发一次 `type=1` 的聊天请求，由 `add_knowledge` 增量更新 |

## 单进程假设

`ConfigManager`、`MCPManager`、`SkillManager` 都是进程级单例，运行时请用
`--workers 1`（默认）。如要横向扩缩，需要把配置文件改放共享存储（NFS /
对象存储），MCP 连接池也要改成 per-worker 维护。
