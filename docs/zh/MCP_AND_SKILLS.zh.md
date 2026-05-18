# MCP 与 Skill

## MCP 工具调用模式

通过 `POST /admin/api/mcp/mode` 全局设置：

* **`passthrough`**（默认）— 网关不向模型暴露任何 MCP 工具。如果前端
  本身就是 MCP 客户端（这也是项目最初的设计），它会继续负责发现工具、
  调用工具，并把工具返回值写进下一轮请求的消息列表里。这种模式下网关
  纯粹是 LLM 代理。
* **`server_side`** — 网关聚合所有启用的 MCP server 上的 `tools/list`，
  并把它们注入到上游请求。当模型返回 `tool_calls` 时，预期流程是：

  1. 解析全限定名 `<server>__<tool>`；
  2. 调用 `MCPManager.call_tool(...)`；
  3. 把结果以 `role="tool"` 写回消息列表；
  4. 继续循环直到模型返回普通文本。

  当前 `dev` 分支实现了 1–4 的前半部分：会广播工具列表，并把
  `function_call`/`tool_calls` 暴露在 `ChatCompletionResponseChoice`
  里（`finish_reason="function_call"`）。**完整闭环还没接入**，因为绝大多数
  线上部署使用 passthrough（前端已经会 MCP），server_side 目前需要客户端
  发起下一轮请求来推进。

## 添加 MCP Server

页面：`/admin` → MCP Servers Tab，填写 `name`、`transport`，
以及 `command`+`args`（stdio）或 `url`+`headers`（sse / streamable_http）。

REST：

```bash
curl -X PUT http://localhost:8000/admin/api/mcp/servers/filesystem \
  -H "Content-Type: application/json" \
  -d '{
        "enabled": true,
        "transport": "stdio",
        "command": "npx",
        "args": ["-y","@modelcontextprotocol/server-filesystem","/srv/data"],
        "description": "只读文件系统"
      }'
```

连接是按需建立的：第一次执行 `tools/list` 或 `call_tool` 时才连上，
`ClientSession` 在进程生命周期里复用（直到再次 `PUT` 或 `DELETE`
该 server，缓存会被作废）。

健康检查：`GET /admin/api/mcp/health`，返回每台 server 的
`{enabled, transport, connected, tools}`。

## Skill

一个 Skill 是 `skills/<name>/` 目录下的 `SKILL.md`，由 YAML front matter
+ Markdown 正文组成：

```markdown
---
name: example
description: "示例 Skill：解释 <think> 块"
version: "0.1.0"
auto_inject: false
triggers:
  keywords: ["think", "思考"]
  regex:
    - "<think>"
files: []
---
# 示例 Skill

正文是 Markdown，`read_skill("example")` 返回的就是这一段。
```

* 当 `auto_inject: true` 且 `triggers.keywords` / `triggers.regex` 命中
  最后一条用户消息时，正文会自动拼到 system 提示词里，**无需工具调用**。
* 无论是否启用 MCP，模型都可见两个虚拟工具：`list_skills` 与
  `read_skill(name)`。若要在 server_side 模式下接入，使用
  `SkillManager.virtual_tools()` 即可拿到工具定义。

Skill 会在启动时和 `POST /admin/api/skills/reload` 时扫描；**正文本身始终
按需从磁盘读取**，所以编辑 `SKILL.md` 后，下次 `read_skill` 调用就能拿到新
内容，不用重启或 reload。
