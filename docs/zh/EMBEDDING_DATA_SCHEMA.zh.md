# 知识库数据结构

## 目录布局

```
embedding/
└── <character>/
    ├── identity.json                # 字符串数组（"我是 xx" 之类的别名，用于第一人称改写）
    ├── setting/
    │   ├── *.mem                    # 人工编辑的源文本
    │   └── vector/
    │       ├── materials.jsonl      # 一行一条 FAISS 行的 JSON 元数据
    │       ├── index.faiss          # 基于 materials 的 FAISS IndexFlatL2
    │       └── binaries/            # 可选：非文本素材的二进制
    ├── knowledge/...                # 与 setting/ 结构一致
    └── expression/...               # 与 setting/ 结构一致（表情库）
```

`subject` 仅有 `{setting, knowledge, expression}` 三种取值，结构完全相同，
只是用途不同。

## `materials.jsonl`

每一行都是一条自描述记录：

```jsonc
// 段落
{"id": 0, "kind": "para", "type": "text", "text": "天童爱丽丝是…", "tags": ["人物","主角"]}

// 非文本段落（例如要给 VLM 查找的图片描述）
{"id": 5, "kind": "para", "type": "image", "text": "图片描述用于检索",
 "binary": "binaries/0005.png", "tags": []}

// 标签行 -- 它在 FAISS 里的 embedding 是"标签字符串"本身。当查询命中
// 一条标签行时，搜索会扩展到 target_ids 列出的段落。
{"id": 7, "kind": "tag", "type": "text", "text": "主角",
 "target_ids": [0, 1, 3]}
```

约束：

* `id` 与 FAISS 行号保持一致，所有记录按 FAISS 顺序写入。
* `kind` 取值 `"para"` 或 `"tag"`。旧的 pkl 布局把这两类拆成 3 个文件，
  我们合并为同一份流，通过 `kind` 区分。
* `kind == "tag"` 时，`text` 是标签字符串，`target_ids` 是带该标签的
  段落 id 列表。
* `kind == "para"` 时的 `tags` 只是元信息，检索时真正用于标签扩展的是
  独立的标签行。
* `type` 目前只用 `"text"`，预留这个字段是为了将来引入
  `"image"` / `"audio"` 等非文本行时不改 schema 版本。

## 从 pkl 迁移

旧布局是：

```
vector/
├── materials.pkl        # list[str]，FAISS 行序对齐（段落 + 标签名）
├── tags_map.pkl         # dict[str, list[int]]
├── paragraphs.pkl       # list[str]，仅段落
└── index.faiss
```

`embedding/migrate.py` 会从这三个 pkl 复原 `materials.jsonl`，FAISS 行序
完全保留，所以 `index.faiss` 不用重算。原 pkl 文件改名为 `*.pkl.bak`，**绝不
会删除**。迁移幂等：如果某个 subject 的 `materials.jsonl` 已存在，会跳过。

`main.py` 的 lifespan 钩子里会自动跑一次。也可以手动跑：

```bash
python -m embedding.migrate
```

## 从 `*.mem` 重建索引

如果你直接编辑了 `embedding/<character>/<subject>/` 下的 `.mem` 文件，
调用 `generate_vector(character, subject)` 可以重建（或者发一次 `type=1`
的聊天请求，由 `add_knowledge` 增量更新，它会把内容追加到 `knowledge.mem`
并在 FAISS 索引上 in-place 扩展）。

`add_knowledge` 的增量逻辑：

1. 解析新增内容为段落列表；
2. 按 FAISS 行序追加到 `materials.jsonl`；
3. 对 `index.faiss` 调用 `add(embeddings)`，不重建；
4. 已存在的标签：把新段落 id 追加进其 `target_ids`；新标签：建一条新的
   标签行。

如果状态发现不一致（比如 `index.faiss` 行数与 `materials.jsonl` 行数对
不上），`add_knowledge` 会回退为完整重建。

## Identity

`identity.json` 是一个字符串数组，用于查询改写：

```json
["天童爱丽丝", "爱丽丝", "Arisu"]
```

针对 `setting` 这个 subject 的查询，会把命中任意一项的子串先替换成 "你"
再做 embedding。这是为了对齐旧实现里"setting 段落用第二人称写"的事实，
保持检索一致性。
