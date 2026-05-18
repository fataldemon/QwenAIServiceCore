# Embedding Data Schema

## On-disk layout

```
embedding/
└── <character>/
    ├── identity.json                # array of strings ("我是 xx", aliases for first-person rewrite)
    ├── setting/
    │   ├── *.mem                    # human-editable source text
    │   └── vector/
    │       ├── materials.jsonl      # one JSON object per FAISS row
    │       ├── index.faiss          # FAISS IndexFlatL2 over materials
    │       └── binaries/            # optional: bytes for non-text materials
    ├── knowledge/...                # same shape as setting/
    └── expression/...               # same shape as setting/ (emotion labels)
```

`subject` is one of `{setting, knowledge, expression}`. The shape is
identical; only the content differs.

## `materials.jsonl`

Each line is a self-describing record:

```jsonc
// A paragraph
{"id": 0, "kind": "para", "type": "text", "text": "天童爱丽丝是…", "tags": ["人物","主角"]}

// A non-text paragraph (e.g. image embedding for VLM lookup)
{"id": 5, "kind": "para", "type": "image", "text": "图片描述用于检索",
 "binary": "binaries/0005.png", "tags": []}

// A tag row -- its FAISS embedding is the *tag string itself*. When a
// query lands on a tag row, the search expands to target_ids.
{"id": 7, "kind": "tag", "type": "text", "text": "主角",
 "target_ids": [0, 1, 3]}
```

Invariants:

* `id` equals the row's FAISS position. Records are written in FAISS order.
* `kind` is `"para"` or `"tag"`. The legacy pickle layout had three files;
  we collapse them into one stream and tell paragraphs from tag rows via
  `kind` alone.
* For `kind == "tag"`, `text` is the tag string and `target_ids` lists the
  paragraph row ids that carry this tag.
* For `kind == "para"`, `tags` is informational only — the search uses
  the tag rows for tag-based expansion.
* `type` is currently always `"text"`; the field exists so we can add
  `"image"` / `"audio"` rows later without bumping the schema version.

## Migration from pickles

The legacy layout per subject was:

```
vector/
├── materials.pkl        # list[str], FAISS-aligned (paragraphs + tag names)
├── tags_map.pkl         # dict[str, list[int]]
├── paragraphs.pkl       # list[str], paragraphs only
└── index.faiss
```

`embedding/migrate.py` rebuilds `materials.jsonl` from these three files
while keeping the FAISS row order intact, so the existing `index.faiss`
keeps working without re-embedding. The pickles are renamed to `*.pkl.bak`
(never deleted). The migration is idempotent: if `materials.jsonl` already
exists for a subject, that subject is skipped.

Migration is invoked automatically from `main.py`'s lifespan handler. To
run it manually:

```bash
python -m embedding.migrate
```

## Rebuilding from `*.mem`

If you edit the source `.mem` files in `embedding/<character>/<subject>/`,
trigger a rebuild by calling `generate_vector(character, subject)` (or by
sending a `type=1` chat request, which appends to `knowledge.mem` and
incrementally updates the index in place).

`add_knowledge` performs an incremental append-and-extend:

1. parses the new chunk into paragraph dicts;
2. appends them to `materials.jsonl` (FAISS-ordered);
3. extends `index.faiss` with their embeddings (no rebuild);
4. attaches new paragraph ids to existing tag rows when the tag already
   exists, and creates fresh tag rows for previously-unseen tags.

If anything looks inconsistent (e.g. row count drift between `index.faiss`
and `materials.jsonl`), `add_knowledge` falls back to a full rebuild.

## Identity

`identity.json` is a flat list of strings used during query rewriting:

```json
["天童爱丽丝", "爱丽丝", "Arisu"]
```

Each match in the user's query is replaced with the literal "你" before
the query is embedded for the `setting` subject. This is a copy of the
legacy behaviour and matters because the setting paragraphs are written
in second person.
