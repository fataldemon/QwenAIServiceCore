"""Knowledge-base retrieval & maintenance.

Public surface (called from the chat code path):

* :func:`process_embedding`  -- find setting + knowledge for a query
* :func:`vector_search`      -- low-level FAISS search returning text + ids
* :func:`reorganize_index`   -- LRU-style merge of client/server index lists
* :func:`check_emotion`      -- snap an emotion label to the nearest known one
* :func:`add_knowledge`      -- incremental append + index update
* :func:`generate_vector`    -- full-rebuild of one subject's FAISS index
* :func:`find_material_by_index` -- lookup text payload by FAISS row id

Compared to the legacy implementation this module:

1. Uses the explicit JSONL schema from :mod:`embedding.data_store` instead of
   three pickles.
2. Performs cross-process safety via :func:`os.replace` for index writes; the
   previous ``fcntl``-based lock was POSIX-only and the FastAPI deployment is
   single-process anyway.
3. Loads ``sentence_transformers`` lazily so that ``import embedding.embedding``
   stays cheap during unit tests / migrations.
4. Hides absolute device placement (``cuda``/``cpu``) behind ``EMBEDDING_DEVICE``
   so it boots on a developer laptop without a GPU.

The retrieval semantics intentionally match the legacy behaviour (tag-rows
expand to their target paragraph ids; ``top_k`` is padded with the first
valid index when results are short) so chat output stays bit-for-bit
identical for unchanged inputs.
"""

from __future__ import annotations

import os
import re
import logging
from typing import List, Optional, Tuple

import numpy as np

from .data_store import (
    MaterialRecord,
    VALID_SUBJECTS,
    binaries_dir,
    build_paragraph_records,
    ensure_subject_dirs,
    identity_path,
    index_path,
    load_materials,
    load_identity,
    save_materials,
    subject_dir,
    vector_dir,
)

LOG = logging.getLogger(__name__)

# Legacy ``.mem`` source directory (still used by ``add_knowledge`` so the raw
# text remains human-editable on disk and version-controllable).
DOC_FOLDER_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "embedding",
    "{character}",
    "{subject}",
)


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


_model = None
_faiss = None


def _get_model():
    """Lazily instantiate the SentenceTransformer model."""
    global _model
    if _model is not None:
        return _model
    # Imports kept inside so ``import embedding.embedding`` is cheap.
    from sentence_transformers import SentenceTransformer  # type: ignore

    device = os.environ.get("EMBEDDING_DEVICE", "auto").lower()
    if device == "auto":
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # The embedding-model path is resolved by ``template._get_args``. We call
    # it lazily to avoid a circular import at module load time.
    from template import _get_args

    args = _get_args()
    model_path = args.embedding_path
    LOG.info("Loading SentenceTransformer %s on device=%s", model_path, device)
    _model = SentenceTransformer(model_path, device=device)
    return _model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore

        _faiss = faiss
    return _faiss


# ---------------------------------------------------------------------------
# Reference-URL hygiene (kept for compatibility with the old prompt format)
# ---------------------------------------------------------------------------


def remove_reference_url(text: str) -> str:
    """Strip trailing ``<reference_url:...>`` tags added by some clients."""
    return re.sub(r"<reference_url:[^>]*>", "", text).strip()


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def read_as_content(file_name: str, doc_folder: str) -> str:
    with open(os.path.join(doc_folder, file_name), "r", encoding="utf-8") as f:
        return f.read()


def write_as_memory(file_name: str, doc_folder: str, content: str) -> None:
    os.makedirs(doc_folder, exist_ok=True)
    with open(os.path.join(doc_folder, file_name), "a", encoding="utf-8") as f:
        if not content.endswith("\n"):
            content += "\n"
        f.write(content)


# ---------------------------------------------------------------------------
# Index (re)build
# ---------------------------------------------------------------------------


def _gather_paragraphs_from_mem(doc_folder: str) -> List[dict]:
    """Read all ``.mem`` files in ``doc_folder`` and parse them into dicts.

    Each ``.mem`` file is line-oriented; blank lines split paragraphs. Each
    paragraph may end with ``##tag1##tag2`` to attach tags.
    """
    if not os.path.isdir(doc_folder):
        return []
    parts: List[str] = []
    for name in sorted(os.listdir(doc_folder)):
        if name.endswith(".mem"):
            try:
                parts.append(read_as_content(name, doc_folder))
            except OSError:
                continue
    text = "\n".join(parts)
    paragraphs: List[dict] = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if "##" in s:
            head, *tags = s.split("##")
            tags = [t.strip() for t in tags if t.strip()]
            # De-duplicate tags within a paragraph.
            seen, dedup = set(), []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            paragraphs.append({"text": head.strip(), "tags": dedup, "type": "text"})
        else:
            paragraphs.append({"text": s, "tags": [], "type": "text"})
    return paragraphs


def generate_vector(character: str, subject: str) -> str:
    """Full rebuild of the FAISS index for ``(character, subject)``.

    Reads ``*.mem`` files in the subject folder, derives paragraph + tag rows,
    embeds them and writes ``materials.jsonl`` + ``index.faiss``.
    Returns ``"success"`` / ``"empty"`` / ``"error"`` / ``"subject incorrect"``.
    """
    if subject not in VALID_SUBJECTS:
        return "subject incorrect"

    doc_folder = DOC_FOLDER_TEMPLATE.format(character=character, subject=subject)
    ensure_subject_dirs(character, subject)
    vdir = vector_dir(character, subject)

    paragraphs = _gather_paragraphs_from_mem(doc_folder)
    if not paragraphs:
        # Clean up old index files so future searches do not match stale data.
        for f in ("materials.jsonl", "index.faiss"):
            fp = os.path.join(vdir, f)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        return "empty"

    records = build_paragraph_records(paragraphs)
    search_materials = [r.text for r in records]

    try:
        embeddings = np.asarray(_get_model().encode(search_materials), dtype="float32")
    except Exception as e:
        LOG.exception("Embedding encode failed for %s/%s: %r", character, subject, e)
        return "error"

    faiss = _get_faiss()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    save_materials(character, subject, records)
    # Write the FAISS index atomically.
    tmp_index = index_path(character, subject) + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, index_path(character, subject))
    return "success"


def _materials_count_in_index(character: str, subject: str) -> int:
    """Return ``index.ntotal`` if the index exists, else ``0``."""
    p = index_path(character, subject)
    if not os.path.exists(p):
        return 0
    try:
        idx = _get_faiss().read_index(p)
        return int(idx.ntotal)
    except Exception:
        return 0


def add_knowledge(content: str, character: str) -> str:
    """Append a knowledge chunk to ``knowledge.mem`` and update FAISS in place.

    Falls back to :func:`generate_vector` if the existing on-disk state is
    inconsistent (e.g. index row count != JSONL row count).
    """
    content = remove_reference_url(content)
    subject = "knowledge"
    doc_folder = DOC_FOLDER_TEMPLATE.format(character=character, subject=subject)
    ensure_subject_dirs(character, subject)
    write_as_memory("knowledge.mem", doc_folder, content)

    materials_existing = load_materials(character, subject)
    if not materials_existing or _materials_count_in_index(character, subject) != len(
        materials_existing
    ):
        return generate_vector(character, subject)

    # Parse the new chunk into paragraph dicts.
    parsed: List[dict] = []
    for raw in content.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if "##" in s:
            head, *tags = s.split("##")
            tags = [t.strip() for t in tags if t.strip()]
            seen, dedup = set(), []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            parsed.append({"text": head.strip(), "tags": dedup, "type": "text"})
        else:
            parsed.append({"text": s, "tags": [], "type": "text"})
    if not parsed:
        return "success"

    # Append new paragraph rows after the existing ones (FAISS-order preserved).
    next_id = len(materials_existing)
    new_para_records: List[MaterialRecord] = []
    new_tag_rows: dict = {}  # tag -> list of paragraph ids (only new)

    # Existing tag rows: we need to know which tags already have a row, so new
    # paragraphs can attach to them without adding a duplicate tag row.
    existing_tag_to_row: dict = {r.text: r.id for r in materials_existing if r.kind == "tag"}

    for p in parsed:
        rec = MaterialRecord(
            id=next_id,
            kind="para",
            type="text",
            text=p["text"],
            tags=list(p["tags"]),
        )
        new_para_records.append(rec)
        for tag in p["tags"]:
            if tag in existing_tag_to_row:
                # We'll patch the existing tag row's target_ids below.
                continue
            new_tag_rows.setdefault(tag, []).append(next_id)
        next_id += 1

    # Patch existing tag rows in-place.
    materials_patched: List[MaterialRecord] = []
    for r in materials_existing:
        if r.kind == "tag":
            adds = []
            for p_rec in new_para_records:
                if r.text in p_rec.tags:
                    adds.append(p_rec.id)
            if adds:
                rr = MaterialRecord(**{**r.__dict__, "target_ids": list(r.target_ids) + adds})
                materials_patched.append(rr)
            else:
                materials_patched.append(r)
        else:
            materials_patched.append(r)

    # Build brand-new tag rows for tags that did not exist before.
    fresh_tag_records: List[MaterialRecord] = []
    for tag, target_ids in new_tag_rows.items():
        # Also include any new paragraph ids in this chunk that mention the
        # tag (already collected in target_ids).
        fresh_tag_records.append(
            MaterialRecord(
                id=next_id,
                kind="tag",
                type="text",
                text=tag,
                target_ids=list(target_ids),
            )
        )
        next_id += 1

    # Embed and append to FAISS.
    new_rows_for_index: List[str] = [r.text for r in new_para_records] + [
        r.text for r in fresh_tag_records
    ]
    if new_rows_for_index:
        try:
            embeddings = np.asarray(_get_model().encode(new_rows_for_index), dtype="float32")
        except Exception as e:
            LOG.exception("Embedding encode failed during add_knowledge: %r", e)
            return "error"
        faiss = _get_faiss()
        idx = faiss.read_index(index_path(character, subject))
        idx.add(embeddings)
        tmp_index = index_path(character, subject) + ".tmp"
        faiss.write_index(idx, tmp_index)
        os.replace(tmp_index, index_path(character, subject))

    save_materials(
        character,
        subject,
        materials_patched + new_para_records + fresh_tag_records,
    )
    return "success"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def get_identity(character: str) -> List[str]:
    identities = load_identity(character)
    print(identities)
    return identities


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def _load_index_and_materials(
    character: str, subject: str
) -> Optional[Tuple[object, List[MaterialRecord]]]:
    materials = load_materials(character, subject)
    p = index_path(character, subject)
    if not materials or not os.path.exists(p):
        return None
    try:
        idx = _get_faiss().read_index(p)
    except Exception:
        return None
    if int(idx.ntotal) != len(materials):
        # Inconsistent state -- caller should treat as "no results".
        LOG.warning(
            "Index/materials length mismatch for %s/%s (%d vs %d)",
            character,
            subject,
            int(idx.ntotal),
            len(materials),
        )
        return None
    return idx, materials


def vector_search(
    question: str, top_k: int, character: str, subject: str, instruct: str
) -> Tuple[List[str], List[int]]:
    """FAISS top-k search returning ``(texts, paragraph_row_ids)``."""
    loaded = _load_index_and_materials(character, subject)
    if loaded is None:
        return [""], [0]
    idx, materials = loaded

    question = get_detailed_instruct(instruct, question)
    if subject == "setting":
        for identity in get_identity(character):
            if identity:
                question = question.replace(identity, "你")

    search = np.asarray(_get_model().encode([question]), dtype="float32")
    n_total = int(idx.ntotal)
    search_k = max(1, min(top_k * 3 + 1, n_total))
    accuracy, matches = idx.search(search, search_k)
    LOG.debug("vector_search %s/%s: %s %s", character, subject, accuracy, matches)

    result_index_list: List[int] = []
    log_info = ""
    for raw_i in matches[0]:
        i = int(raw_i)
        if i < 0 or i >= len(materials):
            continue
        rec = materials[i]
        if rec.kind == "tag":
            for j in rec.target_ids:
                if 0 <= j < len(materials) and j not in result_index_list:
                    result_index_list.append(int(j))
                    log_info += f"tag={rec.text} -> {j};"
        else:
            if i not in result_index_list:
                result_index_list.append(i)
                log_info += f"para={i};"

    # Pad short lists with the first valid index, matching legacy behaviour.
    if len(result_index_list) < top_k:
        pad = result_index_list[0] if result_index_list else 0
        while len(result_index_list) < top_k:
            result_index_list.append(pad)

    topk_indices = result_index_list[:top_k]
    result = [materials[i].text.strip() for i in topk_indices]
    LOG.debug("vector_search picks=%s log=%s", topk_indices[::-1], log_info)
    return result, topk_indices[::-1]


def find_material_by_index(
    index_list: List[int], character: str, subject: str
) -> List[str]:
    materials = load_materials(character, subject)
    out: List[str] = []
    for i in index_list:
        if 0 <= i < len(materials):
            out.append(materials[i].text.strip())
        else:
            out.append("")
    return out


def reorganize_index(
    base_list: List[int], append_list: List[int], max_length: int
) -> List[int]:
    """LRU-style merge: append items, evicting duplicates from the front."""
    for member in append_list:
        if member in base_list:
            base_list.remove(member)
        base_list.append(member)
    return base_list[-max_length:]


def process_embedding(
    content: str,
    top_k: int,
    character: str,
    client_buffer: List[int],
    max_length: int,
    client_information: str = "",
) -> Tuple[str, List[int]]:
    """Compose the setting + knowledge knowledge string for a chat turn."""
    search_result, server_embedding_index_list = vector_search(
        question=content,
        character=character,
        subject="setting",
        top_k=top_k,
        instruct="给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息",
    )
    embedding_index = reorganize_index(
        base_list=list(client_buffer or []),
        append_list=server_embedding_index_list,
        max_length=max_length,
    )
    server_embedding_list = find_material_by_index(
        index_list=embedding_index, character=character, subject="setting"
    )
    knowledge, _knowledge_idx = vector_search(
        question=content,
        character="_shared",
        subject="knowledge",
        top_k=3,
        instruct="给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息",
    )
    full_knowledge = (
        f"这些是你知道的事实：{server_embedding_list}\n"
        f"这些是你了解的知识：{knowledge}\n"
        f"{client_information}\n"
    )
    return full_knowledge, embedding_index


def check_emotion(emotion: str, character: str) -> str:
    """Snap ``【xxx】`` to the closest expression in the character's library."""
    emotion_text = emotion.replace("【", "").replace("】", "")
    materials = load_materials(character, "expression")
    texts = [m.text for m in materials]
    if emotion_text in texts:
        LOG.info("emotion %s validated as-is", emotion_text)
        return emotion
    loaded = _load_index_and_materials(character, "expression")
    if loaded is None or not texts:
        return emotion
    idx, _ = loaded
    search = np.asarray(
        _get_model().encode(
            [get_detailed_instruct("找到与给出的表情表达情感最相近的表情", emotion_text)]
        ),
        dtype="float32",
    )
    _accuracy, matches = idx.search(search, 1)
    pos = int(matches[0][0])
    if pos < 0 or pos >= len(materials):
        return emotion
    snapped = materials[pos].text.strip()
    LOG.info("emotion %s -> %s", emotion_text, snapped)
    return f"【{snapped}】"


if __name__ == "__main__":  # pragma: no cover -- manual smoke test
    logging.basicConfig(level=logging.INFO)
    print("Knowledge:" + generate_vector("tendou_arisu", "knowledge"))
