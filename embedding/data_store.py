"""Knowledge-base data layer (new ``jsonl + binaries`` schema).

The legacy embedding store kept three pickles (``materials.pkl``,
``tags_map.pkl``, ``paragraphs.pkl``) alongside a FAISS index. Each entry was
just a bare string, with tags piggy-backing on the same flat list. That works
for text but does not extend gracefully to multimodal records.

This module replaces that layout with an explicit record schema:

``embedding/<character>/<subject>/``
    * ``materials.jsonl``   -- one JSON record per line, ordered to match the
      FAISS row index, including tag pseudo-rows.
    * ``binaries/``         -- optional sidecar files for non-text records.
    * ``vector/index.faiss``-- the FAISS index (unchanged binary format).

Record schema (every line of ``materials.jsonl``)::

    {
      "id":         <int>,     # stable row index in the FAISS matrix
      "kind":       "para"|"tag",
      "type":       "text"|"image"|"audio"|"video",
      "text":       <str>,     # main searchable text; for "tag" kind: the
                               # tag name itself; for media: a textual
                               # description that drives the vector
      "media_ref":  null | { "source": ..., ... }, # see ContentPart.ref
      "tags":       [<str>, ...],   # only meaningful for "para" rows
      "target_ids": [<int>, ...],   # only for "tag" rows -- paragraph row
                                    # indices the tag points to
    }

The schema is forward-compatible: adding new optional fields will not break
older readers as long as readers ignore unknown keys.

The store is process-local and intentionally not async-safe across processes.
We assume the FastAPI app is single-process (the typical deployment), and use
an in-process :class:`asyncio.Lock` per ``(character, subject)`` pair to
serialize writes. Reads are lock-free copies.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

EMBEDDING_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embedding"
)

VALID_SUBJECTS = ("setting", "expression", "behaviour", "memory", "knowledge")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MaterialRecord:
    """A single row in the FAISS-aligned materials list."""

    id: int
    kind: str = "para"  # "para" or "tag"
    type: str = "text"  # "text" / "image" / "audio" / "video"
    text: str = ""
    media_ref: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialRecord":
        return cls(
            id=int(data.get("id", 0)),
            kind=str(data.get("kind", "para")),
            type=str(data.get("type", "text")),
            text=str(data.get("text", "")),
            media_ref=data.get("media_ref") or None,
            tags=list(data.get("tags", []) or []),
            target_ids=list(data.get("target_ids", []) or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "type": self.type,
            "text": self.text,
            "media_ref": self.media_ref,
            "tags": list(self.tags),
            "target_ids": list(self.target_ids),
        }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def subject_dir(character: str, subject: str) -> str:
    return os.path.join(EMBEDDING_ROOT, character, subject)


def vector_dir(character: str, subject: str) -> str:
    return os.path.join(subject_dir(character, subject), "vector")


def materials_path(character: str, subject: str) -> str:
    return os.path.join(vector_dir(character, subject), "materials.jsonl")


def index_path(character: str, subject: str) -> str:
    return os.path.join(vector_dir(character, subject), "index.faiss")


def binaries_dir(character: str, subject: str) -> str:
    return os.path.join(subject_dir(character, subject), "binaries")


def identity_path(character: str) -> str:
    return os.path.join(EMBEDDING_ROOT, character, "identity.mem")


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------


_subject_locks: Dict[Tuple[str, str], asyncio.Lock] = {}


def _get_lock(character: str, subject: str) -> asyncio.Lock:
    key = (character, subject)
    lock = _subject_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _subject_locks[key] = lock
    return lock


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def list_characters() -> List[str]:
    if not os.path.isdir(EMBEDDING_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(EMBEDDING_ROOT)):
        path = os.path.join(EMBEDDING_ROOT, name)
        if os.path.isdir(path):
            out.append(name)
    return out


def list_subjects(character: str) -> List[str]:
    base = os.path.join(EMBEDDING_ROOT, character)
    if not os.path.isdir(base):
        return []
    return [s for s in VALID_SUBJECTS if os.path.isdir(os.path.join(base, s))]


def load_materials(character: str, subject: str) -> List[MaterialRecord]:
    path = materials_path(character, subject)
    if not os.path.exists(path):
        return []
    out: List[MaterialRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = MaterialRecord.from_dict(json.loads(line))
            except Exception:
                continue
            out.append(rec)
    return out


def load_paragraphs_only(character: str, subject: str) -> List[MaterialRecord]:
    """Convenience: paragraph rows only (no tag rows)."""
    return [r for r in load_materials(character, subject) if r.kind == "para"]


def load_identity(character: str) -> List[str]:
    path = identity_path(character)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: str, lines: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
    os.replace(tmp, path)


def save_materials(
    character: str, subject: str, records: List[MaterialRecord]
) -> None:
    """Overwrite ``materials.jsonl`` with ``records`` (one JSON per line)."""
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    _atomic_write(materials_path(character, subject), lines)


def save_identity(character: str, identities: List[str]) -> None:
    path = identity_path(character)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in identities:
            line = line.strip()
            if not line:
                continue
            f.write(line + "\n")
    os.replace(tmp, path)


def ensure_subject_dirs(character: str, subject: str) -> None:
    os.makedirs(vector_dir(character, subject), exist_ok=True)
    os.makedirs(binaries_dir(character, subject), exist_ok=True)


def delete_character(character: str) -> None:
    base = os.path.join(EMBEDDING_ROOT, character)
    if os.path.isdir(base):
        shutil.rmtree(base)


def create_character(character: str) -> None:
    base = os.path.join(EMBEDDING_ROOT, character)
    os.makedirs(base, exist_ok=True)
    for sub in VALID_SUBJECTS:
        ensure_subject_dirs(character, sub)
    if not os.path.exists(identity_path(character)):
        save_identity(character, [])


# ---------------------------------------------------------------------------
# Convenience builders for CRUD
# ---------------------------------------------------------------------------


def build_paragraph_records(
    paragraphs: List[Dict[str, Any]],
) -> List[MaterialRecord]:
    """Turn a list of ``{text, tags?, type?, media_ref?}`` items into rows.

    Tag rows (``kind="tag"``) are derived automatically from the union of
    every paragraph's ``tags``. Their ``target_ids`` point to all paragraphs
    that mention them. IDs are assigned sequentially starting at 0, with
    paragraph rows first and tag rows appended after.
    """
    para_records: List[MaterialRecord] = []
    tag_map: Dict[str, List[int]] = {}
    for i, p in enumerate(paragraphs):
        text = str(p.get("text", "")).strip()
        ptype = str(p.get("type", "text"))
        media_ref = p.get("media_ref")
        tags = [t for t in (p.get("tags") or []) if str(t).strip()]
        rec = MaterialRecord(
            id=i,
            kind="para",
            type=ptype,
            text=text,
            media_ref=media_ref,
            tags=list(tags),
        )
        para_records.append(rec)
        for tag in tags:
            tag_map.setdefault(tag, []).append(i)

    tag_records: List[MaterialRecord] = []
    next_id = len(para_records)
    for tag, target_ids in tag_map.items():
        tag_records.append(
            MaterialRecord(
                id=next_id,
                kind="tag",
                type="text",
                text=tag,
                target_ids=list(target_ids),
            )
        )
        next_id += 1
    return para_records + tag_records


__all__ = [
    "EMBEDDING_ROOT",
    "VALID_SUBJECTS",
    "MaterialRecord",
    "subject_dir",
    "vector_dir",
    "materials_path",
    "index_path",
    "binaries_dir",
    "identity_path",
    "list_characters",
    "list_subjects",
    "load_materials",
    "load_paragraphs_only",
    "load_identity",
    "save_materials",
    "save_identity",
    "ensure_subject_dirs",
    "delete_character",
    "create_character",
    "build_paragraph_records",
]
