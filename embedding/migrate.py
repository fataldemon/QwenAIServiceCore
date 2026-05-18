"""One-shot migration from the legacy pickle schema to ``materials.jsonl``.

The legacy layout per subject was::

    embedding/<character>/<subject>/vector/
        materials.pkl       # list[str]   -- paragraphs + tags, FAISS-aligned
        tags_map.pkl        # dict[str, list[int]]  -- tag -> paragraph ids
        paragraphs.pkl      # list[str]   -- paragraphs only (no tags)
        index.faiss         # FAISS index
        srch_embeddings.npy # (optional) raw embeddings

This module rewrites ``materials.jsonl`` from the three pkl files, preserving
FAISS row ordering so the existing ``index.faiss`` keeps working. Original
pkl files are renamed to ``*.pkl.bak`` -- never deleted -- so a rollback is
always possible.

Migration is idempotent: if ``materials.jsonl`` already exists for a subject,
that subject is skipped.

This module performs no embedding computations; it only reshuffles metadata.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Tuple

from .data_store import (
    MaterialRecord,
    VALID_SUBJECTS,
    EMBEDDING_ROOT,
    materials_path,
    vector_dir,
    list_characters,
    save_materials,
)

LOG = logging.getLogger(__name__)


def _legacy_paths(character: str, subject: str) -> Tuple[str, str, str]:
    vdir = vector_dir(character, subject)
    return (
        os.path.join(vdir, "materials.pkl"),
        os.path.join(vdir, "tags_map.pkl"),
        os.path.join(vdir, "paragraphs.pkl"),
    )


def _read_pickle(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # pragma: no cover -- corrupt pkls are user-fixable
        LOG.warning("Failed reading %s: %r", path, e)
        return None


def _rename_to_bak(path: str) -> None:
    if not os.path.exists(path):
        return
    bak = path + ".bak"
    # Don't overwrite an existing backup -- pick a fresh suffix instead.
    if os.path.exists(bak):
        i = 1
        while os.path.exists(f"{bak}.{i}"):
            i += 1
        bak = f"{bak}.{i}"
    try:
        os.replace(path, bak)
        LOG.info("Backed up legacy pickle: %s -> %s", path, bak)
    except OSError as e:  # pragma: no cover
        LOG.warning("Could not rename %s to backup: %r", path, e)


def migrate_subject(character: str, subject: str) -> bool:
    """Migrate one subject. Returns True if a migration actually happened."""
    target = materials_path(character, subject)
    if os.path.exists(target):
        return False
    materials_pkl, tagsmap_pkl, paragraphs_pkl = _legacy_paths(character, subject)
    if not (os.path.exists(materials_pkl) and os.path.exists(tagsmap_pkl)):
        # Nothing to migrate -- subject is either empty or freshly created.
        return False

    materials: List[str] = _read_pickle(materials_pkl) or []
    tags_map: Dict[str, List[int]] = _read_pickle(tagsmap_pkl) or {}
    paragraphs: List[str] = _read_pickle(paragraphs_pkl) or []

    # In the legacy layout the FAISS index is built from ``materials``, which
    # is ``paragraphs + tag_names`` in that exact order. So:
    #   row 0..len(paragraphs)-1   are paragraph rows
    #   row len(paragraphs)..      are tag rows
    n_para = len(paragraphs)
    # Sanity: ``materials`` length should be n_para + len(tags_map)
    expected_tag_rows = len(materials) - n_para
    if expected_tag_rows < 0:
        LOG.warning(
            "Inconsistent legacy data for %s/%s: materials=%d, paragraphs=%d",
            character,
            subject,
            len(materials),
            n_para,
        )

    # Build a quick reverse lookup from tag-row index to tag string by walking
    # ``materials[n_para:]``. We do not rely on ``tags_map`` ordering for the
    # row indices.
    tag_rows_by_idx: Dict[int, str] = {}
    for offset in range(max(0, expected_tag_rows)):
        idx = n_para + offset
        if idx < len(materials):
            tag_rows_by_idx[idx] = str(materials[idx])

    # For each paragraph, find its tags by reverse lookup in ``tags_map``.
    para_to_tags: Dict[int, List[str]] = {}
    for tag, ids in tags_map.items():
        for pid in ids or []:
            if 0 <= pid < n_para:
                para_to_tags.setdefault(pid, []).append(str(tag))

    records: List[MaterialRecord] = []
    for pid in range(n_para):
        records.append(
            MaterialRecord(
                id=pid,
                kind="para",
                type="text",
                text=str(paragraphs[pid]),
                tags=list(para_to_tags.get(pid, [])),
            )
        )
    for tag_idx, tag_text in tag_rows_by_idx.items():
        records.append(
            MaterialRecord(
                id=tag_idx,
                kind="tag",
                type="text",
                text=tag_text,
                target_ids=list(tags_map.get(tag_text, []) or []),
            )
        )

    save_materials(character, subject, records)

    # Back up the legacy files (kept on disk, not in git -- see .gitignore).
    for p in (materials_pkl, tagsmap_pkl, paragraphs_pkl):
        _rename_to_bak(p)
    LOG.info(
        "Migrated %s/%s: %d paragraphs, %d tag rows",
        character,
        subject,
        n_para,
        len(tag_rows_by_idx),
    )
    return True


def migrate_all() -> Dict[str, List[str]]:
    """Migrate every (character, subject) pair under ``embedding/``.

    Returns a dict keyed by character, with the list of subjects that were
    migrated. Existing JSONL files are left untouched.
    """
    if not os.path.isdir(EMBEDDING_ROOT):
        return {}
    summary: Dict[str, List[str]] = {}
    for character in list_characters():
        migrated: List[str] = []
        for subject in VALID_SUBJECTS:
            if migrate_subject(character, subject):
                migrated.append(subject)
        if migrated:
            summary[character] = migrated
    return summary


if __name__ == "__main__":  # pragma: no cover -- manual usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = migrate_all()
    if not result:
        print("Nothing to migrate.")
    else:
        for c, subs in result.items():
            print(f"{c}: migrated subjects -> {', '.join(subs)}")
