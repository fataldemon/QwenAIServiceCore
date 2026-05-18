"""Per-character persona management.

Personas live alongside the embedding data they belong to::

    embedding/<character>/persona.json

A persona is a small JSON document with these fields (all optional except
``display_name``)::

    {
      "display_name":     "天童爱丽丝",
      "setting":          "system prompt body, may contain {embeddings}",
      "reply_instruction": "appended after `setting`",
      "image_setting":    "optional figure / illustration framing",
      "max_chat_len":      15000,
      "max_analysis_len":  6000,
      "max_quick_reply":   600,
      "default_temperature": 0.7,
      "default_top_p":     0.9
    }

When ``chat_on_setting`` builds the system prompt for a character, it first
asks the :class:`PersonaManager` for that character's persona. If no
``persona.json`` is present, the request is treated as a generic completion
without any character framing -- the legacy hard-coded "Alice" prompt that
used to live in :mod:`template` is gone, so every persona is now data.

The legacy strings (``SETTING`` / ``REPLY_INSTRUCTION`` / ``IMAGE_SETTING``
from :mod:`template`) are still imported once at boot to seed
``embedding/tendou_arisu/persona.json`` if that file does not yet exist.
That keeps existing deployments working without manual migration.

This module is process-wide singleton, file-backed, and safe to call from
both sync (Gradio callbacks) and async (admin REST) contexts. The async
mutators take an ``asyncio.Lock``; the read paths are lock-free and rely on
copy-on-write of the in-memory dict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

PERSONA_FILENAME = "persona.json"
DEFAULT_EMBEDDING_ROOT = "embedding"


@dataclass
class Persona:
    """One character's persona configuration.

    Mirrors the fields described in this module's docstring. Numeric defaults
    of ``None`` mean "fall back to the application-level default" (see
    :mod:`template`).
    """

    character: str  # folder name under embedding/, also the primary key
    display_name: str = ""
    setting: str = ""
    reply_instruction: str = ""
    image_setting: str = ""
    max_chat_len: Optional[int] = None
    max_analysis_len: Optional[int] = None
    max_quick_reply: Optional[int] = None
    default_temperature: Optional[float] = None
    default_top_p: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, character: str, data: Dict[str, Any]) -> "Persona":
        known = {
            "display_name",
            "setting",
            "reply_instruction",
            "image_setting",
            "max_chat_len",
            "max_analysis_len",
            "max_quick_reply",
            "default_temperature",
            "default_top_p",
        }
        kwargs: Dict[str, Any] = {"character": character}
        for k in known:
            if k in data and data[k] is not None:
                kwargs[k] = data[k]
        # Anything we don't know is parked in ``extra`` so user-supplied
        # fields round-trip through GET/PUT unchanged.
        extra = {k: v for k, v in data.items() if k not in known}
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # ``character`` is the URL key, not a body field.
        out.pop("character", None)
        extra = out.pop("extra", {}) or {}
        out.update(extra)
        return out


class PersonaManager:
    """File-backed registry of :class:`Persona` objects.

    The on-disk layout is one ``persona.json`` per character::

        embedding/
          tendou_arisu/
            persona.json
            identity.mem
            setting/
            knowledge/
            expression/
          some_other_character/
            persona.json
            ...

    Characters without a ``persona.json`` are simply not returned from
    :meth:`get_persona` -- callers should treat that as "no persona, run a
    generic completion".
    """

    def __init__(self, root: str = DEFAULT_EMBEDDING_ROOT) -> None:
        self._root = root
        self._lock = asyncio.Lock()
        # ``_thread_lock`` covers the in-memory cache mutation path used by
        # both sync (UI) and async (REST) callers.
        self._thread_lock = threading.Lock()
        self._cache: Dict[str, Persona] = {}
        self._reload_from_disk()

    # ---- internal -----------------------------------------------------

    def _character_dir(self, character: str) -> str:
        return os.path.join(self._root, character)

    def _persona_path(self, character: str) -> str:
        return os.path.join(self._character_dir(character), PERSONA_FILENAME)

    def _reload_from_disk(self) -> None:
        cache: Dict[str, Persona] = {}
        if not os.path.isdir(self._root):
            with self._thread_lock:
                self._cache = cache
            return
        for entry in sorted(os.listdir(self._root)):
            char_dir = os.path.join(self._root, entry)
            if not os.path.isdir(char_dir):
                continue
            pp = os.path.join(char_dir, PERSONA_FILENAME)
            if not os.path.isfile(pp):
                continue
            try:
                with open(pp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("persona.json must be a JSON object")
                cache[entry] = Persona.from_dict(entry, data)
            except Exception as e:
                LOG.warning("Failed to load persona for %s: %r", entry, e)
        with self._thread_lock:
            self._cache = cache

    @staticmethod
    def _atomic_write(path: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ---- public reads -------------------------------------------------

    def get_persona(self, character: str) -> Optional[Persona]:
        if not character:
            return None
        with self._thread_lock:
            p = self._cache.get(character)
            return deepcopy(p) if p is not None else None

    def list_personas(self) -> List[Persona]:
        with self._thread_lock:
            return [deepcopy(p) for p in self._cache.values()]

    def reload(self) -> None:
        self._reload_from_disk()

    # ---- public writes ------------------------------------------------

    async def upsert_persona(
        self, character: str, data: Dict[str, Any]
    ) -> Persona:
        if not character or "/" in character or "\\" in character:
            raise ValueError("invalid character name")
        # Reject obvious template-string errors early.
        setting = data.get("setting") or ""
        if setting and "{embeddings}" not in setting:
            LOG.warning(
                "persona %s has no {embeddings} placeholder; knowledge-base "
                "retrieval will not be injected.",
                character,
            )
        async with self._lock:
            persona = Persona.from_dict(character, data)
            # Ensure the character directory and its standard subfolders
            # exist so the rest of the pipeline (embedding, .mem files) can
            # find a place to write things.
            for sub in ("setting", "knowledge", "expression"):
                os.makedirs(
                    os.path.join(self._character_dir(character), sub),
                    exist_ok=True,
                )
            self._atomic_write(self._persona_path(character), persona.to_dict())
            with self._thread_lock:
                self._cache[character] = persona
            return deepcopy(persona)

    async def delete_persona(self, character: str) -> bool:
        async with self._lock:
            path = self._persona_path(character)
            removed = False
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed = True
                except OSError as e:
                    LOG.warning("delete_persona: %r", e)
                    return False
            with self._thread_lock:
                if character in self._cache:
                    del self._cache[character]
                    removed = True
            return removed

    # ---- bootstrap migration -----------------------------------------

    def seed_legacy_alice_if_missing(
        self,
        *,
        character: str,
        display_name: str,
        setting: str,
        reply_instruction: str,
        image_setting: str = "",
    ) -> None:
        """One-shot migration: write a persona file for a legacy character.

        Idempotent: skipped if the persona.json already exists. Intended to
        be called once from FastAPI's lifespan handler so existing
        deployments keep working without manual file authoring.
        """
        path = self._persona_path(character)
        if os.path.isfile(path):
            return
        if not os.path.isdir(self._character_dir(character)):
            # Don't seed a persona for a character whose embedding folder
            # does not exist -- that would create empty folders for a
            # character the user never actually had.
            return
        data = {
            "display_name": display_name,
            "setting": setting,
            "reply_instruction": reply_instruction,
            "image_setting": image_setting,
        }
        try:
            self._atomic_write(path, data)
            with self._thread_lock:
                self._cache[character] = Persona.from_dict(character, data)
            LOG.info("Seeded legacy persona for %s at %s", character, path)
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to seed persona for %s: %r", character, e)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_INSTANCE: Optional[PersonaManager] = None
_INSTANCE_LOCK = threading.Lock()


def get_persona_manager() -> PersonaManager:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = PersonaManager()
    return _INSTANCE
