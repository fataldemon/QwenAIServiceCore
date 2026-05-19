"""Skill discovery and lookup.

A *skill* is a folder under ``skills/<skill_name>/`` containing a ``SKILL.md``
file. The file starts with a YAML front-matter block (``---``-delimited)
describing the skill's metadata; everything after the front matter is the
markdown body that gets returned to the LLM on demand.

This module:

1. Scans ``skills/`` at startup, parses every ``SKILL.md``, and keeps the
   metadata in memory. The body is **not** cached -- it is read on demand by
   :meth:`SkillManager.read_skill` so that authors can edit a skill and have
   the next call pick it up without a service restart.
2. Exposes two virtual tools that any model can call (regardless of whether
   MCP is configured): ``list_skills`` and ``read_skill``. These appear in
   :meth:`SkillManager.virtual_tools`.
3. Implements optional :func:`auto_inject`: if a skill's front-matter sets
   ``auto_inject: true`` and any of its ``triggers.keywords`` /
   ``triggers.regex`` matches the latest user message, the skill body is
   merged into the system prompt automatically, saving one tool-call round
   trip.

The YAML front-matter is parsed with PyYAML; the parser tolerates skills
without front matter (such files are simply skipped).
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SKILLS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills"
)

LOG = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str = ""
    version: str = ""
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    auto_inject: bool = False
    files: List[str] = field(default_factory=list)
    skill_dir: str = ""

    def matches(self, text: str) -> bool:
        if not text:
            return False
        haystack = text.lower()
        for kw in self.keywords:
            if kw and kw.lower() in haystack:
                return True
        for pat in self.regex_patterns:
            try:
                if re.search(pat, text, re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False


def _split_frontmatter(text: str) -> tuple:
    """Return ``(front_matter_yaml_text, body)``.

    Front matter must be ``---``-delimited and appear at the top of the file.
    Returns ``("", text)`` if no front matter is found.
    """
    if not text.startswith("---"):
        return "", text
    # Find the second '---' on its own line.
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return "", text
    return m.group(1), m.group(2)


class SkillManager:
    """Singleton-style skill manager."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self.reload()

    # ----- discovery --------------------------------------------------------

    def reload(self) -> None:
        self._skills.clear()
        if not os.path.isdir(SKILLS_ROOT):
            return
        try:
            import yaml  # type: ignore
        except Exception as e:
            LOG.warning("pyyaml not available; skill front-matter will be ignored: %r", e)
            yaml = None  # type: ignore

        for entry in sorted(os.listdir(SKILLS_ROOT)):
            if entry.startswith("."):
                continue
            sdir = os.path.join(SKILLS_ROOT, entry)
            md = os.path.join(sdir, "SKILL.md")
            if not os.path.isfile(md):
                continue
            try:
                with open(md, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError as e:
                LOG.warning("Cannot read %s: %r", md, e)
                continue
            fm_text, _body = _split_frontmatter(raw)
            meta: Dict[str, Any] = {}
            if fm_text and yaml is not None:
                try:
                    meta = yaml.safe_load(fm_text) or {}
                except Exception as e:
                    LOG.warning("Bad front matter in %s: %r", md, e)
                    meta = {}
            skill = Skill(
                name=str(meta.get("name") or entry),
                description=str(meta.get("description", "")),
                version=str(meta.get("version", "")),
                keywords=list(((meta.get("triggers") or {}).get("keywords")) or []),
                regex_patterns=list(((meta.get("triggers") or {}).get("regex")) or []),
                auto_inject=bool(meta.get("auto_inject", False)),
                files=list(meta.get("files") or []),
                skill_dir=sdir,
            )
            self._skills[skill.name] = skill
        LOG.info("Loaded %d skill(s)", len(self._skills))

    # ----- public API -------------------------------------------------------

    def list_skills(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sk in self._skills.values():
            out.append(
                {
                    "name": sk.name,
                    "description": sk.description,
                    "version": sk.version,
                    "auto_inject": sk.auto_inject,
                }
            )
        return out

    def read_skill(self, name: str) -> Optional[str]:
        sk = self._skills.get(name)
        if sk is None:
            return None
        path = os.path.join(sk.skill_dir, "SKILL.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return None
        _fm, body = _split_frontmatter(raw)
        return body

    def read_skill_raw(self, name: str) -> Optional[str]:
        """Read the full SKILL.md content including front matter."""
        sk = self._skills.get(name)
        if sk is None:
            return None
        path = os.path.join(sk.skill_dir, "SKILL.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def write_skill(self, name: str, body: str) -> bool:
        """Write (create or update) a SKILL.md file. Returns True on success."""
        try:
            sdir = os.path.join(SKILLS_ROOT, name)
            os.makedirs(sdir, exist_ok=True)
            path = os.path.join(sdir, "SKILL.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            self.reload()
            return True
        except OSError as e:
            LOG.warning("write_skill(%s): %r", name, e)
            return False

    def delete_skill(self, name: str) -> bool:
        """Delete a skill directory and its SKILL.md. Returns True on success."""
        sdir = os.path.join(SKILLS_ROOT, name)
        if not os.path.isdir(sdir):
            return False
        try:
            import shutil
            shutil.rmtree(sdir)
            self.reload()
            return True
        except OSError as e:
            LOG.warning("delete_skill(%s): %r", name, e)
            return False

    def auto_inject_for(self, user_text: str) -> List[str]:
        """Return bodies of every skill that auto-injects on ``user_text``."""
        out: List[str] = []
        for sk in self._skills.values():
            if not sk.auto_inject:
                continue
            if sk.matches(user_text):
                body = self.read_skill(sk.name)
                if body:
                    out.append(body)
        return out

    # ----- virtual tools (always exposed to the LLM) ------------------------

    def virtual_tools(self) -> List[Dict[str, Any]]:
        """OpenAI ``tools=[...]`` entries that any backend can advertise."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": "List available skill modules (names + descriptions).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_skill",
                    "description": "Read the body of a skill module by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            },
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)


_singleton: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    global _singleton
    if _singleton is None:
        _singleton = SkillManager()
    return _singleton
