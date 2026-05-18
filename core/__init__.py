"""Core infrastructure modules.

This package gathers the cross-cutting concerns of the service:

* :mod:`core.config_manager` -- load/save provider & MCP configuration.
* :mod:`core.content_normalizer` -- unify legacy placeholder strings and
  OpenAI-style multimodal content arrays into an ordered representation.
* :mod:`core.mcp_manager` -- maintain a pool of MCP client connections.
* :mod:`core.skill_manager` -- discover skills under ``skills/`` and offer
  ``list_skills`` / ``read_skill`` virtual tools.
"""
