---
name: example_skill
description: |
  A demo skill that shows how to declare metadata and content for a skill.
  Replace this file when authoring real skills.
version: 0.1.0
triggers:
  keywords:
    - "示例技能"
    - "example skill"
  regex: []
auto_inject: false
files: []
---

# Example Skill

This is the body of the skill. When the LLM calls `read_skill("example_skill")`
through the built-in virtual tools, the entire markdown body (everything below
the front-matter) will be returned as the skill content.

## How the agent should use this skill

1. Read the metadata via `list_skills` to discover available skills.
2. Call `read_skill(name)` to load the body when needed.
3. Apply the instructions described here when answering the user.

> Note: If `auto_inject` is `true` AND any of `triggers.keywords` (case
> insensitive substring match) or `triggers.regex` matches the latest user
> message, the body of this skill will be prepended into the system prompt
> automatically, saving a tool-call round trip.
