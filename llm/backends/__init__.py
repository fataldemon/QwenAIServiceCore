"""LLM backend implementations.

A *backend* is the thin layer that turns the service's internal chat
representation into upstream HTTP/RPC calls and yields back text (optionally
streamed). The legacy in-process vLLM ``AsyncLLMEngine`` is **not** used on
this branch -- all inference goes through an OpenAI-compatible HTTP endpoint
(``vllm serve``, DashScope, DeepSeek, ...). The legacy code in
``llm/local_llm_manage.py`` is preserved as dead code so it can be revived if
we ever want to merge an in-process mode back in.
"""

from .base import LLMBackend, GenerationResult, StreamChunk  # noqa: F401
from .openai_compatible import OpenAICompatibleBackend  # noqa: F401
from .registry import get_backend  # noqa: F401
