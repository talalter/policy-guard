"""LLM judge package - provider-agnostic interface for contradiction detection.

Provider is selected at startup via settings.llm_provider ("openai" or "claude").
Both implementations share the same tool definitions and agentic loop logic;
only the API client and message formatting differ.

Public API:
    BaseLLMJudge    - abstract base for type annotations
    create_llm_judge - factory that returns the configured provider instance
"""

from backend.config import settings
from backend.core.llm_judge.base import BaseLLMJudge
from backend.core.llm_judge.claude_judge import ClaudeJudge
from backend.core.llm_judge.openai_judge import OpenAIJudge

__all__ = ["BaseLLMJudge", "create_llm_judge"]


def create_llm_judge() -> BaseLLMJudge:
    """Instantiate the LLM judge for the configured provider.

    Reads settings.llm_provider to select between OpenAI and Claude.
    Raises ValueError for unknown provider values.
    """
    if settings.llm_provider == "openai":
        return OpenAIJudge()
    if settings.llm_provider == "claude":
        return ClaudeJudge()
    raise ValueError(
        f"Unknown llm_provider: {settings.llm_provider!r}. Valid values: 'openai', 'claude'."
    )
