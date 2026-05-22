"""Core module: contradiction detection pipeline components.

Exposes:
    - NLIScorer: local NLI model for fast sentence-pair scoring
    - BaseLLMJudge: abstract base for LLM judge providers (OpenAI / Claude)
    - create_llm_judge: factory that returns the configured provider instance
    - Router: confidence-based routing orchestrating NLI + LLM escalation
    - Aggregator: converts contradictions into a scored report
"""

__all__ = [
    "NLIScorer",
    "BaseLLMJudge",
    "create_llm_judge",
    "Router",
    "Aggregator",
    "nli_to_contradiction",
]

from backend.core.aggregator import Aggregator
from backend.core.llm_judge import BaseLLMJudge, create_llm_judge
from backend.core.nli_scorer import NLIScorer
from backend.core.router import Router, nli_to_contradiction

