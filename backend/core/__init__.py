"""Core module: contradiction detection pipeline components.

Exposes:
    - NLIScorer: local NLI model for fast sentence-pair scoring
    - LLMJudge: GPT-4o with structured outputs for reasoning about multi-hop contradictions
    - Router: confidence-based routing orchestrating NLI + LLM escalation
    - Aggregator: converts contradictions into a scored report
"""

__all__ = [
    "NLIScorer",
    "LLMJudge",
    "Router",
    "Aggregator",
    "nli_to_contradiction",
]

from backend.core.aggregator import Aggregator
from backend.core.llm_judge import LLMJudge
from backend.core.nli_scorer import NLIScorer
from backend.core.router import Router, nli_to_contradiction
