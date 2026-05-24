"""Integrations with third-party agent frameworks.

Currently exposes:
    FaithfulnessGuard       - LangChain callback that checks agent responses
                              against tool outputs for faithfulness violations
    FaithfulnessViolationError - raised when raise_on_violation=True
"""

from backend.integrations.langchain_guard import FaithfulnessGuard, FaithfulnessViolationError

__all__ = ["FaithfulnessGuard", "FaithfulnessViolationError"]
