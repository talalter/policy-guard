"""FastAPI dependency providers - inject shared objects from app.state.

All Depends() callables live here so route modules stay focused on
HTTP concerns and the injection wiring is visible in one place.
"""

from fastapi import Request

from backend.core import Aggregator, NLIScorer, Router
from backend.core.llm_judge import BaseLLMJudge


def get_router(request: Request) -> Router:
    """Inject the Router instance from app state."""
    return request.app.state.router


def get_nli_scorer(request: Request) -> NLIScorer:
    """Inject the NLIScorer instance from app state."""
    return request.app.state.nli_scorer


def get_llm_judge(request: Request) -> BaseLLMJudge:
    """Inject the LLM judge instance from app state."""
    return request.app.state.llm_judge


def get_aggregator(request: Request) -> Aggregator:
    """Inject the Aggregator instance from app state."""
    return request.app.state.aggregator


def get_db(request: Request):
    """Inject the MongoDB database from app state (may be None)."""
    return getattr(request.app.state, "db", None)
