"""Detection endpoints: POST /check, /check/nli-only, /check/llm-only."""

import logging
import time
from datetime import datetime, timezone

try:
    import openai  # type: ignore
    _RateLimitError: type = openai.RateLimitError
    _APIError: type = openai.APIError
except ImportError:
    _RateLimitError = type(None)
    _APIError = type(None)

from fastapi import APIRouter, Depends, Header, HTTPException

from backend.api.deps import get_aggregator, get_db, get_llm_judge, get_nli_scorer, get_router
from backend.config import settings
from backend.core import Aggregator, NLIScorer, Router, nli_to_contradiction
from backend.core.llm_judge import BaseLLMJudge
from backend.models import CheckRequest, Contradiction, ContradictionReport

logger = logging.getLogger(__name__)
router = APIRouter()


async def _persist_run(db, body: CheckRequest, report: ContradictionReport, session_id: str | None) -> str:
    """Insert a detection run into MongoDB and return its string _id."""
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "session_id": session_id,
        "context": body.context,
        "response": body.response,
        "provider": settings.llm_provider,
        "contradictions": [c.model_dump(mode="json") for c in report.contradictions],
        "faithfulness_score": report.faithfulness_score,
        "method_used": report.method_used.value,
        "processing_time_ms": report.processing_time_ms,
    }
    result = await db.detection_runs.insert_one(doc)
    return str(result.inserted_id)


@router.post("/check", response_model=ContradictionReport)
async def check(
    body: CheckRequest,
    pipeline: Router = Depends(get_router),
    aggregator: Aggregator = Depends(get_aggregator),
    db=Depends(get_db),
    x_session_id: str | None = Header(default=None),
) -> ContradictionReport:
    """Run the full ensemble pipeline (NLI + LLM confidence-based routing).

    Confident NLI pairs are resolved locally for free.  Uncertain pairs are
    escalated to the LLM judge, which catches multi-hop contradictions NLI
    misses.  The run is persisted to MongoDB when available.
    """
    t0 = time.perf_counter()
    try:
        contradictions, metadata = pipeline.route(body.context, body.response)
    except _RateLimitError as exc: # type: ignore
        raise HTTPException(status_code=429, detail="LLM rate limit — retry after a moment") from exc
    except _APIError as exc: # type: ignore
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unhandled error in POST /check")
        raise HTTPException(status_code=500, detail="Pipeline error") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    report = aggregator.aggregate(contradictions, metadata, elapsed_ms)
    if db is not None:
        report.run_id = await _persist_run(db, body, report, x_session_id)
    return report


@router.post("/check/nli-only", response_model=ContradictionReport)
async def check_nli_only(
    body: CheckRequest,
    nli_scorer: NLIScorer = Depends(get_nli_scorer),
    aggregator: Aggregator = Depends(get_aggregator),
) -> ContradictionReport:
    """Run NLI-only detection — no LLM escalation.

    All sentence pairs are scored by the cross-encoder.  Every pair labelled
    'contradiction' with confidence ≥ NLI_CONFIDENCE_THRESHOLD is returned
    as a finding.  Used by the benchmark tab to isolate NLI performance.
    """
    logger.debug(
        "POST /check/nli-only  context=%d chars  response=%d chars",
        len(body.context), len(body.response),
    )
    t0 = time.perf_counter()
    try:
        contradictions: list[Contradiction] = []
        total_pairs = 0
        for result in nli_scorer.score(body.context, body.response):
            total_pairs += 1
            if result.label == "contradiction" and result.confidence >= settings.nli_confidence_threshold:
                contradictions.append(nli_to_contradiction(result))
    except Exception as exc:
        logger.exception("Unhandled error in POST /check/nli-only")
        raise HTTPException(status_code=500, detail="NLI pipeline error") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "POST /check/nli-only  pairs=%d  found=%d  time=%.1fms",
        total_pairs, len(contradictions), elapsed_ms,
    )
    metadata = {"nli_pairs_checked": total_pairs, "nli_caught": len(contradictions), "llm_escalated": 0, "llm_caught": 0}
    return aggregator.aggregate(contradictions, metadata, elapsed_ms)


@router.post("/check/llm-only", response_model=ContradictionReport)
async def check_llm_only(
    body: CheckRequest,
    llm_judge: BaseLLMJudge = Depends(get_llm_judge),
    aggregator: Aggregator = Depends(get_aggregator),
) -> ContradictionReport:
    """Run LLM-only detection — the model sees the full context without NLI pre-filter.

    Passes an empty uncertain_pairs list so the judge reasons over the whole
    document independently.  Catches multi-hop contradictions that NLI misses
    because NLI only sees one sentence pair at a time.
    """
    t0 = time.perf_counter()
    try:
        contradictions = llm_judge.judge(context=body.context, response=body.response, candidate_pairs=[], uncertain_pairs=[])
    except _RateLimitError as exc: # type: ignore
        raise HTTPException(status_code=429, detail="LLM rate limit — retry after a moment") from exc
    except _APIError as exc: # type: ignore
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unhandled error in POST /check/llm-only")
        raise HTTPException(status_code=500, detail="LLM pipeline error") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    metadata = {"nli_pairs_checked": 0, "nli_caught": 0, "llm_escalated": 0, "llm_caught": len(contradictions)}
    return aggregator.aggregate(contradictions, metadata, elapsed_ms)
