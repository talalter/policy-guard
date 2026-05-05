"""FastAPI application: exposes the contradiction-detector pipeline as an HTTP API.

Three check endpoints let the frontend and benchmark runner call each detection
method independently for comparison:
    POST /check           — full ensemble (NLI + GPT-4o routing)
    POST /check/nli-only  — DeBERTa only, no GPT-4o escalation
    POST /check/llm-only  — GPT-4o only, no NLI pre-filter
    GET  /health          — liveness probe for deployment health checks

Architecture notes:
    - Models are loaded once inside a lifespan context manager (startup/shutdown).
    - All heavy objects live in app.state and are injected via FastAPI's Depends().
    - The NLI and LLM sub-components are reused from the Router instance so the
      model weights are loaded exactly once, regardless of which endpoint is called.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import openai  # type: ignore
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.core import Aggregator, LLMJudge, NLIScorer, Router, nli_to_contradiction
from backend.models import (
    BenchmarkResult,
    CheckRequest,
    Contradiction,
    ContradictionReport,
)

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── lifespan: load models once at startup ────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models on startup; release resources on shutdown.

    Router.__init__ loads both NLIScorer and LLMJudge.  We reuse those
    instances for the nli-only and llm-only endpoints so model weights are
    loaded exactly once per process.
    """
    logger.info("Loading models — this may take a moment on first run.")
    router = Router()
    app.state.router = router
    app.state.nli_scorer = router.get_scorer()  # reuse; avoids loading weights twice
    app.state.llm_judge = router.get_judge()
    app.state.aggregator = Aggregator()
    logger.info("All components ready.  Application is accepting requests.")
    yield
    logger.info("Application shutting down.")


# ── app factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Extracted into a factory so test suites can call create_app() with
    overridden app.state rather than importing the module-level `app` object.
    """
    _app = FastAPI(
        title="Contradiction Detector",
        description=(
            "Detects when an LLM response contradicts its source context "
            "using a NLI model with optional GPT-4o escalation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [u.strip() for u in settings.frontend_url.split(",") if u.strip()]

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app


app = create_app()


# ── dependency providers ──────────────────────────────────────────────────────


def get_router(request: Request) -> Router:
    """Inject the Router instance from app state."""
    return request.app.state.router


def get_nli_scorer(request: Request) -> NLIScorer:
    """Inject the NLIScorer instance from app state."""
    return request.app.state.nli_scorer


def get_llm_judge(request: Request) -> LLMJudge:
    """Inject the LLMJudge instance from app state."""
    return request.app.state.llm_judge


def get_aggregator(request: Request) -> Aggregator:
    """Inject the Aggregator instance from app state."""
    return request.app.state.aggregator


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 once the app is ready."""
    return {"status": "ok"}


@app.post("/check", response_model=ContradictionReport)
async def check(
    body: CheckRequest,
    router: Router = Depends(get_router),
    aggregator: Aggregator = Depends(get_aggregator),
) -> ContradictionReport:
    """Run the full ensemble pipeline (NLI + GPT-4o confidence-based routing).

    Confident NLI pairs are resolved locally for free.  Uncertain pairs are
    escalated to GPT-4o, which catches multi-hop contradictions NLI misses.
    """
    t0 = time.perf_counter()
    try:
        contradictions, metadata = router.route(body.context, body.response)
    except openai.RateLimitError as exc:
        raise HTTPException(status_code=429, detail="OpenAI rate limit — retry after a moment") from exc
    except openai.APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unhandled error in POST /check")
        raise HTTPException(status_code=500, detail="Pipeline error") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return aggregator.aggregate(contradictions, metadata, elapsed_ms)


@app.post("/check/nli-only", response_model=ContradictionReport)
async def check_nli_only(
    body: CheckRequest,
    nli_scorer: NLIScorer = Depends(get_nli_scorer),
    aggregator: Aggregator = Depends(get_aggregator),
) -> ContradictionReport:
    """Run NLI-only detection — no GPT-4o escalation.

    All sentence pairs are scored by DeBERTa.  Every pair labelled
    'contradiction' with confidence ≥ NLI_CONFIDENCE_THRESHOLD is returned
    as a finding.  Used by the benchmark tab to isolate NLI performance.
    """
    logger.debug(
        "POST /check/nli-only  context=%d chars (%r...)  response=%d chars (%r...)",
        len(body.context),
        body.context[:60],
        len(body.response),
        body.response[:60],
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
        "POST /check/nli-only done  pairs_scored=%d  contradictions_found=%d  time=%.1fms",
        total_pairs,
        len(contradictions),
        elapsed_ms,
    )

    metadata = {
        "nli_pairs_checked": total_pairs,
        "nli_caught": len(contradictions),
        "llm_escalated": 0,
        "llm_caught": 0,
    }
    return aggregator.aggregate(contradictions, metadata, elapsed_ms)


def _data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


def _dataset_key(path: Path) -> str:
    """Extract the dataset key from a benchmark results filename."""
    return path.stem.removeprefix("benchmark_results_")


def _dataset_label(key: str) -> str:
    return key.replace("_", " ").title()


def _available_datasets() -> list[dict]:
    candidates = sorted(_data_dir().glob("benchmark_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"key": _dataset_key(p), "label": _dataset_label(_dataset_key(p))} for p in candidates]


@app.get("/benchmark-datasets")
async def get_benchmark_datasets() -> list[dict]:
    """List available benchmark datasets by scanning data/benchmark_results_*.json files."""
    datasets = _available_datasets()
    if not datasets:
        raise HTTPException(status_code=404, detail="No benchmark results found. Run: python -m backend.benchmark")
    return datasets


@app.get("/benchmark-results", response_model=list[BenchmarkResult])
async def get_benchmark_results(dataset: str | None = None) -> list[BenchmarkResult]:
    """Return saved benchmark results for the given dataset key.

    If no dataset is specified, returns the most recently generated file.
    Run ``python -m backend.benchmark`` first to generate files.
    """
    data_dir = _data_dir()
    if dataset:
        results_path = data_dir / f"benchmark_results_{dataset}.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")
    else:
        candidates = sorted(data_dir.glob("benchmark_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise HTTPException(status_code=404, detail="No benchmark results found. Run: python -m backend.benchmark")
        results_path = candidates[0]
    with open(results_path) as f:
        return [BenchmarkResult(**row) for row in json.load(f)]


@app.post("/check/llm-only", response_model=ContradictionReport)
async def check_llm_only(
    body: CheckRequest,
    llm_judge: LLMJudge = Depends(get_llm_judge),
    aggregator: Aggregator = Depends(get_aggregator),
) -> ContradictionReport:
    """Run LLM-only detection — GPT-4o sees the full context without NLI pre-filter.

    Passes an empty uncertain_pairs list so the judge reasons over the whole
    document independently.  Catches multi-hop contradictions that NLI misses
    because NLI only sees one sentence pair at a time.
    """
    t0 = time.perf_counter()
    try:
        contradictions = llm_judge.judge(
            context=body.context,
            response=body.response,
            uncertain_pairs=[],
        )
    except openai.RateLimitError as exc:
        raise HTTPException(status_code=429, detail="OpenAI rate limit — retry after a moment") from exc
    except openai.APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unhandled error in POST /check/llm-only")
        raise HTTPException(status_code=500, detail="LLM pipeline error") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    metadata = {
        "nli_pairs_checked": 0,
        "nli_caught": 0,
        "llm_escalated": 0,
        "llm_caught": len(contradictions),
    }
    return aggregator.aggregate(contradictions, metadata, elapsed_ms)
