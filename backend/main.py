"""FastAPI application factory and lifespan.

ML models are loaded once inside the lifespan context manager and stored on
app.state.  All route handlers and dependency providers live in backend/api/.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.benchmark import router as benchmark_router
from backend.api.routes.check import router as check_router
from backend.api.routes.feedback import router as feedback_router
from backend.api.routes.history import router as history_router
from backend.config import settings
from backend.core import Aggregator, Router
from backend.db import connect as db_connect, disconnect as db_disconnect

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and connect to MongoDB on startup; clean up on shutdown.

    Router.__init__ loads both NLIScorer and LLMJudge.  We reuse those
    instances for the nli-only and llm-only endpoints so model weights are
    loaded exactly once per process.  MongoDB connection is best-effort -
    the app starts successfully even if no Mongo instance is available.
    """
    logger.info("Loading models - this may take a moment on first run.")
    pipeline = Router()
    app.state.router = pipeline
    app.state.nli_scorer = pipeline.get_scorer()
    app.state.llm_judge = pipeline.get_judge()
    app.state.aggregator = Aggregator()
    app.state.db = await db_connect()
    logger.info("All components ready.  Application is accepting requests.")
    yield
    db_disconnect()
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Extracted into a factory so test suites can call create_app() with
    overridden app.state rather than importing the module-level `app` object.
    """
    _app = FastAPI(
        title="Policy Guard",
        description=(
            "Runtime guardrail that reads existing policy documents and detects "
            "when an AI agent's planned actions violate them - no manual rule encoding required."
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
    _app.include_router(check_router)
    _app.include_router(feedback_router)
    _app.include_router(history_router)
    _app.include_router(benchmark_router)

    @_app.get("/health")
    async def health() -> dict:
        """Liveness probe - returns 200 once the app is ready."""
        return {"status": "ok"}

    return _app


app = create_app()
