"""Shared fixtures for integration tests.

All fixtures are session-scoped — ML models take ~30s to load and must not
be re-instantiated for every test.  Mark any test that uses these fixtures
with @pytest.mark.integration so they can be skipped in fast CI runs:

    pytest -m "not integration"   # unit tests only
    pytest -m integration         # full pipeline tests
"""

import json
import pathlib

import pytest

_EXAMPLES_PATH = pathlib.Path(__file__).parent.parent / "data" / "ragtruth_sample.json"


def pytest_configure(config):
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers", "integration: slow tests that load ML models or call external APIs"
    )


@pytest.fixture(scope="session")
def examples() -> list[dict]:
    """Load labeled examples from the benchmark dataset."""
    if not _EXAMPLES_PATH.exists():
        pytest.skip(f"Benchmark data not found: {_EXAMPLES_PATH}")
    with _EXAMPLES_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def nli_scorer():
    """Load NLIScorer once per test session — model download ~1.5 GB on first run."""
    from backend.core import NLIScorer
    return NLIScorer()


@pytest.fixture(scope="session")
def llm_judge():
    """Instantiate the configured LLM judge once per test session."""
    from backend.core.llm_judge import create_llm_judge
    return create_llm_judge()


@pytest.fixture(scope="session")
def router():
    """Load the full routing pipeline once per test session."""
    from backend.core import Router
    return Router()


@pytest.fixture(scope="session")
def aggregator():
    """Instantiate Aggregator once per test session."""
    from backend.core import Aggregator
    return Aggregator()
