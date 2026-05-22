"""Integration tests for the confidence-based Router pipeline.

Requires local model weights and a valid LLM API key.
Run with: pytest -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_routes_contradiction_via_ensemble(router):
    """A clear numerical contradiction is caught by the ensemble pipeline."""
    context = "The session token expires after 15 minutes of inactivity."
    response = "Session tokens expire after 60 minutes of inactivity."
    contradictions, metadata = router.route(context, response)
    assert contradictions, "Expected router to detect a timeout contradiction"
    assert metadata["llm_called"], "Expected LLM to be called for this case"


def test_metadata_keys_are_present(router):
    """route() always returns the expected metadata keys."""
    context = "The endpoint accepts GET requests only."
    response = "The endpoint accepts GET requests only."
    _, metadata = router.route(context, response)
    expected_keys = {
        "nli_pairs_checked", "nli_candidates", "llm_escalated",
        "llm_called", "llm_caught", "after_dedup",
    }
    assert expected_keys.issubset(metadata.keys()), f"Missing keys: {expected_keys - metadata.keys()}"


def test_faithful_response_returns_empty(router):
    """A faithful response produces no contradictions."""
    context = "The API uses HTTPS for all connections."
    response = "All API connections use HTTPS."
    contradictions, _ = router.route(context, response)
    assert not contradictions, f"Faithful response incorrectly flagged: {contradictions}"


def test_benchmark_accuracy(router, examples):
    """Full ensemble achieves at least 85% accuracy on the RAGTruth benchmark sample."""
    correct = 0
    for ex in examples:
        contradictions, _ = router.route(ex["context"], ex["response"])
        if bool(contradictions) == ex["has_contradiction"]:
            correct += 1
    accuracy = correct / len(examples)
    assert accuracy >= 0.85, f"Router accuracy {accuracy:.1%} is below the 85% floor on {len(examples)} examples"
