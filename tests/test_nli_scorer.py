"""Integration tests for NLIScorer.

Requires local model weights (~1.5 GB, cached after first download).
Run with: pytest -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_flags_direct_numerical_contradiction(nli_scorer):
    """A clear numeric mismatch is flagged at high confidence."""
    context = "The API rate limit is 100 requests per minute."
    response = "The API rate limit is 1000 requests per minute."
    results = list(nli_scorer.score(context, response))
    contradictions = [r for r in results if r.label == "contradiction" and r.confidence >= 0.75]
    assert contradictions, "Expected NLI to flag a 10× rate limit discrepancy"


def test_passes_faithful_paraphrase(nli_scorer):
    """A faithful paraphrase is not flagged as a contradiction."""
    context = "Authentication uses OAuth 2.0 bearer tokens."
    response = "The API authenticates requests with OAuth 2.0 bearer tokens."
    results = list(nli_scorer.score(context, response))
    contradictions = [r for r in results if r.label == "contradiction" and r.confidence >= 0.75]
    assert not contradictions, f"Faithful paraphrase was incorrectly flagged: {contradictions}"


def test_streams_results(nli_scorer):
    """score() returns an iterator, not a list."""
    context = "The timeout is 30 seconds."
    response = "The timeout is 60 seconds."
    stream = nli_scorer.score(context, response)
    assert hasattr(stream, "__iter__"), "score() must return an iterable"
    results = list(stream)
    assert len(results) > 0, "Expected at least one scored pair"


def test_benchmark_accuracy(nli_scorer, examples):
    """NLI achieves at least 70% accuracy on the RAGTruth benchmark sample."""
    correct = 0
    for ex in examples:
        results = list(nli_scorer.score(ex["context"], ex["response"]))
        found = any(r.label == "contradiction" and r.confidence >= 0.75 for r in results)
        if found == ex["has_contradiction"]:
            correct += 1
    accuracy = correct / len(examples)
    assert accuracy >= 0.70, f"NLI accuracy {accuracy:.1%} is below the 70% floor on {len(examples)} examples"
