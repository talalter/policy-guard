"""Unit tests for Aggregator — no model loading required."""

import pytest

from backend.models import Contradiction, DetectionMethod, Severity


def _make_contradiction(severity: Severity, method: DetectionMethod, confidence: float = 0.9) -> Contradiction:
    """Factory helper to build a minimal Contradiction for testing."""
    return Contradiction(
        response_span="span",
        context_span="span",
        explanation="test",
        severity=severity,
        method=method,
        confidence=confidence,
    )


def test_perfect_score_with_no_contradictions(aggregator):
    """Empty contradiction list yields a faithfulness score of 1.0."""
    report = aggregator.aggregate([], {"nli_pairs_checked": 5, "nli_caught": 0, "llm_escalated": 0, "llm_caught": 0}, 100.0)
    assert report.faithfulness_score == 1.0


def test_direct_contradiction_reduces_score(aggregator):
    """A DIRECT contradiction (penalty=0.30) lowers the score below 0.75."""
    c = _make_contradiction(Severity.DIRECT, DetectionMethod.NLI)
    report = aggregator.aggregate([c], {"nli_pairs_checked": 3, "nli_caught": 1, "llm_escalated": 0, "llm_caught": 0}, 50.0)
    assert report.faithfulness_score == pytest.approx(0.70, abs=0.01)


def test_score_never_goes_below_zero(aggregator):
    """Multiple severe contradictions floor at 0.0, never negative."""
    contradictions = [_make_contradiction(Severity.DIRECT, DetectionMethod.LLM) for _ in range(10)]
    report = aggregator.aggregate(contradictions, {"nli_pairs_checked": 0, "nli_caught": 0, "llm_escalated": 1, "llm_caught": 10}, 200.0)
    assert report.faithfulness_score >= 0.0


def test_method_inferred_as_ensemble_when_both_ran(aggregator):
    """Method is ENSEMBLE when both NLI and LLM contributed contradictions."""
    nli_c = _make_contradiction(Severity.PARTIAL, DetectionMethod.NLI)
    llm_c = _make_contradiction(Severity.MULTIHOP, DetectionMethod.LLM)
    report = aggregator.aggregate(
        [nli_c, llm_c],
        {"nli_pairs_checked": 4, "nli_caught": 1, "llm_escalated": 1, "llm_caught": 1},
        150.0,
    )
    assert report.method_used == DetectionMethod.ENSEMBLE


def test_processing_time_is_recorded(aggregator):
    """processing_time_ms is stored in the report."""
    report = aggregator.aggregate([], {"nli_pairs_checked": 2, "nli_caught": 0, "llm_escalated": 0, "llm_caught": 0}, 123.4)
    assert report.processing_time_ms == pytest.approx(123.4)
