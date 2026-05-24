"""Unit tests for Aggregator - no model loading required."""

import pytest

from backend.models import Violation, DetectionMethod, Severity


def _make_violation(severity: Severity, method: DetectionMethod, confidence: float = 0.9) -> Violation:
    """Factory helper to build a minimal Violation for testing."""
    return Violation(
        response_span="span",
        context_span="span",
        explanation="test",
        severity=severity,
        method=method,
        confidence=confidence,
    )


def test_perfect_score_with_no_violations(aggregator):
    """Empty violation list yields a compliance score of 1.0 (empty product)."""
    report = aggregator.aggregate([], {"nli_pairs_checked": 5, "nli_caught": 0, "llm_escalated": 0, "llm_caught": 0}, 100.0)
    assert report.compliance_score == 1.0


def test_blocking_violation_reduces_score(aggregator):
    """A BLOCKING violation (weight=1.0) at confidence=0.9 → score = 1 - 1.0×0.9 = 0.10."""
    v = _make_violation(Severity.BLOCKING, DetectionMethod.NLI, confidence=0.9)
    report = aggregator.aggregate([v], {"nli_pairs_checked": 3, "nli_caught": 1, "llm_escalated": 0, "llm_caught": 0}, 50.0)
    assert report.compliance_score == pytest.approx(0.10, abs=0.01)


def test_blocking_violation_at_full_confidence_gives_zero(aggregator):
    """A BLOCKING violation at confidence=1.0 → score = 0.0 (hard fail by the formula)."""
    v = _make_violation(Severity.BLOCKING, DetectionMethod.LLM, confidence=1.0)
    report = aggregator.aggregate([v], {"nli_pairs_checked": 0, "nli_caught": 0, "llm_escalated": 1, "llm_caught": 1}, 50.0)
    assert report.compliance_score == pytest.approx(0.0, abs=0.001)


def test_confidence_weights_the_penalty(aggregator):
    """Two WARNING violations at confidence=0.9 compound multiplicatively: (1-0.4×0.9)² ≈ 0.41."""
    violations = [_make_violation(Severity.WARNING, DetectionMethod.LLM, confidence=0.9) for _ in range(2)]
    report = aggregator.aggregate(violations, {"nli_pairs_checked": 0, "nli_caught": 0, "llm_escalated": 1, "llm_caught": 2}, 100.0)
    assert report.compliance_score == pytest.approx(0.41, abs=0.01)


def test_method_inferred_as_ensemble_when_both_ran(aggregator):
    """Method is ENSEMBLE when both NLI and LLM contributed violations."""
    nli_v = _make_violation(Severity.WARNING, DetectionMethod.NLI)
    llm_v = _make_violation(Severity.INFERRED, DetectionMethod.LLM)
    report = aggregator.aggregate(
        [nli_v, llm_v],
        {"nli_pairs_checked": 4, "nli_caught": 1, "llm_escalated": 1, "llm_caught": 1},
        150.0,
    )
    assert report.method_used == DetectionMethod.ENSEMBLE


def test_processing_time_is_recorded(aggregator):
    """processing_time_ms is stored in the report."""
    report = aggregator.aggregate([], {"nli_pairs_checked": 2, "nli_caught": 0, "llm_escalated": 0, "llm_caught": 0}, 123.4)
    assert report.processing_time_ms == pytest.approx(123.4)
