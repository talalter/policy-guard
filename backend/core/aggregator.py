"""Aggregator: converts a raw violation list into a scored ViolationReport.

Responsibilities:
    - Compute a compliance_score in [0, 1] via confidence-weighted survival product.
    - Infer which DetectionMethod(s) produced the violations.
    - Assemble and return a fully populated ViolationReport.

Kept deliberately stateless - every method is a pure function except the
public ``aggregate`` entry point, which just delegates to them.
"""

import logging
import math
from typing import Final

from backend.config import settings
from backend.models import (
    Violation,
    ViolationReport,
    DetectionMethod,
    Severity,
)

logger = logging.getLogger(__name__)

# Risk weight per severity level, used in the survival-product formula.
# Interpretation: probability-of-block per unit confidence.
# BLOCKING weight=1.0: one certain BLOCKING violation → compliance_score=0 by design.
SEVERITY_WEIGHTS: Final[dict[Severity, float]] = {
    Severity.BLOCKING: 0.85,  # one certain violation → ~15%; three → ~0%
    Severity.INFERRED: 0.50,  # derived from multiple rules; one → ~50%
    Severity.WARNING:  0.20,  # partial restriction; one → ~80%
}


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute exact LLM cost from API token counts and configured per-token prices."""
    return round(
        input_tokens * settings.llm_input_cost_per_token
        + output_tokens * settings.llm_output_cost_per_token,
        8,
    )


def _compute_compliance_score(violations: list[Violation]) -> float:
    """Return a compliance score in [0.0, 1.0] using a confidence-weighted survival product.

    Formula:
        score = product(1.0 - SEVERITY_WEIGHTS[v.severity] * v.confidence
                        for v in violations)

    Probabilistic interpretation: models P(action is compliant) as the joint
    probability that no violation independently blocks execution.  Violations are
    assumed independent after Jaccard deduplication, making the product formula exact
    under this model.

    Properties:
    - No violations → 1.0 (empty product).
    - Confidence-calibrated: a detection at 0.87 penalises less than one at 0.99.
    - Bounded to [0, 1] by construction — no clamping needed.
    - Diminishing returns: each additional violation reduces the remaining compliant
      probability mass by a smaller absolute amount (mathematically principled).
    - BLOCKING at confidence=1.0 → 1 - 1.0×1.0 = 0.0 (hard fail, naturally).
    """
    score = math.prod(1.0 - SEVERITY_WEIGHTS[v.severity] * v.confidence for v in violations)
    logger.debug("Compliance score: %.4f  violations=%d", score, len(violations))
    return round(score, 4)


def _infer_method(
    violations: list[Violation],
    metadata: dict,
) -> DetectionMethod:
    """Determine which DetectionMethod(s) contributed to the final report.

    Priority:
        1. If violations carry fingerprints from both NLI and LLM → ENSEMBLE.
        2. If only one method appears in the violation list → that method.
        3. If the list is empty, fall back to metadata to distinguish a clean
           NLI-only run from a clean ensemble run.
    """
    if not violations:
        # No violations found; infer from whether LLM was ever invoked.
        return (
            DetectionMethod.ENSEMBLE
            if metadata.get("llm_called", False)
            else DetectionMethod.NLI
        )

    methods_used = {v.method for v in violations}

    if len(methods_used) > 1:
        return DetectionMethod.ENSEMBLE
    sole_method = next(iter(methods_used))
    # Even if all caught violations came from one method, if both NLI and
    # LLM ran it counts as an ensemble run.  The mirror check covers the
    # common case: ensemble route where LLM catches everything but NLI still
    # pre-filtered (nli_pairs_checked > 0 distinguishes from llm-only).
    if metadata.get("llm_called", False) and metadata.get("nli_pairs_checked", 0) > 0:
        return DetectionMethod.ENSEMBLE
    return sole_method


class Aggregator:
    """Builds a ViolationReport from a violation list and routing metadata."""

    def aggregate(
        self,
        violations: list[Violation],
        metadata: dict,
        processing_time_ms: float,
    ) -> ViolationReport:
        """Compute compliance score and assemble the final ViolationReport.

        Args:
            violations: Deduplicated list from Router (or a single method).
            metadata: Routing metadata dict with keys:
                        nli_pairs_checked, nli_caught, llm_escalated, llm_caught.
            processing_time_ms: Wall-clock time for the full pipeline call.

        Returns:
            A fully populated ViolationReport.
        """
        compliance_score = _compute_compliance_score(violations)
        method_used = _infer_method(violations, metadata)

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        report = ViolationReport(
            compliance_score=compliance_score,
            violations=violations,
            method_used=method_used,
            nli_pairs_checked=metadata.get("nli_pairs_checked", 0),
            nli_candidates=metadata.get("nli_candidates", 0),
            llm_escalations=metadata.get("llm_escalated", 0),
            processing_time_ms=round(processing_time_ms, 2),
            overall_reasoning=metadata.get("overall_reasoning") or None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_compute_cost(input_tokens, output_tokens),
        )

        logger.info(
            "Report assembled: compliance_score=%.4f, method=%s, violations=%d, "
            "nli_pairs=%d, llm_escalations=%d, tokens=%d+%d, cost=$%.8f, time=%.1fms",
            report.compliance_score,
            report.method_used,
            len(report.violations),
            report.nli_pairs_checked,
            report.llm_escalations,
            report.input_tokens,
            report.output_tokens,
            report.cost_usd,
            report.processing_time_ms,
        )

        return report
