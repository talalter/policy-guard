"""Aggregator: converts a raw contradiction list into a scored ContradictionReport.

Responsibilities:
    - Apply a weighted severity penalty to compute a faithfulness_score in [0, 1].
    - Infer which DetectionMethod(s) produced the contradictions.
    - Assemble and return a fully populated ContradictionReport.

Kept deliberately stateless — every method is a pure function except the
public ``aggregate`` entry point, which just delegates to them.
"""

import logging
from typing import Final

from backend.models import (
    Contradiction,
    ContradictionReport,
    DetectionMethod,
    Severity,
)

logger = logging.getLogger(__name__)

# Penalty subtracted from the faithfulness score per contradiction by severity.
# Direct contradiction is the harshest; partial is the mildest.
SEVERITY_PENALTIES: Final[dict[Severity, float]] = {
    Severity.DIRECT: 0.30,
    Severity.PARTIAL: 0.15,
    Severity.MULTIHOP: 0.20,
}


def _compute_faithfulness(contradictions: list[Contradiction]) -> float:
    """Return a faithfulness score in [0.0, 1.0] using weighted severity penalties.

    Formula:
        score = max(0.0, 1.0 - sum(penalty(c) for c in contradictions))

    A response with no contradictions scores 1.0; enough severe contradictions
    floor the score at 0.0.
    """
    total_penalty = sum(SEVERITY_PENALTIES[c.severity] for c in contradictions)
    faithfulness_score = max(0.0, 1.0 - total_penalty)
    logger.debug(
        "Faithfulness: %.3f (total_penalty=%.3f, contradictions=%d)",
        faithfulness_score,
        total_penalty,
        len(contradictions),
    )
    return round(faithfulness_score, 4)


def _infer_method(
    contradictions: list[Contradiction],
    metadata: dict,
) -> DetectionMethod:
    """Determine which DetectionMethod(s) contributed to the final report.

    Priority:
        1. If contradictions carry fingerprints from both NLI and LLM → ENSEMBLE.
        2. If only one method appears in the contradiction list → that method.
        3. If the list is empty, fall back to metadata to distinguish a clean
           NLI-only run from a clean ensemble run.
    """
    if not contradictions:
        # No contradictions found; infer from whether LLM was ever invoked.
        return (
            DetectionMethod.ENSEMBLE
            if metadata.get("llm_escalated", 0) > 0
            else DetectionMethod.NLI
        )

    methods_used = {c.method for c in contradictions}

    if len(methods_used) > 1:
        return DetectionMethod.ENSEMBLE
    sole_method = next(iter(methods_used))
    # Even if all *caught* contradictions came from one method, if the other
    # method ran and escalated pairs it still counts as an ensemble run.
    if sole_method == DetectionMethod.NLI and metadata.get("llm_escalated", 0) > 0:
        return DetectionMethod.ENSEMBLE
    return sole_method


class Aggregator:
    """Builds a ContradictionReport from a contradiction list and routing metadata."""

    def aggregate(
        self,
        contradictions: list[Contradiction],
        metadata: dict,
        processing_time_ms: float,
    ) -> ContradictionReport:
        """Compute faithfulness score and assemble the final ContradictionReport.

        Args:
            contradictions: Deduplicated list from Router (or a single method).
            metadata: Routing metadata dict with keys:
                        nli_pairs_checked, nli_caught, llm_escalated, llm_caught.
            processing_time_ms: Wall-clock time for the full pipeline call.

        Returns:
            A fully populated ContradictionReport.
        """
        faithfulness_score = _compute_faithfulness(contradictions)
        method_used = _infer_method(contradictions, metadata)

        report = ContradictionReport(
            faithfulness_score=faithfulness_score,
            contradictions=contradictions,
            method_used=method_used,
            nli_pairs_checked=metadata.get("nli_pairs_checked", 0),
            llm_escalations=metadata.get("llm_escalated", 0),
            processing_time_ms=round(processing_time_ms, 2),
        )

        logger.info(
            "Report assembled: faithfulness_score=%.4f, method=%s, contradictions=%d, "
            "nli_pairs=%d, llm_escalations=%d, time=%.1fms",
            report.faithfulness_score,
            report.method_used,
            len(report.contradictions),
            report.nli_pairs_checked,
            report.llm_escalations,
            report.processing_time_ms,
        )

        return report
