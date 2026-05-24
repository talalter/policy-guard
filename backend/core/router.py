"""Confidence-based router that orchestrates NLI scoring and LLM arbitration.

Architecture:
    1. Stream NLIResult objects from NLIScorer in one pass.
    2. Branch each result into a 'candidate' bucket (≥ threshold) or an
       'uncertain' bucket (≥ escalation_floor) without buffering the full list.
    3. Pass all NLI pairs (candidates + uncertain) to the LLM judge as hints.
    4. The LLM makes every final output decision - NLI narrows the search space,
       never bypasses review.
    5. Deduplicate overlapping response spans using Jaccard similarity.

NLI's role is pre-filtering: finding sentence pairs worth examining, and skipping
the LLM entirely when the document is clearly neutral (peak NLI score below floor).
"""

import logging

from backend.config import settings
from backend.core.llm_judge import BaseLLMJudge, create_llm_judge
from backend.core.nli_scorer import NLIScorer
from backend.models import Violation, DetectionMethod, NLIResult, Severity
from backend.utils.dedup import deduplicate

logger = logging.getLogger(__name__)

_THRESHOLD = settings.nli_confidence_threshold
_BLOCKING_CONFIDENCE_CUTOFF = settings.direct_severity_threshold
_ESCALATION_FLOOR = settings.nli_escalation_floor
_LLM_SIGNAL_FLOOR = settings.llm_signal_floor
_FORCE_LLM = settings.force_llm


def nli_to_violation(result: NLIResult) -> Violation:
    """Convert a high-confidence NLI contradiction result to a Violation object.

    Used by the benchmark's NLI-only evaluation path. Not used in the ensemble
    route() - NLI candidates are passed to the LLM as hints there.
    """
    severity = (
        Severity.BLOCKING
        if result.confidence >= _BLOCKING_CONFIDENCE_CUTOFF
        else Severity.WARNING
    )
    return Violation(
        response_span=result.pair.hypothesis,
        context_span=result.pair.premise,
        explanation=(
            f"NLI model classified this pair as contradiction "
            f"(confidence {result.confidence:.0%})."
        ),
        severity=severity,
        method=DetectionMethod.NLI,
        confidence=result.confidence,
    )


def _partition_results(
    nli_stream,
    threshold: float,
) -> tuple[list[NLIResult], list[NLIResult], int, float]:
    """Consume the NLI stream in one pass, partitioning into candidates and uncertain.

    Candidates: NLI is confident (label=contradiction AND confidence ≥ threshold).
    Uncertain:  NLI sees signal but is not confident (contradiction_score ≥ floor).
    Both lists are passed to the LLM judge as hints; neither is output directly.

    Returns:
        (candidate_pairs, uncertain_pairs, total_pairs_checked, max_contradiction_score)
    """
    candidate_pairs: list[NLIResult] = []
    uncertain_pairs: list[NLIResult] = []
    total_pairs = 0
    max_contradiction_score = 0.0

    for nli_result in nli_stream:
        total_pairs += 1
        max_contradiction_score = max(max_contradiction_score, nli_result.contradiction_score)
        if nli_result.label == "contradiction" and nli_result.confidence >= threshold:
            candidate_pairs.append(nli_result)
            logger.debug(
                "NLI candidate (conf=%.2f): %r → %r",
                nli_result.confidence,
                nli_result.pair.premise[:60],
                nli_result.pair.hypothesis[:60],
            )
        elif nli_result.contradiction_score >= _ESCALATION_FLOOR:
            # NLI sees some contradiction signal but is not confident - send to LLM.
            # Purely neutral pairs (low contradiction_score) are skipped entirely.
            uncertain_pairs.append(nli_result)
            logger.debug(
                "Uncertain pair (contradiction_score=%.2f): %r",
                nli_result.contradiction_score,
                nli_result.pair.hypothesis[:60],
            )

    return candidate_pairs, uncertain_pairs, total_pairs, max_contradiction_score


class Router:
    """Orchestrates NLIScorer and LLMJudge with NLI pre-filtering.

    Instantiates both sub-components once so their models stay resident in
    memory across multiple calls - critical for low-latency production use.
    """

    def __init__(self) -> None:
        """Load NLI and LLM components at construction time."""
        logger.info("Initialising Router (threshold=%.2f)", _THRESHOLD)
        self._scorer = NLIScorer()
        self._judge = create_llm_judge()

    def get_scorer(self) -> NLIScorer:
        """Return the shared NLIScorer instance."""
        return self._scorer

    def get_judge(self) -> BaseLLMJudge:
        """Return the shared LLM judge instance."""
        return self._judge

    def route(
        self, context: str, response: str
    ) -> tuple[list[Violation], dict]:
        """Run the full detection pipeline and return violations + metadata.

        Steps:
            1. Stream NLI results and partition into candidates / uncertain.
            2. Pass all NLI pairs to the LLM judge as focused hints.
            3. Deduplicate by span overlap, return sorted by confidence.

        Args:
            context: Source document the response should be faithful to.
            response: LLM-generated text under evaluation.

        Returns:
            A tuple of:
                - list[Violation] sorted by confidence descending.
                - dict with routing metadata for ViolationReport.
        """
        nli_stream = self._scorer.score(context, response)
        candidate_pairs, uncertain_pairs, total_pairs, max_nli_score = _partition_results(
            nli_stream, _THRESHOLD
        )

        logger.info(
            "NLI: %d pairs checked, %d candidates, %d uncertain, peak_score=%.2f",
            total_pairs,
            len(candidate_pairs),
            len(uncertain_pairs),
            max_nli_score,
        )

        llm_should_run = _FORCE_LLM or max_nli_score >= _LLM_SIGNAL_FLOOR

        if not llm_should_run:
            logger.info(
                "LLM skipped - peak NLI score %.2f is below signal floor %.2f",
                max_nli_score,
                _LLM_SIGNAL_FLOOR,
            )
            return [], {
                "nli_pairs_checked": total_pairs,
                "nli_candidates": 0,
                "llm_escalated": 0,
                "llm_called": False,
                "llm_caught": 0,
                "after_dedup": 0,
            }

        llm_violations = self._judge.judge(
            context=context,
            response=response,
            candidate_pairs=candidate_pairs,
            uncertain_pairs=uncertain_pairs,
        )
        logger.info("LLM judge returned %d violation(s)", len(llm_violations))

        all_violations = deduplicate(llm_violations)
        all_violations.sort(key=lambda v: v.confidence, reverse=True)

        usage = self._judge.get_last_usage()
        metadata = {
            "nli_pairs_checked": total_pairs,
            "nli_candidates": len(candidate_pairs),
            "llm_escalated": len(uncertain_pairs),
            "llm_called": True,
            "llm_caught": len(llm_violations),
            "after_dedup": len(all_violations),
            "overall_reasoning": self._judge.get_last_reasoning(),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

        logger.info(
            "Router complete: %d unique violation(s) (llm=%d, dedup_dropped=%d)",
            len(all_violations),
            len(llm_violations),
            len(llm_violations) - len(all_violations),
        )

        return all_violations, metadata
