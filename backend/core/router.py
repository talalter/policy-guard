"""Confidence-based router that orchestrates NLI scoring and LLM escalation.

Architecture:
    1. Stream NLIResult objects from NLIScorer in one pass.
    2. Branch each result into a 'confident' bucket (≥ threshold) or an
       'uncertain' bucket (< threshold) without buffering the full list.
    3. Convert confident contradiction pairs directly to Contradiction objects.
    4. Escalate uncertain pairs to LLMJudge in a single batched call.
    5. Merge both lists and deduplicate overlapping response spans using
       Jaccard similarity so the same contradiction is never shown twice.

This is a production ML routing pattern: keep cost near zero for the common
case (NLI resolves it), pay only for the hard cases (multi-hop reasoning).
"""

import logging

from backend.config import settings
from backend.core.llm_judge import LLMJudge  # type: ignore
from backend.core.nli_scorer import NLIScorer
from backend.models import Contradiction, DetectionMethod, NLIResult, Severity
from backend.utils.dedup import deduplicate

logger = logging.getLogger(__name__)

_THRESHOLD = settings.nli_confidence_threshold

# NLI confidence cutoff for DIRECT vs PARTIAL severity assignment.
_DIRECT_CONFIDENCE_CUTOFF = settings.direct_severity_threshold

# Minimum contradiction_score for a pair to be escalated to GPT-4o.
# Intentionally independent of _THRESHOLD — changing the confidence threshold
# should not silently move the escalation floor.
_ESCALATION_FLOOR = settings.nli_escalation_floor


def nli_to_contradiction(result: NLIResult) -> Contradiction:
    """Convert a high-confidence NLI contradiction result to a Contradiction object.

    Severity is inferred from confidence because NLI operates on sentence pairs
    and cannot detect multi-hop patterns — those are reserved for the LLM path.
    """
    severity = (
        Severity.DIRECT
        if result.confidence >= _DIRECT_CONFIDENCE_CUTOFF
        else Severity.PARTIAL
    )
    return Contradiction(
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
) -> tuple[list[Contradiction], list[NLIResult], int]:
    """Consume the NLI stream in one pass, partitioning into confident and uncertain.

    Returns:
        (nli_contradictions, uncertain_pairs, total_pairs_checked)
    """
    nli_contradictions: list[Contradiction] = []
    uncertain_pairs: list[NLIResult] = []
    total_pairs = 0

    for nli_result in nli_stream:
        total_pairs += 1
        if nli_result.label == "contradiction" and nli_result.confidence >= threshold:
            nli_contradictions.append(nli_to_contradiction(nli_result))
            logger.debug(
                "NLI confident contradiction (conf=%.2f): %r → %r",
                nli_result.confidence,
                nli_result.pair.premise[:60],
                nli_result.pair.hypothesis[:60],
            )
        elif nli_result.contradiction_score >= _ESCALATION_FLOOR:
            # Escalate pairs where NLI sees some contradiction signal but is not
            # confident enough — includes low-confidence "contradiction" labels.
            # Purely neutral pairs (low contradiction_score) are skipped entirely.
            uncertain_pairs.append(nli_result)
            logger.debug(
                "Escalating uncertain pair (contradiction_score=%.2f): %r",
                nli_result.contradiction_score,
                nli_result.pair.hypothesis[:60],
            )

    return nli_contradictions, uncertain_pairs, total_pairs


class Router:
    """Orchestrates NLIScorer and LLMJudge with confidence-based routing.

    Instantiates both sub-components once so their models stay resident in
    memory across multiple calls — critical for low-latency production use.
    """

    def __init__(self) -> None:
        """Load NLI and LLM components at construction time."""
        logger.info("Initialising Router (threshold=%.2f)", _THRESHOLD)
        self._scorer = NLIScorer()
        self._judge = LLMJudge()

    def get_scorer(self) -> NLIScorer:
        """Return the shared NLIScorer instance."""
        return self._scorer

    def get_judge(self) -> LLMJudge:
        """Return the shared LLMJudge instance."""
        return self._judge

    def route(
        self, context: str, response: str
    ) -> tuple[list[Contradiction], dict]:
        """Run the full detection pipeline and return contradictions + metadata.

        Steps:
            1. Stream NLI results and partition into confident / uncertain.
            2. Escalate uncertain pairs to GPT-4o in one batched call.
            3. Merge, deduplicate by span overlap, return sorted by confidence.

        Args:
            context: Source document the response should be faithful to.
            response: LLM-generated text under evaluation.

        Returns:
            A tuple of:
                - list[Contradiction] sorted by confidence descending.
                - dict with routing metadata for ContradictionReport.
        """
        nli_stream = self._scorer.score(context, response)
        nli_contradictions, uncertain_pairs, total_pairs = _partition_results(
            nli_stream, _THRESHOLD
        )

        logger.info(
            "NLI: %d pairs checked, %d confident contradictions, %d escalated",
            total_pairs,
            len(nli_contradictions),
            len(uncertain_pairs),
        )

        llm_contradictions: list[Contradiction] = []
        # Call the LLM when there are uncertain pairs to validate, OR when NLI
        # found zero confident contradictions — the latter ensures multi-hop cases
        # (where NLI sees no signal at the sentence-pair level) always reach GPT-4o
        # for a full-document reasoning pass.
        if uncertain_pairs or not nli_contradictions:
            llm_contradictions = self._judge.judge(
                context=context,
                response=response,
                uncertain_pairs=uncertain_pairs,
            )
            logger.info("LLM judge returned %d contradiction(s)", len(llm_contradictions))

        all_contradictions = deduplicate(nli_contradictions + llm_contradictions)
        all_contradictions.sort(key=lambda c: c.confidence, reverse=True)

        metadata = {
            "nli_pairs_checked": total_pairs,
            "nli_caught": len(nli_contradictions),
            "llm_escalated": len(uncertain_pairs),
            "llm_caught": len(llm_contradictions),
            "after_dedup": len(all_contradictions),
        }

        logger.info(
            "Router complete: %d unique contradiction(s) (nli=%d, llm=%d, dedup_dropped=%d)",
            len(all_contradictions),
            len(nli_contradictions),
            len(llm_contradictions),
            (len(nli_contradictions) + len(llm_contradictions)) - len(all_contradictions),
        )

        return all_contradictions, metadata
