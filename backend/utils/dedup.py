"""Deduplication utilities: Jaccard similarity and contradiction span matching."""

from backend.models import Contradiction


def tokenize(text: str) -> set[str]:
    """Return a lowercase token set for Jaccard computation."""
    return {t.lower() for t in text.split() if t.isalpha()}


def jaccard(a: str, b: str) -> float:
    """Compute Jaccard token overlap between two spans (0 = disjoint, 1 = identical)."""
    tokens_a, tokens_b = tokenize(a), tokenize(b)
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def deduplicate(
    contradictions: list[Contradiction],
    threshold: float = 0.5,
) -> list[Contradiction]:
    """Remove contradictions whose response_span overlaps a higher-confidence finding.

    Iterates the list (sorted confidence-descending) and drops any entry whose
    response_span has Jaccard similarity ≥ threshold with an already-accepted span.
    This prevents showing the same surface error twice when both NLI and the LLM flag it.

    Args:
        contradictions: List of Contradiction objects to deduplicate.
        threshold: Jaccard similarity threshold for deduplication.

    Returns:
        Deduplicated list sorted by confidence descending.
    """
    sorted_by_conf = sorted(contradictions, key=lambda c: c.confidence, reverse=True)
    deduplicated: list[Contradiction] = []

    for candidate in sorted_by_conf:
        is_duplicate = any(
            jaccard(candidate.response_span, accepted.response_span) >= threshold
            for accepted in deduplicated
        )
        if not is_duplicate:
            deduplicated.append(candidate)

    return deduplicated
