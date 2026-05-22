"""Integration tests for LLM judge providers.

Requires a valid API key (OPENAI_API_KEY or ANTHROPIC_API_KEY depending on
LLM_PROVIDER setting).
Run with: pytest -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_flags_direct_contradiction(llm_judge):
    """Judge flags an explicit factual contradiction with high confidence."""
    context = "The maximum payload size for the API is 10 MB."
    response = "You can send payloads up to 100 MB to the API."
    contradictions = llm_judge.judge(context=context, response=response, candidate_pairs=[], uncertain_pairs=[])
    assert contradictions, "Expected LLM judge to flag a 10× payload size discrepancy"
    assert contradictions[0].confidence >= 0.85


def test_passes_faithful_summary(llm_judge):
    """Judge does not flag a faithful summary as a contradiction."""
    context = "Rate limiting is applied per API key. The default limit is 60 requests per minute."
    response = "Each API key is subject to rate limiting, with a default of 60 requests per minute."
    contradictions = llm_judge.judge(context=context, response=response, candidate_pairs=[], uncertain_pairs=[])
    assert not contradictions, f"Faithful summary was incorrectly flagged: {contradictions}"


def test_drops_paraphrase_findings(llm_judge):
    """Judge drops findings it classifies as paraphrases via is_paraphrase_or_equivalent."""
    context = "Responses are returned in descending order by creation date."
    response = "Results are sorted newest first."
    contradictions = llm_judge.judge(context=context, response=response, candidate_pairs=[], uncertain_pairs=[])
    assert not contradictions, f"Synonym/paraphrase was flagged as a contradiction: {contradictions}"


def test_benchmark_accuracy(llm_judge, examples):
    """LLM judge achieves at least 75% accuracy on the RAGTruth benchmark sample."""
    correct = 0
    for ex in examples:
        contradictions = llm_judge.judge(
            context=ex["context"],
            response=ex["response"],
            candidate_pairs=[], uncertain_pairs=[],
        )
        if bool(contradictions) == ex["has_contradiction"]:
            correct += 1
    accuracy = correct / len(examples)
    assert accuracy >= 0.75, f"LLM judge accuracy {accuracy:.1%} is below the 75% floor on {len(examples)} examples"
