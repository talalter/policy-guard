"""GPT-4o structured-output judge for contradiction detection.

Uses constrained decoding (client.beta.chat.completions.parse) so the model
cannot return a malformed response.  A `reasoning` field forces chain-of-thought
*before* the structured findings, which measurably improves accuracy on
multi-hop contradictions that NLI misses.
"""

import logging

from openai import OpenAI
from pydantic import BaseModel

from backend.config import settings
from backend.models import Contradiction, DetectionMethod, NLIResult, Severity

logger = logging.getLogger(__name__)

_GPT_MODEL = settings.gpt_model
_LLM_MIN_CONFIDENCE = settings.llm_min_confidence

_SYSTEM_PROMPT = """\
You are a strict fact-checker.  Your default answer is: no contradictions found.
Only add a finding if you can complete this sentence without hedging:

  "The response states [X], but the context explicitly says [Y], and X and Y
   cannot both be true."

Do NOT add a finding when:
- The response uses a synonym or informal equivalent (e.g. "newest first" for
  "creation date descending", "net pay" for "take-home salary").
- The response paraphrases, summarises, or omits details without asserting
  something false.
- The response performs a correct mathematical restatement (e.g. "1200mg/day"
  for "3 × 400mg tablets").
- NOTE: different specific numbers are never equivalent — a different CVSS score,
  port number, version range, duration, or quota is always a real contradiction,
  not a paraphrase.
- Your own reasoning contains any hedge such as "generally", "typically",
  "could be interpreted", or "though they refer to the same" — if you hedge,
  set is_paraphrase_or_equivalent=True and do not add the finding.

For each finding you do add, reason step by step in the `reasoning` field before
filling in the spans, so you can catch your own mistakes.
"""

_USER_TEMPLATE = """\
CONTEXT:
{context}

RESPONSE:
{response}

UNCERTAIN NLI PAIRS (pairs the local model flagged but was not confident about):
{uncertain_pairs}

Analyse the full context and response, then list every contradiction you find.
"""


class _ContradictionItem(BaseModel):
    """Single contradiction finding returned by the model."""

    reasoning: str                    # chain-of-thought before committing to the finding
    is_paraphrase_or_equivalent: bool # True if the response is just restating/paraphrasing
    response_span: str                # exact phrase in the response that contradicts
    context_span: str                 # exact phrase in the context being contradicted
    explanation: str                  # plain-English explanation of the contradiction
    severity: Severity
    confidence: float                 # 0.0–1.0


class _JudgeResponse(BaseModel):
    """Top-level structured output from GPT-4o."""

    overall_reasoning: str              # high-level reasoning before listing findings
    contradictions: list[_ContradictionItem]


def _log_judge_result(
    log: logging.Logger,
    overall_reasoning: str,
    raw_count: int,
    genuine_count: int,
) -> None:
    """Log GPT-4o output at INFO; move per-finding filter detail to DEBUG."""
    log.info(
        "GPT-4o returned %d finding(s), %d genuine after filtering",
        raw_count,
        genuine_count,
    )
    dropped = raw_count - genuine_count
    if dropped:
        log.debug("Dropped %d finding(s) (paraphrase or low-confidence)", dropped)
    log.debug("overall_reasoning length=%d chars", len(overall_reasoning))


def _format_uncertain_pairs(pairs: list[NLIResult]) -> str:
    """Render uncertain NLI pairs as a readable numbered list."""
    if not pairs:
        return "(none — judge the full document independently)"
    lines = []
    for i, nli_result in enumerate(pairs, 1):
        lines.append(
            f"{i}. PREMISE: {nli_result.pair.premise!r}\n"
            f"   HYPOTHESIS: {nli_result.pair.hypothesis!r}\n"
            f"   NLI score: {nli_result.contradiction_score:.2f} (uncertain)"
        )
    return "\n".join(lines)


class LLMJudge:
    """Calls GPT-4o to evaluate contradictions with chain-of-thought reasoning.

    Structured outputs guarantee that every response matches the _JudgeResponse
    schema — OpenAI enforces this at the token-sampling level, so no JSON
    parsing errors are possible.
    """

    def __init__(self) -> None:
        """Instantiate the OpenAI client using the key from settings."""
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        logger.info("LLMJudge initialised (model=%s)", _GPT_MODEL)

    def judge(
        self,
        context: str,
        response: str,
        uncertain_pairs: list[NLIResult],
    ) -> list[Contradiction]:
        """Run GPT-4o over the full context/response and return contradictions.

        Args:
            context: Source document the response should be faithful to.
            response: LLM-generated text under evaluation.
            uncertain_pairs: NLIResult objects with confidence below the routing
                threshold — surfaced to the model as hints, not constraints.

        Returns:
            List of Contradiction objects with method=DetectionMethod.LLM.
        """
        user_message = _USER_TEMPLATE.format(
            context=context,
            response=response,
            uncertain_pairs=_format_uncertain_pairs(uncertain_pairs),
        )

        logger.info(
            "Calling %s with %d uncertain pairs", _GPT_MODEL, len(uncertain_pairs)
        )

        completion = self._client.beta.chat.completions.parse(
            model=_GPT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_JudgeResponse,
        )

        judge_response = completion.choices[0].message.parsed
        if judge_response is None:
            logger.error("GPT-4o returned a null parsed response — structured output failed")
            return []

        genuine_contradictions = [
            item for item in judge_response.contradictions
            if not item.is_paraphrase_or_equivalent and item.confidence >= _LLM_MIN_CONFIDENCE
        ]
        _log_judge_result(logger, judge_response.overall_reasoning, len(judge_response.contradictions), len(genuine_contradictions))

        return [
            Contradiction(
                response_span=item.response_span,
                context_span=item.context_span,
                explanation=item.explanation,
                severity=item.severity,
                method=DetectionMethod.LLM,
                confidence=item.confidence,
            )
            for item in genuine_contradictions
        ]
