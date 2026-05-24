"""Shared Pydantic models used across all parts of the contradiction detector."""

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    BLOCKING = "blocking"    # agent action directly violates a single policy rule
    WARNING  = "warning"     # violation requires mild inference or is partially restricted
    INFERRED = "inferred"    # violation emerges by combining two or more policy rules


class DetectionMethod(str, Enum):
    NLI = "nli"
    LLM = "llm"
    ENSEMBLE = "ensemble"


class FeedbackVerdict(str, Enum):
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"


class SentencePair(BaseModel):
    premise: str             # sentence from context
    hypothesis: str          # sentence from response


class NLIResult(BaseModel):
    pair: SentencePair
    label: str               # "entailment" | "neutral" | "contradiction"
    confidence: float        # softmax probability of the winning label
    contradiction_score: float  # raw softmax score for the NLI contradiction class
    entailment_score: float = 0.0
    neutral_score: float = 0.0


class Violation(BaseModel):
    response_span: str       # exact phrase in the response that violates policy
    context_span: str        # exact phrase in the context being violated
    explanation: str         # plain English explanation
    severity: Severity
    method: DetectionMethod  # which method caught this
    confidence: float        # 0-1, how confident we are this is a real violation


class ViolationReport(BaseModel):
    run_id: str | None = None          # populated when MongoDB persistence is enabled
    compliance_score: float            # 0-1, higher = more compliant
    violations: list[Violation]
    method_used: DetectionMethod
    nli_pairs_checked: int
    nli_candidates: int = 0            # high-confidence NLI candidates passed to the LLM
    llm_escalations: int               # sentence pairs escalated to the LLM judge
    processing_time_ms: float
    overall_reasoning: str | None = None   # LLM chain-of-thought
    input_tokens: int = 0              # exact input token count from API
    output_tokens: int = 0             # exact output token count from API
    cost_usd: float = 0.0              # (input_tokens × price_in) + (output_tokens × price_out)


class CheckRequest(BaseModel):
    context: str = Field(..., max_length=50_000)
    response: str = Field(..., max_length=50_000)


class FeedbackRequest(BaseModel):
    violation_index: int
    verdict: FeedbackVerdict


class HistoryItem(BaseModel):
    run_id: str
    timestamp: str                     # ISO 8601
    compliance_score: float
    violation_count: int
    method_used: str
    provider: str
    context_snippet: str               # first 100 chars of context


class HistoryDetail(BaseModel):
    run_id: str
    timestamp: str                     # ISO 8601
    compliance_score: float
    method_used: str
    provider: str
    context: str
    response: str
    violations: list[Violation]


class StatsResponse(BaseModel):
    total_runs: int
    total_violations: int
    confirmed_rate: float              # fraction of feedback marked "confirmed"


class BenchmarkResult(BaseModel):
    method: DetectionMethod
    precision: float
    recall: float
    f1: float
    f1_ci_low: float               # bootstrap 95% CI lower bound
    f1_ci_high: float              # bootstrap 95% CI upper bound
    fpr: float                     # false positive rate: FP / (FP + TN)
    auc_roc: float                 # threshold-independent discrimination score
    per_difficulty: dict[str, dict[str, float]]  # {easy|medium|hard: {precision, recall, f1}}
    avg_latency_ms: float
    estimated_cost_per_call: float
