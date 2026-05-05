"""Shared Pydantic models used across all parts of the contradiction detector."""

from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    DIRECT = "direct"        # premise directly negates hypothesis
    PARTIAL = "partial"      # premise partially contradicts hypothesis
    MULTIHOP = "multi-hop"   # requires combining multiple premises


class DetectionMethod(str, Enum):
    NLI = "nli"
    LLM = "llm"
    ENSEMBLE = "ensemble"


class SentencePair(BaseModel):
    premise: str             # sentence from context
    hypothesis: str          # sentence from response
    pair_index: int          # position in the cross-product matrix


class NLIResult(BaseModel):
    pair: SentencePair
    label: str               # "entailment" | "neutral" | "contradiction"
    confidence: float        # softmax probability of the winning label
    contradiction_score: float  # raw softmax score for the contradiction class


class Contradiction(BaseModel):
    response_span: str       # exact phrase in the response that contradicts
    context_span: str        # exact phrase in the context being contradicted
    explanation: str         # plain English explanation
    severity: Severity
    method: DetectionMethod  # which method caught this
    confidence: float        # 0-1, how confident we are this is a real contradiction


class ContradictionReport(BaseModel):
    faithfulness_score: float          # 0-1, higher = more faithful
    contradictions: list[Contradiction]
    method_used: DetectionMethod
    nli_pairs_checked: int
    llm_escalations: int               # how many pairs were sent to GPT-4o
    processing_time_ms: float


class CheckRequest(BaseModel):
    context: str
    response: str


class BenchmarkResult(BaseModel):
    method: DetectionMethod
    precision: float
    recall: float
    f1: float
    f1_ci_low: float               # bootstrap 95% CI lower bound
    f1_ci_high: float              # bootstrap 95% CI upper bound
    fpr: float                     # false positive rate: FP / (FP + TN)
    auc_roc: float                 # threshold-independent discrimination score
    per_severity: dict[str, dict[str, float]]  # {type: {precision, recall, f1}}
    avg_latency_ms: float
    estimated_cost_per_call: float
