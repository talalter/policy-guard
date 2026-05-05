"""Benchmark runner: evaluates NLI-only, LLM-only, and ensemble detection methods.

Loads labeled examples from data/examples.json, runs all three methods on each,
computes Precision / Recall / F1, measures wall-clock latency, estimates
GPT-4o cost, and writes results to data/benchmark_results.json.

Prediction rule:
    An example is considered a *positive* prediction if the pipeline returns
    at least one Contradiction object (len(contradictions) > 0).

Cost model (OpenAI pricing, early 2026):
    Input:  $2.50 / 1M tokens
    Output: $10.00 / 1M tokens
    Token approximation: len(text) // 4  (4 chars ≈ 1 token, OpenAI convention)
    NLI runs locally — always $0.00.

Usage:
    python -m backend.tools.benchmark
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings  # noqa: E402
from backend.core import LLMJudge, NLIScorer, Router  # noqa: E402
from backend.models import BenchmarkResult, DetectionMethod  # noqa: E402

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_EXAMPLES_PATH = _DATA_DIR / "ragtruth_sample.json"
_RESULTS_PATH = _DATA_DIR / "benchmark_results_ragtruth_sample_llm_only.json"

# OpenAI pricing as of early 2026: $2.50/1M input tokens, $10.00/1M output tokens
_INPUT_COST_PER_TOKEN: float = 2.50 / 1_000_000
_OUTPUT_COST_PER_TOKEN: float = 10.00 / 1_000_000

# Approximate fixed overhead: system prompt + template structure
_SYSTEM_PROMPT_TOKENS = 120
_OUTPUT_TOKENS_ESTIMATE = 300


@dataclass
class _MethodRun:
    """Accumulated per-example results for one detection method."""

    predictions: list[bool]
    scores: list[float]      # continuous confidence score per example, used for AUC-ROC
    latencies_ms: list[float]
    costs: list[float]


def _load_examples(path: Path) -> list[dict]:
    """Load labeled examples from the JSON file at path."""
    with open(path) as f:
        examples = json.load(f)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def _count_tokens(text: str) -> int:
    """Approximate token count using the 4-chars-per-token convention."""
    return max(1, len(text) // 4)


def _estimate_llm_cost(context: str, response: str) -> float:
    """Estimate GPT-4o cost for a single context+response call."""
    input_tokens = (
        _SYSTEM_PROMPT_TOKENS + _count_tokens(context) + _count_tokens(response)
    )
    return (
        input_tokens * _INPUT_COST_PER_TOKEN
        + _OUTPUT_TOKENS_ESTIMATE * _OUTPUT_COST_PER_TOKEN
    )


def _compute_metrics(
    ground_truth: list[bool],
    predictions: list[bool],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for binary contradiction detection.

    Returns (precision, recall, f1) rounded to 4 decimal places.
    Undefined metrics (zero denominator) are returned as 0.0.
    """
    tp = sum(g and p for g, p in zip(ground_truth, predictions))
    fp = sum((not g) and p for g, p in zip(ground_truth, predictions))
    fn = sum(g and (not p) for g, p in zip(ground_truth, predictions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _compute_fpr(ground_truth: list[bool], predictions: list[bool]) -> float:
    """Compute False Positive Rate: FP / (FP + TN).

    Answers: of all faithful responses, what fraction did we wrongly flag?
    Directly maps to alert fatigue — the security practitioner's primary concern.
    """
    fp = sum((not g) and p for g, p in zip(ground_truth, predictions))
    tn = sum((not g) and (not p) for g, p in zip(ground_truth, predictions))
    return round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0


def _compute_auc_roc(ground_truth: list[bool], scores: list[float]) -> float:
    """Compute AUC-ROC across all confidence thresholds.

    Unlike F1 at a fixed threshold, AUC-ROC measures intrinsic discriminative
    power — how well the model separates positives from negatives regardless of
    where the decision boundary is set.
    """
    if len(set(ground_truth)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(ground_truth, scores)), 4)


def _compute_bootstrap_ci(
    ground_truth: list[bool],
    predictions: list[bool],
    n_iter: int = 1000,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for F1.

    Resamples the existing predictions with replacement to quantify uncertainty
    without additional model calls. Wide intervals signal that more test data
    is needed before drawing strong conclusions.
    """
    import random
    pairs = list(zip(ground_truth, predictions))
    n = len(pairs)
    f1_scores = []
    for _ in range(n_iter):
        sample = random.choices(pairs, k=n)
        _, _, f1 = _compute_metrics([g for g, _ in sample], [p for _, p in sample])
        f1_scores.append(f1)
    f1_scores.sort()
    low = int(0.025 * n_iter)
    high = int(0.975 * n_iter)
    return round(f1_scores[low], 4), round(f1_scores[high], 4)


def _compute_per_severity(
    examples: list[dict],
    predictions: list[bool],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 broken down by contradiction_type.

    Reveals where each method struggles — e.g. NLI handles direct contradictions
    well but misses multi-hop ones that require full-document reasoning.
    """
    from collections import defaultdict
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for ex, pred in zip(examples, predictions):
        severity = ex.get("contradiction_type") or "none"
        groups[severity].append((ex["has_contradiction"], pred))

    result = {}
    for severity, pairs in groups.items():
        if severity == "none":
            continue
        gt = [g for g, _ in pairs]
        preds = [p for _, p in pairs]
        precision, recall, f1 = _compute_metrics(gt, preds)
        result[severity] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def _run_nli_only(
    examples: list[dict],
    scorer: NLIScorer,
) -> _MethodRun:
    """Run NLI-only detection on every example; NLI escalation cost is always $0."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in examples:
        t_start = time.perf_counter()
        # Materialize once so we can safely compute multiple metrics.
        results = list(scorer.score(example["context"], example["response"]))
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        predictions.append(any(r.label == "contradiction" for r in results))
        scores.append(max((r.contradiction_score for r in results), default=0.0))
        costs.append(0.0)
    return _MethodRun(predictions=predictions, scores=scores, latencies_ms=latencies_ms, costs=costs)


def _run_llm_only(
    examples: list[dict],
    judge: LLMJudge,
) -> _MethodRun:
    """Run LLM-only detection (no NLI pre-filter) on every example."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in examples:
        t_start = time.perf_counter()
        contradictions = judge.judge(
            context=example["context"],
            response=example["response"],
            uncertain_pairs=[],
        )
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        predictions.append(len(contradictions) > 0)
        scores.append(max((c.confidence for c in contradictions), default=0.0))
        costs.append(_estimate_llm_cost(example["context"], example["response"]))
    return _MethodRun(predictions=predictions, scores=scores, latencies_ms=latencies_ms, costs=costs)


def _run_ensemble(
    examples: list[dict],
    router: Router,
) -> _MethodRun:
    """Run ensemble detection (NLI + conditional LLM escalation) on every example."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in examples:
        t_start = time.perf_counter()
        contradictions, metadata = router.route(example["context"], example["response"])
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        predictions.append(len(contradictions) > 0)
        scores.append(max((c.confidence for c in contradictions), default=0.0))
        # Cost is $0 when NLI resolved it without LLM escalation.
        if metadata.get("llm_escalated", 0) > 0:
            costs.append(_estimate_llm_cost(example["context"], example["response"]))
        else:
            costs.append(0.0)
    return _MethodRun(predictions=predictions, scores=scores, latencies_ms=latencies_ms, costs=costs)


def _build_result(
    method: DetectionMethod,
    run: _MethodRun,
    ground_truth: list[bool],
    examples: list[dict],
) -> BenchmarkResult:
    """Assemble a BenchmarkResult from accumulated run data and ground truth."""
    precision, recall, f1 = _compute_metrics(ground_truth, run.predictions)
    f1_ci_low, f1_ci_high = _compute_bootstrap_ci(ground_truth, run.predictions)
    fpr = _compute_fpr(ground_truth, run.predictions)
    auc_roc = _compute_auc_roc(ground_truth, run.scores)
    per_severity = _compute_per_severity(examples, run.predictions)
    avg_latency = sum(run.latencies_ms) / len(run.latencies_ms)
    avg_cost = sum(run.costs) / len(run.costs)
    return BenchmarkResult(
        method=method,
        precision=precision,
        recall=recall,
        f1=f1,
        f1_ci_low=f1_ci_low,
        f1_ci_high=f1_ci_high,
        fpr=fpr,
        auc_roc=auc_roc,
        per_severity=per_severity,
        avg_latency_ms=round(avg_latency, 1),
        estimated_cost_per_call=round(avg_cost, 6),
    )


def _print_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results: main metrics table + per-severity breakdown."""
    labels = {
        DetectionMethod.NLI: "NLI only",
        DetectionMethod.LLM: "GPT-4o only",
        DetectionMethod.ENSEMBLE: "Ensemble",
    }

    # Main metrics table
    col = (14, 10, 8, 8, 14, 6, 9, 13, 16)
    header = (
        f"{'Method':<{col[0]}}{'Precision':>{col[1]}}{'Recall':>{col[2]}}"
        f"{'F1':>{col[3]}}{'Avg Latency':>{col[4]}}{'FPR':>{col[5]}}"
        f"{'AUC-ROC':>{col[6]}}{'F1 95% CI':>{col[7]}}{'Est. Cost/call':>{col[8]}}"
    )
    separator = "  ".join("-" * w for w in col)
    print(header)
    print(separator)
    for r in results:
        name = labels.get(r.method, r.method.value)
        ci = f"[{r.f1_ci_low:.2f},{r.f1_ci_high:.2f}]"
        print(
            f"{name:<{col[0]}}"
            f"{r.precision:>{col[1]}.2f}"
            f"{r.recall:>{col[2]}.2f}"
            f"{r.f1:>{col[3]}.2f}"
            f"{r.avg_latency_ms / 1000:>{col[4] - 1}.1f}s"
            f"{r.fpr:>{col[5]}.2f}"
            f"{r.auc_roc:>{col[6]}.2f}"
            f"{ci:>{col[7]}}"
            f"  ${r.estimated_cost_per_call:>{col[8] - 3}.4f}"
        )

    # Per-severity breakdown
    severity_order = ["direct", "partial", "multi-hop"]
    all_severities = sorted(
        {s for r in results for s in r.per_severity},
        key=lambda s: severity_order.index(s) if s in severity_order else 99,
    )
    if not all_severities:
        return

    print("\nPer-severity F1:")
    sev_col = (14, 10, 10, 10)
    sev_header = (
        f"{'Method':<{sev_col[0]}}"
        + "".join(f"{s:>{sev_col[1]}}" for s in all_severities)
    )
    print(sev_header)
    print("-" * (sev_col[0] + sev_col[1] * len(all_severities)))
    for r in results:
        name = labels.get(r.method, r.method.value)
        row = f"{name:<{sev_col[0]}}"
        for s in all_severities:
            f1 = r.per_severity.get(s, {}).get("f1", float("nan"))
            row += f"{f1:>{sev_col[1]}.2f}" if not (f1 != f1) else f"{'—':>{sev_col[1]}}"
        print(row)


def _save_results(results: list[BenchmarkResult], path: Path) -> None:
    """Serialize benchmark results to JSON at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    logger.info("Results saved to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contradiction detection benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_EXAMPLES_PATH,
        help="Path to a labeled examples JSON file (default: data/examples.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write results JSON (default: data/benchmark_results.json or "
             "data/benchmark_results_<stem>.json for non-default datasets)",
    )
    return parser.parse_args()


def main() -> None:
    """Load examples, run all three methods, print the comparison table, and save."""
    args = _parse_args()
    dataset_path = args.dataset

    results_path = args.output
    if results_path is None:
        if dataset_path == _EXAMPLES_PATH:
            results_path = _RESULTS_PATH
        else:
            results_path = _DATA_DIR / f"benchmark_results_{dataset_path.stem}.json"

    print("Loading models (this may take a moment on first run)...")
    router = Router()
    scorer = router.get_scorer()  # reuse; avoids loading weights twice
    judge = router.get_judge()

    examples = _load_examples(dataset_path)
    ground_truth = [ex["has_contradiction"] for ex in examples]
    print(f"Running benchmark on {len(examples)} examples from {dataset_path.name}...\n")

    print("  [1/3] NLI only  (local, free)...")
    nli_run = _run_nli_only(examples, scorer)
    print("  [2/3] LLM only  (GPT-4o, all examples)...")
    llm_run = _run_llm_only(examples, judge)

    print("  [3/3] Ensemble  (NLI + conditional GPT-4o)...")
    ensemble_run = _run_ensemble(examples, router)

    results = [
        _build_result(DetectionMethod.NLI, nli_run, ground_truth, examples),  # type: ignore
        _build_result(DetectionMethod.LLM, llm_run, ground_truth, examples),  # type: ignore
        _build_result(DetectionMethod.ENSEMBLE, ensemble_run, ground_truth, examples),  # type: ignore
    ]

    print()
    _save_results(results, results_path)
    _print_table(results)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
