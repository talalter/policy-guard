"""Benchmark runner: evaluates NLI-only, LLM-only, and ensemble detection methods.

Loads labeled examples from data/examples.json, runs all three methods on each,
computes Precision / Recall / F1, measures wall-clock latency, estimates
GPT-5.4-mini cost, and writes results to data/benchmark_results.json.

Prediction rule:
    An example is considered a *positive* prediction if the pipeline returns
    at least one Violation object (len(violations) > 0).

Cost model (gpt-5.4-mini standard pricing, 2026):
    Token counts: read from resp.usage after every API call.
    Multi-turn cost: summed across all tool-loop iterations (each request charges for
    the full growing conversation, so simple summing gives the true billed amount).
    NLI runs locally - always $0.00.

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

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings  # noqa: E402
from backend.core import BaseLLMJudge, NLIScorer, Router  # noqa: E402
from backend.core.router import nli_to_violation  # noqa: E402
from backend.models import BenchmarkResult, DetectionMethod  # noqa: E402
from backend.utils.dedup import deduplicate  # noqa: E402

class _TqdmHandler(logging.StreamHandler):
    """Routes log records through tqdm.write() so they don't break progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.root.setLevel(settings.log_level.upper())
logging.root.handlers = [_handler]
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_EXAMPLES_PATH = _DATA_DIR / "agent_action_policy_benchmark_v1.json"
_RESULTS_PATH = _DATA_DIR / "benchmark_results_agent_action_policy_benchmark_v1_1.json"

# gpt-5.4-mini standard pricing: $0.75/1M input tokens, $4.50/1M output tokens
# (Batch API is half this; benchmark uses real-time calls so standard rates apply.)
_INPUT_COST_PER_TOKEN: float = 0.75 / 1_000_000
_OUTPUT_COST_PER_TOKEN: float = 4.50 / 1_000_000


@dataclass
class _MethodRun:
    """Accumulated per-example results for one detection method."""

    predictions: list[bool]
    scores: list[float]      # continuous confidence score per example, used for AUC-ROC
    latencies_ms: list[float]
    costs: list[float]


_POLICY_LABEL_TO_BOOL: dict[str, bool] = {"FAIL": True, "PARTIAL": True, "PASS": False}


def _flatten_policy_benchmark(data: dict) -> tuple[list[dict], int]:
    """Flatten the nested policy-benchmark format into a flat list of examples.

    Label mapping:
        FAIL / PARTIAL  → has_violation=True
        PASS            → has_violation=False
        UNCERTAIN       → excluded (ground truth genuinely unknown)

    Returns (examples, uncertain_count).
    """
    flat: list[dict] = []
    uncertain = 0
    for policy in data["policies"]:
        for ex in policy["examples"]:
            if ex["label"] == "UNCERTAIN":
                uncertain += 1
                continue
            flat.append({
                "context": policy["policy_text"],
                "response": ex["response"],
                "has_violation": _POLICY_LABEL_TO_BOOL[ex["label"]],
                "contradiction_type": ex.get("primary_reasoning_type", "unknown"),
                "difficulty": ex.get("difficulty", "unknown"),
                "label": ex["label"],
                "policy_id": policy["policy_id"],
                "example_id": ex["example_id"],
            })
    return flat, uncertain


def _load_examples(path: Path) -> list[dict]:
    """Load labeled examples from JSON; auto-detects the nested policy benchmark format."""
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "policies" in raw:
        examples, uncertain_count = _flatten_policy_benchmark(raw)
        logger.info(
            "Loaded %d examples from %s (policy benchmark; %d UNCERTAIN excluded)",
            len(examples), path, uncertain_count,
        )
    else:
        examples = raw
        logger.info("Loaded %d examples from %s", len(examples), path)

    return examples # type: ignore


def _count_tokens(text: str) -> int:
    """Approximate token count using the 4-chars-per-token convention."""
    return max(1, len(text) // 4)


def _actual_llm_cost(judge: "BaseLLMJudge") -> float:
    """Compute exact cost from token counts returned by the provider API."""
    usage = judge.get_last_usage()
    return (
        usage["input_tokens"] * _INPUT_COST_PER_TOKEN
        + usage["output_tokens"] * _OUTPUT_COST_PER_TOKEN
    )


def _compute_metrics(
    ground_truth: list[bool],
    predictions: list[bool],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for binary violation detection.

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
    Directly maps to alert fatigue - the security practitioner's primary concern.
    """
    fp = sum((not g) and p for g, p in zip(ground_truth, predictions))
    tn = sum((not g) and (not p) for g, p in zip(ground_truth, predictions))
    return round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0


def _compute_auc_roc(ground_truth: list[bool], scores: list[float]) -> float:
    """Compute AUC-ROC across all confidence thresholds.

    Unlike F1 at a fixed threshold, AUC-ROC measures intrinsic discriminative
    power - how well the model separates positives from negatives regardless of
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


def _compute_per_group(
    examples: list[dict],
    predictions: list[bool],
    key: str,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 broken down by an arbitrary example field.

    Used for both contradiction_type (legacy) and primary_reasoning_type / difficulty
    (policy benchmark format).
    """
    from collections import defaultdict
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for ex, pred in zip(examples, predictions):
        group = ex.get(key) or "none"
        groups[group].append((ex["has_violation"], pred))

    result = {}
    for group, pairs in groups.items():
        if group == "none":
            continue
        gt = [g for g, _ in pairs]
        preds = [p for _, p in pairs]
        precision, recall, f1 = _compute_metrics(gt, preds)
        result[group] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def _compute_per_difficulty(
    examples: list[dict],
    predictions: list[bool],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 broken down by difficulty (easy / medium / hard)."""
    return _compute_per_group(examples, predictions, "difficulty")


def _run_nli_only(
    examples: list[dict],
    scorer: NLIScorer,
) -> _MethodRun:
    """Run NLI-only detection on every example; NLI escalation cost is always $0."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in tqdm(examples, desc="NLI only", unit="ex"):
        t_start = time.perf_counter()
        results = list(scorer.score(example["context"], example["response"]))
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        # Apply the same confidence gate and deduplication the router uses so the
        # NLI-only metric is computed on the same basis as the ensemble path.
        violations = deduplicate([
            nli_to_violation(r)
            for r in results
            if r.label == "contradiction" and r.confidence >= settings.nli_confidence_threshold
        ])
        predictions.append(len(violations) > 0)
        scores.append(max((v.confidence for v in violations), default=0.0))
        costs.append(0.0)
    return _MethodRun(predictions=predictions, scores=scores, latencies_ms=latencies_ms, costs=costs)


def _run_llm_only(
    examples: list[dict],
    judge: BaseLLMJudge,
) -> _MethodRun:
    """Run LLM-only detection (no NLI pre-filter) on every example."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in tqdm(examples, desc="LLM only", unit="ex"):
        t_start = time.perf_counter()
        violations = judge.judge(
            context=example["context"],
            response=example["response"],
            candidate_pairs=[],
            uncertain_pairs=[],
        )
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        predictions.append(len(violations) > 0)
        scores.append(max((v.confidence for v in violations), default=0.0))
        costs.append(_actual_llm_cost(judge))
    return _MethodRun(predictions=predictions, scores=scores, latencies_ms=latencies_ms, costs=costs)


def _run_ensemble(
    examples: list[dict],
    router: Router,
    judge: BaseLLMJudge,
) -> _MethodRun:
    """Run ensemble detection (NLI + conditional LLM escalation) on every example."""
    predictions, scores, latencies_ms, costs = [], [], [], []
    for example in tqdm(examples, desc="Ensemble", unit="ex"):
        t_start = time.perf_counter()
        violations, metadata = router.route(example["context"], example["response"])
        latencies_ms.append((time.perf_counter() - t_start) * 1000)
        predictions.append(len(violations) > 0)
        scores.append(max((v.confidence for v in violations), default=0.0))
        # Cost is $0 when NLI resolved it without LLM escalation.
        if metadata.get("llm_escalated", 0) > 0:
            costs.append(_actual_llm_cost(judge))
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
    per_difficulty = _compute_per_difficulty(examples, run.predictions)
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
        per_difficulty=per_difficulty,
        avg_latency_ms=round(avg_latency, 1),
        estimated_cost_per_call=round(avg_cost, 6),
    )


def _print_table(results: list[BenchmarkResult], examples: list[dict]) -> None:
    """Print benchmark results: main metrics table + per-severity breakdown."""
    labels = {
        DetectionMethod.NLI: "NLI only",
        DetectionMethod.LLM: f"{settings.gpt_model} only",
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

    # Per-difficulty breakdown
    difficulty_order = ["easy", "medium", "hard"]
    all_difficulties = [d for d in difficulty_order if any(d in r.per_difficulty for r in results)]
    if all_difficulties:
        from collections import Counter
        diff_counts = Counter(ex.get("difficulty", "unknown") for ex in examples)
        print("\nPer-difficulty F1:")
        col_w = 10
        diff_header = f"{'Method':<14}" + "".join(f"{d:>{col_w}}" for d in all_difficulties)
        note = "  (" + " · ".join(f"n={diff_counts.get(d, 0)} {d}" for d in all_difficulties) + ")"
        print(diff_header + note)
        print("-" * (14 + col_w * len(all_difficulties)))
        for r in results:
            name = labels.get(r.method, r.method.value)
            row = f"{name:<14}"
            for d in all_difficulties:
                f1 = r.per_difficulty.get(d, {}).get("f1", float("nan"))
                row += f"{f1:>{col_w}.2f}" if not (f1 != f1) else f"{'N/A':>{col_w}}"
            print(row)



def _save_results(results: list[BenchmarkResult], path: Path) -> None:
    """Serialize benchmark results to JSON at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    logger.info("Results saved to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy violation detection benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_EXAMPLES_PATH,
        help="Path to a labeled examples JSON file (default: data/examples.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_RESULTS_PATH,
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
    ground_truth = [ex["has_violation"] for ex in examples]
    print(f"Running benchmark on {len(examples)} examples from {dataset_path.name}...\n")

    nli_run = _run_nli_only(examples, scorer)
    llm_run = _run_llm_only(examples, judge)
    ensemble_run = _run_ensemble(examples, router, judge)

    method_runs = [
        (DetectionMethod.NLI, nli_run),
        (DetectionMethod.LLM, llm_run),
        (DetectionMethod.ENSEMBLE, ensemble_run),
    ]
    results = [
        _build_result(method, run, ground_truth, examples)
        for method, run in method_runs
    ]

    print()
    _save_results(results, results_path)
    _print_table(results, examples)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
