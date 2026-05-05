"""Manual test script for the Aggregator.

Runs the full pipeline (Router → Aggregator) on every example in
data/examples.json and prints the final ContradictionReport for each.
This is the closest thing to an end-to-end test before the FastAPI layer exists.

Run from the backend/ directory:
    python test_aggregator.py
"""

import json
import logging
import pathlib
import time

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from backend.core import Aggregator, Router

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_EXAMPLES_PATH = pathlib.Path(__file__).parent.parent / "data" / "examples.json"


def load_examples() -> list[dict]:
    """Load labeled examples from the shared data file."""
    with _EXAMPLES_PATH.open() as f:
        return json.load(f)


def run_case(router: Router, agg: Aggregator, example: dict) -> bool:
    """Run one example through the full pipeline and print the report. Returns True if PASS."""
    print(f"\n{'='*60}")
    print(f"CASE [{example['id']}]: {example['contradiction_type'].upper()} — {example.get('notes', '')}")
    print(f"{'='*60}")
    print(f"Context:  {example['context'][:120]}{'...' if len(example['context']) > 120 else ''}")
    print(f"Response: {example['response'][:120]}{'...' if len(example['response']) > 120 else ''}")
    print(f"Expected contradiction: {example['has_contradiction']}")
    print()

    t0 = time.perf_counter()
    contradictions, meta = router.route(
        context=example["context"],
        response=example["response"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    report = agg.aggregate(contradictions, meta, elapsed_ms)

    print(f"Faithfulness score : {report.faithfulness_score:.4f}")
    print(f"Method used        : {report.method_used.value}")
    print(f"NLI pairs checked  : {report.nli_pairs_checked}")
    print(f"LLM escalations    : {report.llm_escalations}")
    print(f"Processing time    : {report.processing_time_ms:.1f}ms")
    print(f"Contradictions     : {len(report.contradictions)}")

    for c in report.contradictions:
        print(f"  [{c.method.value:8}] [{c.severity.value:9}] conf={c.confidence:.2f}")
        print(f"    Response span: {c.response_span!r}")
        print(f"    Context span:  {c.context_span!r}")
        print(f"    Explanation:   {c.explanation}")

    passed = bool(report.contradictions) == example["has_contradiction"]
    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    """Load the pipeline once, then run all examples from data/examples.json."""
    examples = load_examples()
    print(f"Loaded {len(examples)} examples from {_EXAMPLES_PATH}")
    print("Initialising router and aggregator...")
    router = Router()
    agg = Aggregator()
    print("Ready.\n")

    results = [run_case(router, agg, ex) for ex in examples]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Done. {passed}/{total} passed.")


if __name__ == "__main__":
    main()
