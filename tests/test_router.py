"""Manual test script for the confidence-based router.

Loads examples from data/examples.json and runs each through the full
NLI → routing → LLM-escalation pipeline.  Shows which contradictions
were caught by NLI vs escalated to GPT-4o and prints routing metadata.

Run from the backend/ directory:
    python test_router.py

To add examples, edit data/examples.json — do not add them here.
"""

import json
import logging
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from backend.core import Router

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_EXAMPLES_PATH = pathlib.Path(__file__).parent.parent / "data" / "examples.json"


def load_examples() -> list[dict]:
    """Load labeled examples from the shared data file."""
    with _EXAMPLES_PATH.open() as f:
        return json.load(f)


def run_case(router: Router, example: dict) -> bool:
    """Run a single example through the router and print results. Returns True if PASS."""
    print(f"\n{'='*60}")
    print(f"CASE [{example['id']}]: {example['contradiction_type'].upper()} — {example.get('notes', '')}")
    print(f"{'='*60}")
    print(f"Context:  {example['context'][:120]}{'...' if len(example['context']) > 120 else ''}")
    print(f"Response: {example['response'][:120]}{'...' if len(example['response']) > 120 else ''}")
    print(f"Expected contradiction: {example['has_contradiction']}")
    print()

    contradictions, meta = router.route(
        context=example["context"],
        response=example["response"],
    )

    print(f"Routing metadata:")
    print(f"  NLI pairs checked : {meta['nli_pairs_checked']}")
    print(f"  NLI caught        : {meta['nli_caught']}")
    print(f"  Escalated to LLM  : {meta['llm_escalated']}")
    print(f"  LLM caught        : {meta['llm_caught']}")
    print(f"  After dedup       : {meta['after_dedup']}")
    print()

    print(f"Contradictions found: {len(contradictions)}")
    for c in contradictions:
        print(f"  [{c.method.value:8}] [{c.severity.value:9}] conf={c.confidence:.2f}")
        print(f"    Response span: {c.response_span!r}")
        print(f"    Context span:  {c.context_span!r}")
        print(f"    Explanation:   {c.explanation}")

    passed = bool(contradictions) == example["has_contradiction"]
    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    """Load the router once, then run all examples from data/examples.json."""
    examples = load_examples()
    print(f"Loaded {len(examples)} examples from {_EXAMPLES_PATH}")
    print("Initialising router (loads NLI model + OpenAI client)...")
    router = Router()
    print("Ready.\n")

    results = [run_case(router, ex) for ex in examples]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Done. {passed}/{total} passed.")


if __name__ == "__main__":
    main()
