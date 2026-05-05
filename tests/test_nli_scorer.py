"""Manual test script for NLI scorer.

Loads examples from data/examples.json and runs each through the NLI scorer.

Run from the backend/ directory:
    python test_nli_scorer.py

To add examples, edit data/examples.json — do not add them here.
"""

import json
import logging
import pathlib

from backend.core import NLIScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_EXAMPLES_PATH = pathlib.Path(__file__).parent.parent / "data" / "examples.json"


def load_examples() -> list[dict]:
    """Load labeled examples from the shared data file."""
    with _EXAMPLES_PATH.open() as f:
        return json.load(f)


def run_case(scorer: NLIScorer, example: dict) -> bool:
    """Run a single example and print results. Returns True if PASS."""
    print(f"\n{'='*60}")
    print(f"CASE [{example['id']}]: {example['contradiction_type'].upper()} — {example.get('notes', '')}")
    print(f"{'='*60}")
    print(f"Context:  {example['context'][:120]}{'...' if len(example['context']) > 120 else ''}")
    print(f"Response: {example['response'][:120]}{'...' if len(example['response']) > 120 else ''}")
    print(f"Expected contradiction: {example['has_contradiction']}")
    print()

    results = []
    print("Streaming results (highest similarity first):")
    for r in scorer.score(example["context"], example["response"]):
        results.append(r)
        marker = ">>>" if r.label == "contradiction" else "   "
        print(f"  {marker} [{r.label:15}] conf={r.confidence:.2f}  contra={r.contradiction_score:.2f}")
        print(f"        P: {r.pair.premise}")
        print(f"        H: {r.pair.hypothesis}")

    contradictions = [r for r in results if r.label == "contradiction"]
    print()
    print(f"Total pairs scored: {len(results)}")
    print(f"Contradictions found: {len(contradictions)}")
    for c in contradictions:
        print(f"  - conf={c.confidence:.2f} | {c.pair.premise!r} vs {c.pair.hypothesis!r}")

    passed = bool(contradictions) == example["has_contradiction"]
    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    """Load the scorer once, then run all examples from data/examples.json."""
    examples = load_examples()
    print(f"Loaded {len(examples)} examples from {_EXAMPLES_PATH}")
    print("Loading NLI model (first run downloads ~1.5 GB, cached after that)...")
    scorer = NLIScorer()
    print("Model ready.\n")

    results = [run_case(scorer, ex) for ex in examples]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Done. {passed}/{total} passed.")


if __name__ == "__main__":
    main()
