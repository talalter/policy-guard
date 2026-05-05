"""Manual test script for LLM judge.

Loads examples from data/examples.json and runs each through the LLM judge.
Requires OPENAI_API_KEY to be set in the environment or a .env file in the
project root.

Run from the backend/ directory:
    python test_llm_judge.py

To add examples, edit data/examples.json — do not add them here.
"""

import json
import logging
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from backend.core import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_EXAMPLES_PATH = pathlib.Path(__file__).parent.parent / "data" / "examples.json"


def load_examples() -> list[dict]:
    """Load labeled examples from the shared data file."""
    with _EXAMPLES_PATH.open() as f:
        return json.load(f)


def run_case(judge: LLMJudge, example: dict) -> bool:
    """Run a single example and print results. Returns True if PASS."""
    print(f"\n{'='*60}")
    print(f"CASE [{example['id']}]: {example['contradiction_type'].upper()} — {example.get('notes', '')}")
    print(f"{'='*60}")
    print(f"Context:  {example['context'][:120]}{'...' if len(example['context']) > 120 else ''}")
    print(f"Response: {example['response'][:120]}{'...' if len(example['response']) > 120 else ''}")
    print(f"Expected contradiction: {example['has_contradiction']}")
    print()

    contradictions = judge.judge(
        context=example["context"],
        response=example["response"],
        uncertain_pairs=[],
    )

    print(f"Contradictions found: {len(contradictions)}")
    for c in contradictions:
        print(f"  [{c.severity.value:10}] conf={c.confidence:.2f}")
        print(f"    Response span: {c.response_span!r}")
        print(f"    Context span:  {c.context_span!r}")
        print(f"    Explanation:   {c.explanation}")

    passed = bool(contradictions) == example["has_contradiction"]
    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    """Load the judge once, then run all examples from data/examples.json."""
    examples = load_examples()
    print(f"Loaded {len(examples)} examples from {_EXAMPLES_PATH}")
    print("Initialising LLM judge...")
    judge = LLMJudge()
    print("Ready.\n")

    results = [run_case(judge, ex) for ex in examples]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Done. {passed}/{total} passed.")


if __name__ == "__main__":
    main()
