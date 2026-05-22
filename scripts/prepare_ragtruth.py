"""Download and sample RAGTruth examples for benchmark evaluation.

Loads wandb/RAGTruth-processed from HuggingFace, filters to Summary and QA
task types (natural-language context), samples 50 contradiction examples and
50 faithful examples, and saves to data/ragtruth_sample.json in the same
schema as the benchmark datasets.

Contradiction mapping:
    evident_conflict > 0  →  has_contradiction=True, contradiction_type="direct"
    baseless_info only    →  skipped (fabrication, not contradiction)
    both zero             →  has_contradiction=False, contradiction_type="none"

Usage:
    python scripts/prepare_ragtruth.py
"""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_PER_CLASS = 50
MAX_CONTEXT_CHARS = 1500   # keep examples short enough for fast NLI inference
MAX_RESPONSE_CHARS = 800
VALID_TASK_TYPES = {"Summary", "QA"}
OUT_PATH = Path(__file__).parent.parent / "data" / "ragtruth_sample.json"


def _to_example(row: dict, idx: int, has_contradiction: bool) -> dict:
    """Convert a RAGTruth row into the shared example schema."""
    return {
        "id": f"ragt_{idx:03d}",
        "source": "ragtruth",
        "context": row["context"][:MAX_CONTEXT_CHARS],
        "response": row["output"][:MAX_RESPONSE_CHARS],
        "has_contradiction": has_contradiction,
        "contradiction_type": "direct" if has_contradiction else "none",
        "notes": f"RAGTruth {row['task_type']} example — model: {row['model']}",
    }


def main() -> None:
    """Download RAGTruth, sample balanced examples, and save to data/."""
    from datasets import load_dataset

    logger.info("Loading wandb/RAGTruth-processed from HuggingFace...")
    ds = load_dataset("wandb/RAGTruth-processed", split="train+test")
    logger.info("Loaded %d total examples", len(ds))

    positives, negatives = [], []
    for _row in ds:
        row: dict = dict(_row)
        if row["task_type"] not in VALID_TASK_TYPES:
            continue
        if not row["context"] or not row["output"]:
            continue
        if row.get("quality") != "good":
            continue
        processed = row["hallucination_labels_processed"]
        if processed.get("evident_conflict", 0) > 0:
            positives.append(row)
        elif processed.get("baseless_info", 0) == 0:
            negatives.append(row)

    logger.info("Found %d contradiction examples, %d faithful examples", len(positives), len(negatives))

    random.seed(SEED)
    sampled_pos = random.sample(positives, min(N_PER_CLASS, len(positives)))
    sampled_neg = random.sample(negatives, min(N_PER_CLASS, len(negatives)))

    examples = (
        [_to_example(row, i + 1, has_contradiction=True) for i, row in enumerate(sampled_pos)]
        + [_to_example(row, i + 1 + len(sampled_pos), has_contradiction=False) for i, row in enumerate(sampled_neg)]
    )

    with open(OUT_PATH, "w") as f:
        json.dump(examples, f, indent=2)

    logger.info(
        "Saved %d examples (%d positive, %d negative) to %s",
        len(examples), len(sampled_pos), len(sampled_neg), OUT_PATH,
    )


if __name__ == "__main__":
    main()
