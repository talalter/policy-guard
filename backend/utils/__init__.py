"""Utilities package: text processing, deduplication, and metrics."""

__all__ = [
    "split_sentences",
    "tokenize",
    "jaccard",
    "deduplicate",
]

from backend.utils.dedup import deduplicate, jaccard, tokenize
from backend.utils.text import split_sentences
