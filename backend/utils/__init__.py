"""Utilities package: text processing, deduplication, and metrics."""

__all__ = [
    "split_sentences",
    "get_stopwords",
    "tokenize",
    "jaccard",
    "deduplicate",
]

from backend.utils.dedup import deduplicate, jaccard, tokenize
from backend.utils.text import get_stopwords, split_sentences
