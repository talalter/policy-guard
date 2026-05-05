"""Text processing utilities: sentence splitting, stopword management."""

import logging

import nltk

logger = logging.getLogger(__name__)


def _load_stopwords() -> set[str]:
    """Return NLTK's English stopword corpus, downloading it if absent."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


_STOPWORDS = _load_stopwords()


def _ensure_nltk_punkt() -> None:
    """Download the punkt tokenizer data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# Download once at import time — not on every sentence split.
_ensure_nltk_punkt()


def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences using NLTK's punkt tokenizer."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def get_stopwords() -> set[str]:
    """Return the cached English stopwords set."""
    return _STOPWORDS
