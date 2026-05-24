"""Text processing utilities: sentence splitting and tool context normalisation."""

import json
import logging
from typing import Any

import nltk

logger = logging.getLogger(__name__)


def _ensure_nltk_punkt() -> None:
    """Download the punkt tokenizer data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# Download once at import time - not on every sentence split.
_ensure_nltk_punkt()


def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences using NLTK's punkt tokenizer."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Tool context normalisation ────────────────────────────────────────────────

def _parse_tool_context(context: str) -> tuple[str, str] | None:
    """Return (tool_name, result_text) if context is a tool call result, else None."""
    lines = context.strip().split("\n", 1)
    if len(lines) != 2:
        return None
    if not lines[0].startswith("Tool: ") or not lines[1].startswith("Result: "):
        return None
    return lines[0].removeprefix("Tool: ").strip(), lines[1].removeprefix("Result: ").strip()


def _flatten_json(data: Any) -> list[str]:
    """Recursively flatten a JSON value into 'The <key> is <value>.' sentences."""
    sentences: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                sentences.extend(_flatten_json(value))
            else:
                sentences.append(f"The {key} is {value}.")
    elif isinstance(data, list):
        for item in data:
            sentences.extend(_flatten_json(item))
    else:
        sentences.append(str(data))
    return sentences


def is_tool_context(context: str) -> bool:
    """Return True if context is a structured 'Tool: / Result:' call result."""
    return _parse_tool_context(context) is not None


def flatten_tool_context(context: str) -> str:
    """Convert a structured tool call result to natural language prose.

    Transforms 'Tool: <name>\\nResult: <json>' into readable sentences so the
    bi-encoder similarity filter and NLI cross-encoder can process it effectively.
    Non-tool contexts are returned unchanged.
    """
    parsed = _parse_tool_context(context)
    if parsed is None:
        return context
    tool_name, result_text = parsed
    intro = f"Tool {tool_name} was called."
    try:
        result_data = json.loads(result_text)
        sentences = _flatten_json(result_data)
        return " ".join([intro] + sentences)
    except (json.JSONDecodeError, ValueError):
        return f"{intro} The result was: {result_text}"
