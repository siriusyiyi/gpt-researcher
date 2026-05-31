"""Text processing utilities."""

from __future__ import annotations

import re


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs separated by blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? followed by whitespace.

    Keeps the punctuation attached to the sentence.
    """
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
