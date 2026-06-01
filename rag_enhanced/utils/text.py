"""Text processing utilities."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Sentence splitting with abbreviation awareness
# ---------------------------------------------------------------------------

# Known abbreviation patterns — checked against text BEFORE the period.
# Each pattern matches at the END of a prefix string.
_ABBREV_SINGLE_CAP = re.compile(r"\b[A-Z]\.$")                         # U. S.
_ABBREV_TITLE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r"|Prof|Dr|Mr|Mrs|Ms|Rev|Gen|Col|Sgt|Capt|Lt"
    r"|Inc|Ltd|Corp|Co|St|Jr|Sr|Dept|Univ|Assn|Mt|Ft"
    r"|vs|etc|cf|al|approx|no|vol|pp|ch|sec|fig|eq)\.$",
    re.IGNORECASE,
)
_ABBREV_LATIN_EG = re.compile(r"\be\.g\.$")
_ABBREV_LATIN_IE = re.compile(r"\bi\.e\.$")


def _is_abbreviation_boundary(text: str, period_pos: int) -> bool:
    """Return True if the period at *period_pos* is part of an abbreviation,
    not a real sentence boundary.
    """
    prefix = text[: period_pos + 1]  # up to and including the period

    if _ABBREV_SINGLE_CAP.search(prefix):
        return True
    if _ABBREV_TITLE.search(prefix):
        return True
    if _ABBREV_LATIN_EG.search(prefix):
        return True
    if _ABBREV_LATIN_IE.search(prefix):
        return True

    # Part of an ellipsis (two or more consecutive dots before this one)
    if prefix.rstrip().endswith(".."):
        return True

    return False


# Sentence boundary: [.!?] then whitespace then uppercase or quote
_SENTENCE_BOUNDARY = re.compile(r"([.!?])\s+(?=[A-Z\"'\x80-\xff])")


def split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentences with abbreviation awareness.

    Strategy:
        1. Find all positions where ``[.!?]`` is followed by whitespace
           and an uppercase letter or quote.
        2. Exclude positions that belong to known abbreviations.
        3. Split at the remaining boundaries.
    """
    if not text or not text.strip():
        return []

    # Collect real sentence boundary positions (split AFTER the whitespace)
    boundaries: list[int] = []
    for m in _SENTENCE_BOUNDARY.finditer(text):
        if _is_abbreviation_boundary(text, m.start()):
            continue
        boundaries.append(m.end())  # split after the whitespace

    if not boundaries:
        return [text.strip()]

    result: list[str] = []
    prev = 0
    for pos in boundaries:
        result.append(text[prev:pos].strip())
        prev = pos
    result.append(text[prev:].strip())
    return [s for s in result if s]


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs separated by blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    # Coerce numpy arrays / other sequences to plain lists
    if not isinstance(a, list):
        a = list(a)
    if not isinstance(b, list):
        b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
