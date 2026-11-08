"""Text checks that tie a claim to the passages cited for it, without a model call.

Used by the Fact Checker to keep a verdict no stronger than the cited text, and by the
evaluation to measure citations.
"""

from __future__ import annotations

import re

from app.rag.retriever import bm25_tokens

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_THOUSANDS = re.compile(r"(?<=\d)[ ,  ](?=\d{3}(?!\d))")


def numbers(text: str) -> set[str]:
    """'USD 1 400' and '1,400' -> '1400'; '2,5' -> '2.5'."""
    return {n.replace(",", ".") for n in _NUMBER.findall(_THOUSANDS.sub("", text))}


def stems(text: str) -> set[str]:
    return {t[:6] for t in bm25_tokens(text) if not t.isdigit()}


def word_overlap(claim: str, passage: str) -> float:
    """Share of the claim's words (6-letter stems, no stop words) found in the passage."""
    words = stems(claim)
    return len(words & stems(passage)) / len(words) if words else 0.0


def missing_numbers(claim: str, passages: list[str]) -> set[str]:
    """Numbers of the claim that none of the passages contains."""
    return numbers(claim) - numbers(" ".join(passages))
