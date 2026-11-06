"""Split source texts into overlapping chunks that keep their exact position.

Chunks are built from whole sentences and paragraphs, packed up to ``chunk_size`` tokens,
and the next chunk repeats the last sentences of the previous one (up to ``overlap``
tokens). Each chunk records its character offsets in the source text, so a citation can
point to the exact passage, and its pages for PDFs (pages are separated by form feeds).
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable

from pydantic import BaseModel

TokenCounter = Callable[[str], int]
PAGE_SEPARATOR = "\f"
# A boundary after sentence punctuation followed by whitespace, or at a line break / page break.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+|\f")


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    research_id: str
    chunk_index: int
    text: str
    n_tokens: int
    start_char: int
    end_char: int
    page_start: int | None = None
    page_end: int | None = None


def whitespace_token_count(text: str) -> int:
    """Rough counter (words), used in tests and when no tokenizer is available."""
    return len(text.split())


def _units(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of sentences and paragraphs, whitespace excluded."""
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text):
        if text[start:match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    # trim leading/trailing whitespace inside each span
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        trimmed.append((s, e))
    return trimmed


def _split_long_unit(text: str, start: int, end: int, size: int, count: TokenCounter) -> list[tuple[int, int]]:
    """Hard-split a sentence longer than a chunk, on word boundaries."""
    pieces, piece_start, last_end = [], start, start
    for word in re.finditer(r"\S+", text[start:end]):
        word_end = start + word.end()
        if count(text[piece_start:word_end]) > size and last_end > piece_start:
            pieces.append((piece_start, last_end))
            piece_start = start + word.start()
        last_end = word_end
    pieces.append((piece_start, last_end))
    return pieces


def chunk_text(
    text: str,
    source_id: str,
    research_id: str,
    chunk_size: int,
    overlap: int,
    count: TokenCounter = whitespace_token_count,
    paged: bool | None = None,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    units: list[tuple[int, int]] = []
    for s, e in _units(text):
        units.extend(_split_long_unit(text, s, e, chunk_size, count) if count(text[s:e]) > chunk_size else [(s, e)])
    if not units:
        return []
    tokens = [count(text[s:e]) for s, e in units]

    paged = PAGE_SEPARATOR in text if paged is None else paged
    page_starts = [0] + [m.end() for m in re.finditer(PAGE_SEPARATOR, text)]

    def page_of(position: int) -> int | None:
        return bisect.bisect_right(page_starts, position) if paged else None

    chunks: list[Chunk] = []
    first = 0
    while first < len(units):
        last, total = first, tokens[first]
        while last + 1 < len(units) and total + tokens[last + 1] <= chunk_size:
            last += 1
            total += tokens[last]
        start, end = units[first][0], units[last][1]
        body = text[start:end].replace(PAGE_SEPARATOR, "\n")
        chunks.append(Chunk(
            chunk_id=f"{source_id}-{len(chunks):04d}",
            source_id=source_id,
            research_id=research_id,
            chunk_index=len(chunks),
            text=body,
            n_tokens=count(body),
            start_char=start,
            end_char=end,
            page_start=page_of(start),
            page_end=page_of(end - 1),
        ))
        if last == len(units) - 1:
            break
        # Next chunk starts with the trailing units of this one that fit in the overlap.
        next_first, carried = last + 1, 0
        while next_first - 1 > first and carried + tokens[next_first - 1] <= overlap:
            next_first -= 1
            carried += tokens[next_first]
        first = next_first
    return chunks
