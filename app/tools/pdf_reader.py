"""Text extraction from PDF files with PyMuPDF, page by page (pages are kept for citations)."""

from __future__ import annotations

import re

import pymupdf
from pydantic import BaseModel


class ExtractionError(ValueError):
    """The document could not be turned into text."""


class PdfText(BaseModel):
    pages: list[str]
    total_pages: int
    title: str | None = None
    author: str | None = None
    created: str | None = None  # YYYY-MM-DD when the PDF metadata has it


def _pdf_date(value: str | None) -> str | None:
    """PDF dates look like D:20240305120000+01'00'."""
    match = re.match(r"D?:?(\d{4})(\d{2})(\d{2})", value or "")
    return f"{match[1]}-{match[2]}-{match[3]}" if match else None


def extract_pdf(content: bytes, max_pages: int) -> PdfText:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several types for corrupted files
        raise ExtractionError(f"not a readable PDF: {exc}") from exc
    with document:
        if document.needs_pass:
            raise ExtractionError("the PDF is password-protected")
        pages = [document[i].get_text("text") for i in range(min(max_pages, document.page_count))]
        meta = document.metadata or {}
        return PdfText(
            pages=pages,
            total_pages=document.page_count,
            title=(meta.get("title") or "").strip() or None,
            author=(meta.get("author") or "").strip() or None,
            created=_pdf_date(meta.get("creationDate")),
        )
