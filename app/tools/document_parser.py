"""Document collection: download, extract, clean, de-duplicate and store sources.

    python -m app.tools.document_parser --query "your question" [--max-results 5]
    python -m app.tools.document_parser --url https://example.org/page [--url ...]

For each URL the collector records a ``Source`` (metadata + access status) in SQLite and,
when the text is usable, writes it to ``storage/documents/<research_id>/<source_id>.txt``
(PDF pages separated by form feeds, so later citations can point to a page).

Extracted text is untrusted content: it is stored and later quoted as data, and any
instruction it contains is never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
import uuid
from collections import Counter
from pathlib import Path

import trafilatura
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.database.connection import connect
from app.database.models import AccessStatus, Source, SourceRepository, SourceType
from app.tools.pdf_reader import ExtractionError, extract_pdf
from app.tools.web_scraper import Fetcher, FetchResult, FetchStatus
from app.tools.web_search import SearchResult, normalize_url

PAGE_SEPARATOR = "\f"
EXCERPT_CHARS = 300
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]")  # controls, zero-width space, BOM
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(/|of|sur)\s*\d{1,4})?$", re.IGNORECASE)


# --- Cleaning ----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace, drop control characters and empty runs of lines."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop page numbers and running headers/footers (lines found on most pages)."""
    if len(pages) < 3:
        return [_drop_page_numbers(p) for p in pages]
    page_lines = [{line for line in page.split("\n") if len(line) > 3} for page in pages]
    counts = Counter(line for lines in page_lines for line in lines)
    repeated = {line for line, n in counts.items() if n >= max(3, len(pages) // 2 + 1)}
    return [
        _drop_page_numbers("\n".join(line for line in page.split("\n") if line not in repeated))
        for page in pages
    ]


def _drop_page_numbers(page: str) -> str:
    return "\n".join(line for line in page.split("\n") if not _PAGE_NUMBER.match(line.strip()))


def guess_pdf_title(first_page: str, max_lines: int = 3, max_chars: int = 200) -> str | None:
    """Title from the first lines of page 1, for PDFs without a metadata title.

    Stops at the first line that looks like an author or affiliation line (commas, an
    e-mail address) or that is too long to be part of a title.
    """
    title_lines: list[str] = []
    for line in (line.strip() for line in first_page.split("\n")):
        if not line:
            if title_lines:
                break
            continue
        if "," in line or "@" in line or len(line) > 120 or len(title_lines) == max_lines:
            break
        title_lines.append(line)
    title = " ".join(title_lines)
    return title if 10 <= len(title) <= max_chars else None


def content_hash(text: str) -> str:
    """Hash insensitive to case, punctuation and spacing: same article, same hash."""
    canonical = re.sub(r"[^\w]+", " ", text.lower()).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Extraction --------------------------------------------------------------------------


class Extracted(BaseModel):
    text: str
    source_type: SourceType
    title: str | None = None
    author: str | None = None
    published_date: str | None = None
    site_name: str | None = None
    n_pages: int | None = None


def extract(fetched: FetchResult, max_pdf_pages: int) -> Extracted:
    if fetched.kind == "pdf":
        pdf = extract_pdf(fetched.content or b"", max_pdf_pages)
        pages = remove_repeated_lines([clean_text(p) for p in pdf.pages])
        return Extracted(text=PAGE_SEPARATOR.join(pages), source_type="pdf", title=pdf.title,
                         author=pdf.author, published_date=pdf.created, n_pages=len(pages))
    if fetched.kind == "html":
        doc = trafilatura.bare_extraction(
            fetched.content, url=fetched.final_url, with_metadata=True, include_comments=False,
            include_tables=True, favor_precision=True,
        )
        if doc is None or not doc.text:
            raise ExtractionError("no main content found in the page")
        return Extracted(text=clean_text(doc.text), source_type="web_page", title=doc.title,
                         author=doc.author, published_date=doc.date, site_name=doc.sitename)
    if fetched.kind == "text":
        return Extracted(text=clean_text((fetched.content or b"").decode("utf-8", "replace")),
                         source_type="text")
    raise ExtractionError(f"no extractor for {fetched.kind}")


# --- Collector ---------------------------------------------------------------------------


class CollectTarget(BaseModel):
    url: str
    title_hint: str | None = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> CollectTarget:
        return cls(url=result.url, title_hint=result.title or None)


class CollectionReport(BaseModel):
    research_id: str
    sources: list[Source]

    @property
    def usable(self) -> list[Source]:
        return [s for s in self.sources if s.usable]

    def counts(self) -> dict[str, int]:
        return dict(Counter(s.access_status.value for s in self.sources))


def source_id_for(research_id: str, url: str) -> str:
    return hashlib.sha1(f"{research_id}|{url}".encode()).hexdigest()[:16]


class DocumentCollector:
    def __init__(self, fetcher: Fetcher, repository: SourceRepository, settings: Settings):
        self.fetcher = fetcher
        self.repository = repository
        self.settings = settings

    def collect(self, targets: list[CollectTarget | str], research_id: str) -> CollectionReport:
        sources: list[Source] = []
        for target in targets:
            if isinstance(target, str):
                target = CollectTarget(url=target)
            if sum(s.usable for s in sources) >= self.settings.max_sources_per_research:
                break
            url = normalize_url(target.url)
            if url is None:
                continue
            existing = self.repository.by_url(research_id, url)
            if existing is not None:  # already collected for this research
                sources.append(existing)
                continue
            source = self._collect_one(url, target.title_hint, research_id)
            self.repository.save(source)
            sources.append(source)
        return CollectionReport(research_id=research_id, sources=sources)

    def _collect_one(self, url: str, title_hint: str | None, research_id: str) -> Source:
        sid = source_id_for(research_id, url)
        base = {"source_id": sid, "research_id": research_id, "url": url, "title": title_hint}

        fetched = self.fetcher.fetch(url)
        base.update(final_url=fetched.final_url, collected_at=fetched.fetched_at)
        if fetched.status != FetchStatus.OK:
            return Source(**base, access_status=AccessStatus(fetched.status.value), error=fetched.error)

        try:
            doc = extract(fetched, self.settings.max_pdf_pages)
        except ExtractionError as exc:
            return Source(**base, access_status=AccessStatus.EXTRACTION_FAILED, error=str(exc),
                          source_type="pdf" if fetched.kind == "pdf" else "web_page")

        text = doc.text
        truncated = len(text) > self.settings.max_document_chars
        text = text[: self.settings.max_document_chars]
        title = doc.title or title_hint
        if title is None and doc.source_type == "pdf":
            title = guess_pdf_title(text.split(PAGE_SEPARATOR, 1)[0])
        meta = {
            **base,
            "title": title,
            "author": doc.author,
            "published_date": doc.published_date,
            "site_name": doc.site_name,
            "source_type": doc.source_type,
            "n_pages": doc.n_pages,
            "n_chars": len(text),
            "truncated": truncated,
            "excerpt": text.replace(PAGE_SEPARATOR, " ")[:EXCERPT_CHARS],
        }
        if len(text.replace(PAGE_SEPARATOR, "").strip()) < self.settings.min_document_chars:
            return Source(**meta, access_status=AccessStatus.TOO_SHORT,
                          error=f"fewer than {self.settings.min_document_chars} characters")

        digest = content_hash(text)
        duplicate = self.repository.by_hash(research_id, digest)
        if duplicate is not None:
            return Source(**meta, content_hash=digest, access_status=AccessStatus.DUPLICATE,
                          duplicate_of=duplicate.source_id, error=f"same text as {duplicate.url}")

        relative = Path("documents") / research_id / f"{sid}.txt"
        path = self.settings.storage_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return Source(**meta, content_hash=digest, access_status=AccessStatus.OK,
                      text_path=relative.as_posix())


def load_text(source: Source, settings: Settings) -> str:
    if not source.text_path:
        raise FileNotFoundError(f"source {source.source_id} has no stored text")
    return (settings.storage_dir / source.text_path).read_text(encoding="utf-8")


def build_collector(settings: Settings) -> DocumentCollector:
    fetcher = Fetcher(
        timeout=settings.http_timeout_seconds,
        max_bytes=int(settings.max_download_mb * 1024 * 1024),
        max_redirects=settings.max_redirects,
        user_agent=settings.user_agent,
        respect_robots=settings.respect_robots_txt,
    )
    return DocumentCollector(fetcher, SourceRepository(connect(settings.documents_db_path)), settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect documents from URLs or from a web search.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query")
    group.add_argument("--url", action="append")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--research-id", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_storage()
    research_id = args.research_id or f"cli-{uuid.uuid4().hex[:8]}"
    if args.query:
        from app.tools.web_search import SearchError, build_web_search

        try:
            response = build_web_search(settings).search(
                args.query, args.max_results or settings.max_results_per_query, research_id
            )
        except (SearchError, ValueError) as exc:
            print(f"Search failed: {exc}", file=sys.stderr)
            return 1
        targets = [CollectTarget.from_search_result(r) for r in response.results]
        print(f"Search: {len(targets)} results from {response.provider}")
    else:
        targets = [CollectTarget(url=u) for u in args.url]

    report = build_collector(settings).collect(targets, research_id)
    print(f"Research {research_id}: {report.counts()}")
    for s in report.sources:
        detail = f"{s.n_chars:>7} chars" + (f", {s.n_pages} pages" if s.n_pages else "")
        line = f"[{s.access_status.value:<18}] {s.source_type or '-':<8} {detail}  {s.title or s.url}"
        print(line[:150])
        if s.error and not s.usable:
            print(f"{'':21}{s.error[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
