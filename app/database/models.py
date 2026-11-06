"""Source records: what was collected for a research, with provenance and access status."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

SourceType = Literal["web_page", "pdf", "text"]


class AccessStatus(StrEnum):
    OK = "ok"
    DUPLICATE = "duplicate"
    TOO_SHORT = "too_short"
    EXTRACTION_FAILED = "extraction_failed"
    UNSAFE_URL = "unsafe_url"
    ROBOTS_DISALLOWED = "robots_disallowed"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    TOO_MANY_REDIRECTS = "too_many_redirects"


class Source(BaseModel):
    source_id: str
    research_id: str
    url: str
    final_url: str
    title: str | None = None
    excerpt: str | None = None
    published_date: str | None = None
    author: str | None = None
    site_name: str | None = None
    collected_at: datetime
    source_type: SourceType | None = None
    access_status: AccessStatus
    error: str | None = None
    content_hash: str | None = None
    duplicate_of: str | None = None
    n_chars: int = 0
    n_pages: int | None = None
    truncated: bool = False
    text_path: str | None = None

    @property
    def usable(self) -> bool:
        return self.access_status == AccessStatus.OK


class SourceRepository:
    def __init__(self, connection: sqlite3.Connection):
        self._db = connection

    def save(self, source: Source) -> None:
        row = source.model_dump(mode="json")
        row["truncated"] = int(source.truncated)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        self._db.execute(f"INSERT OR REPLACE INTO sources ({columns}) VALUES ({placeholders})", row)
        self._db.commit()

    def get(self, source_id: str) -> Source | None:
        row = self._db.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return self._to_source(row) if row else None

    def by_url(self, research_id: str, url: str) -> Source | None:
        row = self._db.execute(
            "SELECT * FROM sources WHERE research_id = ? AND url = ?", (research_id, url)
        ).fetchone()
        return self._to_source(row) if row else None

    def by_hash(self, research_id: str, content_hash: str) -> Source | None:
        row = self._db.execute(
            "SELECT * FROM sources WHERE research_id = ? AND content_hash = ? AND access_status = ?",
            (research_id, content_hash, AccessStatus.OK.value),
        ).fetchone()
        return self._to_source(row) if row else None

    def list(self, research_id: str) -> list[Source]:
        rows = self._db.execute(
            "SELECT * FROM sources WHERE research_id = ? ORDER BY collected_at", (research_id,)
        ).fetchall()
        return [self._to_source(r) for r in rows]

    @staticmethod
    def _to_source(row: sqlite3.Row) -> Source:
        data = dict(row)
        data["truncated"] = bool(data["truncated"])
        return Source(**data)


class ChunkRepository:
    """Chunks of usable sources; the text is kept here so BM25 and citations never need Qdrant."""

    def __init__(self, connection: sqlite3.Connection):
        self._db = connection

    def save_many(self, chunks: list) -> None:
        rows = [c.model_dump() for c in chunks]
        if not rows:
            return
        columns = ", ".join(rows[0])
        placeholders = ", ".join(f":{name}" for name in rows[0])
        self._db.executemany(f"INSERT OR REPLACE INTO chunks ({columns}) VALUES ({placeholders})", rows)
        self._db.commit()

    def list(self, research_id: str) -> list:
        from app.rag.chunking import Chunk

        rows = self._db.execute(
            "SELECT * FROM chunks WHERE research_id = ? ORDER BY source_id, chunk_index", (research_id,)
        ).fetchall()
        return [Chunk(**dict(r)) for r in rows]

    def indexed_sources(self, research_id: str) -> set[str]:
        rows = self._db.execute(
            "SELECT DISTINCT source_id FROM chunks WHERE research_id = ?", (research_id,)
        ).fetchall()
        return {r["source_id"] for r in rows}
