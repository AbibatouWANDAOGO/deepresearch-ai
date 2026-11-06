"""SQLite connection and schema. Metadata lives here; extracted text lives in files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT PRIMARY KEY,
    research_id     TEXT NOT NULL,
    url             TEXT NOT NULL,
    final_url       TEXT NOT NULL,
    title           TEXT,
    excerpt         TEXT,
    published_date  TEXT,
    author          TEXT,
    site_name       TEXT,
    collected_at    TEXT NOT NULL,
    source_type     TEXT,
    access_status   TEXT NOT NULL,
    error           TEXT,
    content_hash    TEXT,
    duplicate_of    TEXT,
    n_chars         INTEGER NOT NULL DEFAULT 0,
    n_pages         INTEGER,
    truncated       INTEGER NOT NULL DEFAULT 0,
    text_path       TEXT,
    UNIQUE (research_id, url)
);
CREATE INDEX IF NOT EXISTS idx_sources_research ON sources (research_id);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources (research_id, content_hash);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection
