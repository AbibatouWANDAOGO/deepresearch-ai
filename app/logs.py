"""Structured logs (one JSON object per line) that never contain a secret.

Every line goes through ``redact`` before it is written: the API keys configured in the
settings are replaced by their name, and anything shaped like a key or a bearer token is
masked even when it is not configured (a key pasted into a question or returned in a
provider's error message). ``redact`` is also applied to the error messages stored in the
research history.

    storage/logs/deepresearch.jsonl   rotated at 5 MB, 3 files kept
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

from app.config import Settings

LOGGER_NAME = "deepresearch"
_SECRET_FIELDS = ("anthropic_api_key", "openai_api_key", "tavily_api_key")
_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    re.compile(r"(?i)((?:api[_-]?key|x-api-key|authorization|token|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}&]{6,}"),
]
_known: dict[str, str] = {}  # secret value -> placeholder


def register_secrets(settings: Settings) -> None:
    for field in _SECRET_FIELDS:
        secret = getattr(settings, field, None)
        value = secret.get_secret_value() if secret is not None else ""
        if len(value) >= 6:
            _known[value] = f"[{field.upper()}]"


def redact(text: str) -> str:
    for value, placeholder in _known.items():
        text = text.replace(value, placeholder)
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


class JsonFormatter(logging.Formatter):
    """``logger.info("fetch", extra={"fields": {...}})`` -> {"at", "level", "event", ...fields}."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"at": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "event": record.getMessage(), **getattr(record, "fields", {})}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(entry, ensure_ascii=False, default=str))


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log(event: str, level: int = logging.INFO, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})


def setup_logging(settings: Settings) -> logging.Logger:
    """Idempotent: one file handler, in the storage directory of the latest settings."""
    register_secrets(settings)
    logger = get_logger()
    logger.setLevel(settings.log_level)
    logger.propagate = False
    path = (settings.storage_dir / "logs" / "deepresearch.jsonl").resolve()
    for handler in [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]:
        if handler.baseFilename == str(path):
            return logger
        logger.removeHandler(handler)
        handler.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
