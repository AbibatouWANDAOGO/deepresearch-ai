"""Structured logs and secret redaction."""

import json
import logging

import pytest

from app.config import Settings
from app.database.connection import connect
from app.jobs import ResearchParams, ResearchRepository, ResearchStatus, run_job
from app.logs import JsonFormatter, log, redact, register_secrets, setup_logging

KEY = "tvly-dev-0123456789abcdef"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return Settings(_env_file=None, storage_dir=tmp_path, tavily_api_key="my-configured-key-123")


@pytest.mark.parametrize("text", [
    "Error: invalid key sk-ant-api03-AbCdEfGhIjKlMnOp",
    "openai sk-proj-abcdefghijklmnopqrstuvwx rejected",
    f"tavily said {KEY} is unknown",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload",
    '{"api_key": "abcdef123456", "query": "heat pumps"}',
    "https://api.example.com/v1?token=abcdef123456&q=x",
])
def test_key_shaped_strings_are_masked(text):
    cleaned = redact(text)
    assert "[REDACTED]" in cleaned
    for secret in ("AbCdEfGhIjKlMnOp", "abcdefghijklmnopqrstuvwx", KEY, "eyJhbGciOiJIUzI1NiJ9", "abcdef123456"):
        assert secret not in cleaned


def test_configured_keys_are_replaced_by_their_name(settings):
    register_secrets(settings)
    assert redact("request failed with my-configured-key-123") == "request failed with [TAVILY_API_KEY]"


def test_ordinary_text_is_untouched():
    text = "llm_call input_tokens=141139 output_tokens=25306; 3 sources, see https://example.org/a?page=2"
    assert redact(text) == text


def test_log_lines_are_json_without_secrets(settings):
    setup_logging(settings)
    log("web_search", provider="tavily", errors=[f"401 for key {KEY}"])
    for handler in logging.getLogger("deepresearch").handlers:
        handler.flush()
    lines = (settings.storage_dir / "logs" / "deepresearch.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    assert entry["event"] == "web_search" and entry["provider"] == "tavily" and entry["level"] == "INFO"
    assert KEY not in lines[-1]


def test_formatter_masks_exception_text():
    try:
        raise RuntimeError(f"bad key {KEY}")
    except RuntimeError:
        import sys
        record = logging.LogRecord("deepresearch", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    assert KEY not in JsonFormatter().format(record)


def test_a_failed_research_stores_and_logs_its_error_without_the_key(settings):
    class Leaky:
        def stream(self, *args):
            raise RuntimeError("provider rejected my-configured-key-123")

    repo = ResearchRepository(connect(settings.documents_db_path))
    record = repo.create("q?", "en", ResearchParams())
    with pytest.raises(RuntimeError):
        run_job(record.research_id, base=settings, runner_factory=lambda s: Leaky())
    stored = repo.get(record.research_id)
    assert stored.status == ResearchStatus.FAILED
    assert stored.error == "RuntimeError: provider rejected [TAVILY_API_KEY]"
    text = (settings.storage_dir / "logs" / "deepresearch.jsonl").read_text(encoding="utf-8")
    assert "research_failed" in text and "my-configured-key-123" not in text
