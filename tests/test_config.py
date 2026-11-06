import re

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Tests never read the developer's real .env or environment variables."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- Defaults: free and local ----------------------------------------------------------


def test_defaults_are_free_and_local():
    s = make()
    assert s.llm_provider == "ollama"
    assert s.allow_paid_apis is False
    assert s.llm_model == s.ollama_model
    assert s.search_provider == "tavily"


def test_defaults_follow_the_specification():
    s = make()
    assert (s.chunk_size_tokens, s.chunk_overlap_tokens) == (800, 120)
    assert s.max_extra_research_cycles == 3
    assert s.embedding_model == "intfloat/multilingual-e5-base"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("MAX_SOURCES_PER_RESEARCH", "12")
    s = make()
    assert s.ollama_model == "llama3.2:3b" and s.max_sources_per_research == 12


# --- Paid providers need an explicit opt-in --------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_paid_provider_is_refused_without_opt_in(provider):
    with pytest.raises(ValidationError, match="ALLOW_PAID_APIS"):
        make(llm_provider=provider, anthropic_api_key="k", openai_api_key="k", openai_model="m")


def test_paid_provider_needs_a_key():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        make(llm_provider="anthropic", allow_paid_apis=True)


def test_openai_needs_an_explicit_model():
    with pytest.raises(ValidationError, match="OPENAI_MODEL"):
        make(llm_provider="openai", allow_paid_apis=True, openai_api_key="k")


def test_paid_provider_needs_a_cost_cap():
    with pytest.raises(ValidationError, match="MAX_COST_USD_PER_RESEARCH"):
        make(llm_provider="anthropic", allow_paid_apis=True, anthropic_api_key="k",
             max_cost_usd_per_research=0)


def test_paid_provider_with_opt_in_and_key_is_accepted():
    s = make(llm_provider="anthropic", allow_paid_apis=True, anthropic_api_key="sk-test")
    assert s.llm_model == "claude-opus-5-5"


# --- Limits ----------------------------------------------------------------------------


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make(chunk_size_tokens=500, chunk_overlap_tokens=500)


def test_extra_research_cycles_are_capped_at_three():
    with pytest.raises(ValidationError):
        make(max_extra_research_cycles=4)


def test_rerank_cannot_keep_more_than_retrieved():
    with pytest.raises(ValidationError, match="RERANK_TOP_K"):
        make(retrieval_top_k=5, rerank_top_k=10)


@pytest.mark.parametrize("field", ["http_timeout_seconds", "max_download_mb", "llm_timeout_seconds"])
def test_timeouts_and_sizes_must_be_positive(field):
    with pytest.raises(ValidationError):
        make(**{field: 0})


# --- Secrets ---------------------------------------------------------------------------


def test_secrets_never_appear_in_summary_or_repr():
    secret = "sk-very-secret-value"
    s = make(llm_provider="anthropic", allow_paid_apis=True, anthropic_api_key=secret,
             tavily_api_key=secret)
    assert secret not in str(s.summary())
    assert secret not in repr(s)
    assert s.summary()["Anthropic API key"] == "set"


# --- Files -----------------------------------------------------------------------------


def test_env_example_documents_every_setting():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= documented, f"missing in .env.example: {sorted(expected - documented)}"


def test_env_example_contains_no_secret_values():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
        assert re.search(rf"^{key}=$", text, flags=re.MULTILINE), f"{key} must be empty"


def test_env_file_is_ignored_by_git():
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and "!.env.example" in ignored


def test_storage_directories_are_created(tmp_path):
    s = make(storage_dir=tmp_path / "storage")
    s.ensure_storage()
    assert s.documents_dir.is_dir() and s.reports_dir.is_dir() and s.qdrant_dir.is_dir()
