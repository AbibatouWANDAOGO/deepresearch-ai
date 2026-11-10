"""Application settings, read from environment variables and the .env file.

Paid services are off by default: the project must run at zero cost with a local
model (Ollama). Choosing a paid LLM provider requires ALLOW_PAID_APIS=true, an API
key and a per-research spending cap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLMProvider = Literal["ollama", "anthropic", "openai"]
SearchProvider = Literal["tavily", "duckduckgo"]
PAID_LLM_PROVIDERS = {"anthropic", "openai"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Language model ------------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None  # no default: set it explicitly if you use OpenAI
    llm_timeout_seconds: float = Field(default=300.0, gt=0)
    llm_max_output_tokens: int = Field(default=4096, ge=256)
    ollama_num_ctx: int = Field(default=8192, ge=2048)  # context window requested from Ollama
    ollama_temperature: float = Field(default=0.1, ge=0, le=2)
    # Price used to estimate and cap spending (USD per million tokens, claude-opus-5-5).
    anthropic_input_usd_per_mtok: float = Field(default=4.0, ge=0)
    anthropic_output_usd_per_mtok: float = Field(default=20.0, ge=0)

    # --- Cost guardrails -----------------------------------------------------------
    allow_paid_apis: bool = False
    max_cost_usd_per_research: float = Field(default=0.50, ge=0)

    # --- Web search ----------------------------------------------------------------
    search_provider: SearchProvider = "tavily"
    search_fallback: Literal["duckduckgo", "none"] = "duckduckgo"
    tavily_api_key: SecretStr | None = None
    # Local guard on the free tier (1,000 credits a month; a basic search costs 1 credit).
    tavily_monthly_credit_limit: int = Field(default=1000, ge=0)
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    search_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Research limits -----------------------------------------------------------
    max_extra_research_cycles: int = Field(default=3, ge=0, le=3)
    max_sources_per_research: int = Field(default=20, ge=1, le=100)
    max_research_minutes: float = Field(default=30.0, gt=0)
    # Each LLM call takes about a minute on a laptop CPU: these bound the length of a research.
    max_sub_questions: int = Field(default=3, ge=1, le=8)
    max_claims_per_sub_question: int = Field(default=3, ge=1, le=10)
    passages_per_analysis: int = Field(default=5, ge=1, le=20)
    # A sub-question is answered when at least this share of its checked claims is supported.
    min_supported_ratio: float = Field(default=0.5, ge=0, le=1)

    # --- Downloads -----------------------------------------------------------------
    http_timeout_seconds: float = Field(default=20.0, gt=0)
    max_download_mb: float = Field(default=10.0, gt=0)
    max_redirects: int = Field(default=5, ge=0, le=10)
    respect_robots_txt: bool = True
    # Politeness and rate limit: seconds between two requests to the same site; a robots.txt
    # Crawl-delay is honoured up to max_crawl_delay_seconds.
    min_seconds_per_host: float = Field(default=1.0, ge=0)
    max_crawl_delay_seconds: float = Field(default=10.0, ge=0)
    user_agent: str = "DeepResearchAI/0.1 (research assistant; educational project)"

    # --- Document processing -------------------------------------------------------
    max_pdf_pages: int = Field(default=60, ge=1)
    min_document_chars: int = Field(default=300, ge=0)
    max_document_chars: int = Field(default=200_000, ge=1000)

    # --- RAG -----------------------------------------------------------------------
    # The specification suggests ~800 tokens, but multilingual-e5-base reads at most 512
    # tokens: longer chunks would be silently truncated when embedded. 480 + 15% overlap
    # keeps every chunk entirely visible to the embedding model.
    chunk_size_tokens: int = Field(default=480, ge=50)
    chunk_overlap_tokens: int = Field(default=72, ge=0)
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_batch_size: int = Field(default=16, ge=1)
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_enabled: bool = True
    retrieval_top_k: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=6, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    qdrant_collection: str = "chunks"

    # --- Storage -------------------------------------------------------------------
    storage_dir: Path = PROJECT_ROOT / "storage"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError("RERANK_TOP_K cannot exceed RETRIEVAL_TOP_K")
        if self.llm_provider in PAID_LLM_PROVIDERS:
            if not self.allow_paid_apis:
                raise ValueError(
                    f"LLM_PROVIDER={self.llm_provider} is a paid API: set ALLOW_PAID_APIS=true "
                    "to enable it, or use LLM_PROVIDER=ollama (free, local)"
                )
            if self.max_cost_usd_per_research <= 0:
                raise ValueError("a paid LLM provider needs MAX_COST_USD_PER_RESEARCH > 0")
            if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
                raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
            if self.llm_provider == "openai":
                if self.openai_api_key is None:
                    raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
                if not self.openai_model:
                    raise ValueError("LLM_PROVIDER=openai requires OPENAI_MODEL")
        return self

    # --- Derived values ------------------------------------------------------------
    @property
    def llm_model(self) -> str:
        return {
            "ollama": self.ollama_model,
            "anthropic": self.anthropic_model,
            "openai": self.openai_model or "",
        }[self.llm_provider]

    @property
    def documents_dir(self) -> Path:
        return self.storage_dir / "documents"

    @property
    def reports_dir(self) -> Path:
        return self.storage_dir / "reports"

    @property
    def documents_db_path(self) -> Path:
        return self.storage_dir / "documents.db"

    @property
    def checkpoints_db_path(self) -> Path:
        return self.storage_dir / "checkpoints.db"

    @property
    def search_dir(self) -> Path:
        return self.storage_dir / "search"

    @property
    def qdrant_dir(self) -> Path:
        return self.storage_dir / "qdrant"

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "deepresearch.db"

    def ensure_storage(self) -> None:
        for directory in (self.documents_dir, self.reports_dir, self.qdrant_dir, self.search_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _display_path(path: Path) -> str:
        """Path relative to the project when possible, so screens do not expose local folders."""
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    def summary(self) -> dict[str, str]:
        """Human-readable status. Never includes secret values, only whether they are set."""

        def is_set(secret: SecretStr | None) -> str:
            return "set" if secret is not None else "not set"

        return {
            "LLM provider": self.llm_provider,
            "LLM model": self.llm_model,
            "Paid APIs": "enabled" if self.allow_paid_apis else "disabled",
            "Cost cap per research": f"${self.max_cost_usd_per_research:.2f}",
            "Search provider": self.search_provider,
            "Search fallback": self.search_fallback,
            "Tavily API key": is_set(self.tavily_api_key),
            "Anthropic API key": is_set(self.anthropic_api_key),
            "OpenAI API key": is_set(self.openai_api_key),
            "Extra research cycles (max)": str(self.max_extra_research_cycles),
            "Chunk size / overlap (tokens)": f"{self.chunk_size_tokens} / {self.chunk_overlap_tokens}",
            "Embedding model": self.embedding_model,
            "Storage": self._display_path(self.storage_dir),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
