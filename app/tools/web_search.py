"""Web search behind one interface, with error handling, quotas, retries and raw-result storage.

    python -m app.tools.web_search "your question" [--max-results 5]

Providers return normalized ``SearchResult`` objects. ``WebSearch`` adds what every
caller needs: query validation, retries with backoff for transient failures, a local
monthly credit ledger (refuses a paid-quota search *before* it is sent), a fallback
provider, URL de-duplication and a JSON copy of every raw response.

Search results are untrusted web content: callers must treat titles and snippets as
data, never as instructions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.logs import log

TAVILY_URL = "https://api.tavily.com/search"
MAX_QUERY_LENGTH = 400
MAX_RETRY_WAIT_SECONDS = 30.0


# --- Errors ------------------------------------------------------------------------------


class SearchError(RuntimeError):
    """Base class. ``retryable`` errors may succeed if the same request is sent again."""

    retryable = False


class SearchConfigError(SearchError):
    """The provider cannot be used as configured (e.g. missing API key)."""


class SearchAuthError(SearchError):
    """The provider rejected the credentials."""


class SearchQuotaError(SearchError):
    """A usage limit is reached, at the provider or in the local credit ledger."""


class SearchRequestError(SearchError):
    """The request itself is invalid; another provider would not help."""


class SearchUnavailableError(SearchError):
    """Temporary failure: rate limit, timeout, network error or server error."""

    retryable = True

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# --- Data model --------------------------------------------------------------------------


class SearchResult(BaseModel):
    rank: int = Field(ge=1)
    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    published_date: str | None = None
    provider: str


class SearchResponse(BaseModel):
    query: str
    provider: str
    results: list[SearchResult]
    retrieved_at: datetime
    duration_ms: int
    credits_used: int = 0
    fallback_used: bool = False
    errors: list[str] = Field(default_factory=list)
    raw_path: str | None = None


class ProviderOutput(BaseModel):
    raw: Any
    results: list[SearchResult]
    credits: int = 0


class SearchProvider(Protocol):
    name: str
    credits_per_query: int

    def search(self, query: str, max_results: int) -> ProviderOutput: ...


# --- Providers ---------------------------------------------------------------------------


class TavilySearch:
    """Tavily Search API (basic depth: 1 credit per query)."""

    name = "tavily"
    credits_per_query = 1

    def __init__(self, api_key: str, timeout: float, client: httpx.Client | None = None):
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int) -> ProviderOutput:
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_published_date": True,
            "include_usage": True,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = self._client.post(TAVILY_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise SearchUnavailableError(f"Tavily timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SearchUnavailableError(f"Tavily network error: {exc}") from exc

        self._raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise SearchUnavailableError("Tavily returned invalid JSON") from exc

        results = [
            SearchResult(
                rank=i,
                title=(item.get("title") or "").strip(),
                url=item.get("url") or "",
                snippet=(item.get("content") or "").strip(),
                score=item.get("score"),
                published_date=item.get("published_date"),
                provider=self.name,
            )
            for i, item in enumerate(data.get("results") or [], start=1)
        ]
        credits = (data.get("usage") or {}).get("credits", self.credits_per_query)
        return ProviderOutput(raw=data, results=results, credits=int(credits))

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = response.text[:200]
        if status == 401:
            raise SearchAuthError("Tavily rejected the API key (401)")
        if status in (432, 433):
            raise SearchQuotaError(f"Tavily usage limit exceeded ({status})")
        if status in (400, 422):
            raise SearchRequestError(f"Tavily rejected the request ({status}): {detail}")
        if status in (429, 434) or status >= 500:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else None
            raise SearchUnavailableError(f"Tavily temporarily unavailable ({status})", wait)
        raise SearchRequestError(f"Tavily returned HTTP {status}: {detail}")


class DuckDuckGoSearch:
    """DuckDuckGo through the ``ddgs`` library: no key, no quota, unofficial and less stable."""

    name = "duckduckgo"
    credits_per_query = 0

    def __init__(self, timeout: float, client_factory: Callable[..., Any] | None = None):
        self._timeout = max(1, int(timeout))
        self._client_factory = client_factory

    def search(self, query: str, max_results: int) -> ProviderOutput:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        factory = self._client_factory or DDGS
        try:
            items = factory(timeout=self._timeout).text(query, max_results=max_results)
        except (RatelimitException, TimeoutException) as exc:
            raise SearchUnavailableError(f"DuckDuckGo temporarily unavailable: {exc}") from exc
        except DDGSException as exc:
            raise SearchUnavailableError(f"DuckDuckGo error: {exc}") from exc
        if not items:
            # ddgs also reports "no results" when its backends are blocked: observed to be
            # intermittent for ordinary queries, so it is retried rather than trusted.
            raise SearchUnavailableError("DuckDuckGo returned no results (often a temporary block)")

        results = [
            SearchResult(
                rank=i,
                title=(item.get("title") or "").strip(),
                url=item.get("href") or item.get("url") or "",
                snippet=(item.get("body") or "").strip(),
                provider=self.name,
            )
            for i, item in enumerate(items or [], start=1)
        ]
        return ProviderOutput(raw=items, results=results, credits=0)


class UnconfiguredProvider:
    """Stands in for a provider that cannot run, so the fallback logic stays uniform."""

    credits_per_query = 0

    def __init__(self, name: str, reason: str):
        self.name = name
        self._reason = reason

    def search(self, query: str, max_results: int) -> ProviderOutput:
        raise SearchConfigError(self._reason)


# --- Local credit ledger -----------------------------------------------------------------


class CreditLedger:
    """Credits spent per provider and calendar month, stored in a small JSON file."""

    def __init__(self, path: Path, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._path = path
        self._clock = clock

    def _month(self) -> str:
        return self._clock().strftime("%Y-%m")

    def _load(self) -> dict[str, dict[str, int]]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def used(self, provider: str) -> int:
        return int(self._load().get(self._month(), {}).get(provider, 0))

    def check(self, provider: str, cost: int, limit: int) -> None:
        if cost and self.used(provider) + cost > limit:
            raise SearchQuotaError(
                f"local {provider} limit reached: {self.used(provider)}/{limit} credits "
                f"used in {self._month()}"
            )

    def record(self, provider: str, credits: int) -> None:
        if not credits:
            return
        data = self._load()
        month = data.setdefault(self._month(), {})
        month[provider] = int(month.get(provider, 0)) + credits
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- Service -----------------------------------------------------------------------------


def normalize_url(url: str) -> str | None:
    """Canonical form used for de-duplication; None for non-web URLs."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, parts.query, ""))


def _slug(text: str, length: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:length] or "query"


class WebSearch:
    def __init__(
        self,
        primary: SearchProvider,
        fallback: SearchProvider | None,
        ledger: CreditLedger,
        monthly_credit_limit: int,
        raw_dir: Path,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.primary = primary
        self.fallback = fallback
        self.ledger = ledger
        self.monthly_credit_limit = monthly_credit_limit
        self.raw_dir = raw_dir
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    def search(self, query: str, max_results: int = 5, research_id: str | None = None) -> SearchResponse:
        query = " ".join(query.split())
        if not query:
            raise ValueError("the search query is empty")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"the search query exceeds {MAX_QUERY_LENGTH} characters")

        started = time.perf_counter()
        errors: list[str] = []
        provider, fallback_used = self.primary, False
        try:
            output = self._run(self.primary, query, max_results)
        except SearchRequestError:
            raise  # an invalid request would fail on any provider
        except SearchError as exc:
            errors.append(f"{self.primary.name}: {exc}")
            if self.fallback is None:
                raise
            provider, fallback_used = self.fallback, True
            try:
                output = self._run(self.fallback, query, max_results)
            except SearchError as fallback_exc:
                errors.append(f"{self.fallback.name}: {fallback_exc}")
                raise SearchUnavailableError("all search providers failed: " + "; ".join(errors)) from fallback_exc

        response = SearchResponse(
            query=query,
            provider=provider.name,
            results=self._deduplicate(output.results),
            retrieved_at=self._clock(),
            duration_ms=int((time.perf_counter() - started) * 1000),
            credits_used=output.credits,
            fallback_used=fallback_used,
            errors=errors,
        )
        response.raw_path = str(self._save(response, output.raw, research_id))
        log("web_search", research_id=research_id, provider=provider.name, results=len(response.results),
            credits=output.credits, fallback=fallback_used, ms=response.duration_ms, errors=errors)
        return response

    def _run(self, provider: SearchProvider, query: str, max_results: int) -> ProviderOutput:
        self.ledger.check(provider.name, provider.credits_per_query, self.monthly_credit_limit)
        for attempt in range(self.max_retries + 1):
            try:
                output = provider.search(query, max_results)
            except SearchUnavailableError as exc:
                if attempt == self.max_retries:
                    raise
                wait = exc.retry_after if exc.retry_after is not None else 2.0**attempt
                self._sleep(min(wait, MAX_RETRY_WAIT_SECONDS))
                continue
            self.ledger.record(provider.name, output.credits)
            return output
        raise AssertionError("unreachable")

    @staticmethod
    def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            key = normalize_url(result.url)
            if key is None or key in seen:
                continue
            seen.add(key)
            unique.append(result.model_copy(update={"rank": len(unique) + 1}))
        return unique

    def _save(self, response: SearchResponse, raw: Any, research_id: str | None) -> Path:
        folder = self.raw_dir / (research_id or "adhoc")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = response.retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"{stamp}_{response.provider}_{_slug(response.query)}.json"
        payload = {"response": response.model_dump(mode="json", exclude={"raw_path"}), "raw": raw}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def build_web_search(settings: Settings) -> WebSearch:
    """Wire providers from the configuration."""
    duckduckgo = DuckDuckGoSearch(settings.http_timeout_seconds)
    if settings.search_provider == "tavily":
        if settings.tavily_api_key is None:
            primary: SearchProvider = UnconfiguredProvider("tavily", "TAVILY_API_KEY is not set")
        else:
            primary = TavilySearch(settings.tavily_api_key.get_secret_value(),
                                   settings.http_timeout_seconds)
        fallback = duckduckgo if settings.search_fallback == "duckduckgo" else None
    else:
        primary, fallback = duckduckgo, None
    return WebSearch(
        primary=primary,
        fallback=fallback,
        ledger=CreditLedger(settings.search_dir / "credit_usage.json"),
        monthly_credit_limit=settings.tavily_monthly_credit_limit,
        raw_dir=settings.search_dir,
        max_retries=settings.search_max_retries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one web search and print the results.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    search = build_web_search(settings)
    try:
        response = search.search(args.query, args.max_results or settings.max_results_per_query)
    except (SearchError, ValueError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    label = f"{response.provider} (fallback)" if response.fallback_used else response.provider
    print(f"{len(response.results)} results from {label} in {response.duration_ms} ms, "
          f"{response.credits_used} credit(s)")
    for error in response.errors:
        print(f"  note: {error}")
    for r in response.results:
        print(f"\n{r.rank}. {r.title}\n   {r.url}\n   {r.snippet[:200]}")
    used = search.ledger.used("tavily")
    print(f"\nTavily credits used this month: {used}/{settings.tavily_monthly_credit_limit}")
    print(f"Raw response saved to {Path(response.raw_path).relative_to(settings.storage_dir.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
