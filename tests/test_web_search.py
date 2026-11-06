import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.tools.web_search import (
    CreditLedger,
    DuckDuckGoSearch,
    ProviderOutput,
    SearchAuthError,
    SearchConfigError,
    SearchQuotaError,
    SearchRequestError,
    SearchResult,
    SearchUnavailableError,
    TavilySearch,
    UnconfiguredProvider,
    WebSearch,
    build_web_search,
    normalize_url,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)

TAVILY_OK = {
    "query": "q",
    "results": [
        {"title": " Page A ", "url": "https://a.org/x", "content": "about A", "score": 0.9,
         "published_date": "2026-01-02"},
        {"title": "Page B", "url": "https://b.org/", "content": "about B", "score": 0.7},
    ],
    "response_time": 0.8,
    "usage": {"credits": 1},
}


def tavily_with(handler) -> TavilySearch:
    return TavilySearch("tvly-test", timeout=5, client=httpx.Client(transport=httpx.MockTransport(handler)))


def respond(status: int, body=None, headers=None):
    return lambda request: httpx.Response(status, json=body or {}, headers=headers)


# --- Tavily provider ---------------------------------------------------------------------


def test_tavily_request_and_normalized_results():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=TAVILY_OK)

    out = tavily_with(handler).search("climate policy", 5)
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["query"] == "climate policy" and seen["body"]["max_results"] == 5
    assert seen["body"]["search_depth"] == "basic"  # 1 credit, never "advanced" (2 credits)
    assert [r.title for r in out.results] == ["Page A", "Page B"]
    assert out.results[0].snippet == "about A" and out.results[0].published_date == "2026-01-02"
    assert out.credits == 1 and out.raw == TAVILY_OK


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, SearchAuthError), (432, SearchQuotaError), (433, SearchQuotaError),
     (400, SearchRequestError), (422, SearchRequestError), (429, SearchUnavailableError),
     (434, SearchUnavailableError), (500, SearchUnavailableError), (503, SearchUnavailableError)],
)
def test_tavily_status_codes_map_to_typed_errors(status, error):
    with pytest.raises(error):
        tavily_with(respond(status)).search("q", 5)


def test_tavily_reads_retry_after():
    with pytest.raises(SearchUnavailableError) as info:
        tavily_with(respond(429, headers={"Retry-After": "7"})).search("q", 5)
    assert info.value.retry_after == 7


def test_tavily_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SearchUnavailableError):
        tavily_with(handler).search("q", 5)


def test_api_key_never_appears_in_errors():
    with pytest.raises(SearchAuthError) as info:
        tavily_with(respond(401)).search("q", 5)
    assert "tvly-test" not in str(info.value)


# --- DuckDuckGo provider -----------------------------------------------------------------


class FakeDDGS:
    def __init__(self, items=None, error=None, **kwargs):
        self.items, self.error, self.kwargs = items, error, kwargs

    def text(self, query, max_results):
        if self.error:
            raise self.error
        return self.items[:max_results]


def test_duckduckgo_results_are_normalized():
    items = [{"title": "T", "href": "https://d.org/1", "body": "snippet"}]
    out = DuckDuckGoSearch(5, client_factory=lambda **kw: FakeDDGS(items, **kw)).search("q", 3)
    assert out.results[0].url == "https://d.org/1" and out.results[0].snippet == "snippet"
    assert out.credits == 0


@pytest.mark.parametrize("fake", [
    lambda **kw: FakeDDGS(items=[]),
    lambda **kw: FakeDDGS(error=__import__("ddgs.exceptions").exceptions.DDGSException("No results found.")),
])
def test_duckduckgo_empty_answer_is_retryable(fake):
    # ddgs reports "no results" when blocked, intermittently for ordinary queries
    with pytest.raises(SearchUnavailableError):
        DuckDuckGoSearch(5, client_factory=fake).search("q", 3)


def test_duckduckgo_rate_limit_is_retryable():
    from ddgs.exceptions import RatelimitException

    ddg = DuckDuckGoSearch(5, client_factory=lambda **kw: FakeDDGS(error=RatelimitException("202")))
    with pytest.raises(SearchUnavailableError):
        ddg.search("q", 3)


# --- Service: retries, fallback, quota, storage ------------------------------------------


class ScriptedProvider:
    """Returns or raises the scripted outcomes in order."""

    def __init__(self, name, outcomes, credits_per_query=1):
        self.name, self.outcomes, self.credits_per_query = name, list(outcomes), credits_per_query
        self.calls = 0

    def search(self, query, max_results):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def output(*urls, credits=1, provider="p"):
    results = [SearchResult(rank=i, title=u, url=u, provider=provider) for i, u in enumerate(urls, 1)]
    return ProviderOutput(raw={"urls": list(urls)}, results=results, credits=credits)


@pytest.fixture
def make_service(tmp_path):
    def make(primary, fallback=None, limit=1000, retries=2):
        waits = []
        service = WebSearch(primary, fallback, CreditLedger(tmp_path / "ledger.json", clock=lambda: NOW),
                            monthly_credit_limit=limit, raw_dir=tmp_path / "search",
                            max_retries=retries, sleep=waits.append, clock=lambda: NOW)
        service.waits = waits
        return service
    return make


def test_successful_search_records_credits_and_saves_raw_response(make_service, tmp_path):
    service = make_service(ScriptedProvider("tavily", [output("https://a.org", "https://b.org")]))
    response = service.search("  what   is RAG? ", research_id="r1")
    assert response.query == "what is RAG?" and response.provider == "tavily"
    assert [r.url for r in response.results] == ["https://a.org", "https://b.org"]
    assert service.ledger.used("tavily") == 1
    saved = json.loads(Path(response.raw_path).read_text(encoding="utf-8"))
    assert saved["raw"] == {"urls": ["https://a.org", "https://b.org"]}
    assert saved["response"]["query"] == "what is RAG?"
    assert "r1" in response.raw_path


def test_transient_errors_are_retried_with_backoff(make_service):
    provider = ScriptedProvider("tavily", [SearchUnavailableError("503"),
                                           SearchUnavailableError("429", retry_after=5),
                                           output("https://a.org")])
    service = make_service(provider)
    assert service.search("q").results
    assert provider.calls == 3 and service.waits == [1.0, 5]


def test_retry_wait_is_capped(make_service):
    provider = ScriptedProvider("tavily", [SearchUnavailableError("429", retry_after=3600), output("https://a.org")])
    service = make_service(provider)
    service.search("q")
    assert service.waits == [30.0]


def test_auth_error_is_not_retried_and_falls_back(make_service):
    primary = ScriptedProvider("tavily", [SearchAuthError("bad key")])
    fallback = ScriptedProvider("duckduckgo", [output("https://d.org", credits=0)], credits_per_query=0)
    response = make_service(primary, fallback).search("q")
    assert primary.calls == 1 and response.fallback_used and response.provider == "duckduckgo"
    assert response.errors == ["tavily: bad key"]


def test_exhausted_retries_fall_back(make_service):
    primary = ScriptedProvider("tavily", [SearchUnavailableError("down")] * 3)
    fallback = ScriptedProvider("duckduckgo", [output("https://d.org", credits=0)], credits_per_query=0)
    response = make_service(primary, fallback).search("q")
    assert primary.calls == 3 and response.provider == "duckduckgo"


def test_invalid_request_does_not_fall_back(make_service):
    primary = ScriptedProvider("tavily", [SearchRequestError("bad query")])
    fallback = ScriptedProvider("duckduckgo", [output("https://d.org")])
    with pytest.raises(SearchRequestError):
        make_service(primary, fallback).search("q")
    assert fallback.calls == 0


def test_error_without_fallback_is_raised(make_service):
    with pytest.raises(SearchAuthError):
        make_service(ScriptedProvider("tavily", [SearchAuthError("bad key")])).search("q")


def test_both_providers_failing_gives_one_clear_error(make_service):
    primary = ScriptedProvider("tavily", [SearchAuthError("bad key")])
    fallback = ScriptedProvider("duckduckgo", [SearchUnavailableError("blocked")] * 3, credits_per_query=0)
    with pytest.raises(SearchUnavailableError, match="all search providers failed"):
        make_service(primary, fallback).search("q")


def test_local_quota_blocks_before_sending_and_falls_back(make_service):
    primary = ScriptedProvider("tavily", [output("https://a.org")] * 3)
    fallback = ScriptedProvider("duckduckgo", [output("https://d.org", credits=0)], credits_per_query=0)
    service = make_service(primary, fallback, limit=2)
    service.search("q1")
    service.search("q2")
    response = service.search("q3")
    assert primary.calls == 2, "the third Tavily request must not be sent"
    assert response.provider == "duckduckgo" and "limit reached" in response.errors[0]


def test_ledger_resets_each_month(tmp_path):
    month = {"now": datetime(2026, 11, 30, tzinfo=UTC)}
    ledger = CreditLedger(tmp_path / "l.json", clock=lambda: month["now"])
    ledger.record("tavily", 5)
    assert ledger.used("tavily") == 5
    month["now"] = datetime(2026, 12, 1, tzinfo=UTC)
    assert ledger.used("tavily") == 0


def test_duplicate_and_non_web_urls_are_removed(make_service):
    provider = ScriptedProvider("tavily", [output("https://A.org/page/", "https://a.org/page#top",
                                                  "javascript:alert(1)", "https://b.org")])
    response = make_service(provider).search("q")
    assert [r.url for r in response.results] == ["https://A.org/page/", "https://b.org"]
    assert [r.rank for r in response.results] == [1, 2]


@pytest.mark.parametrize("query", ["", "   ", "x" * 401])
def test_invalid_queries_are_rejected_before_any_call(make_service, query):
    provider = ScriptedProvider("tavily", [])
    with pytest.raises(ValueError):
        make_service(provider).search(query)
    assert provider.calls == 0


def test_normalize_url():
    assert normalize_url("HTTPS://Example.ORG/a/?q=1#frag") == "https://example.org/a?q=1"
    assert normalize_url("ftp://example.org") is None
    assert normalize_url("not a url") is None


# --- Wiring from settings ----------------------------------------------------------------


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_missing_tavily_key_uses_fallback(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    service = build_web_search(settings(storage_dir=tmp_path))
    assert isinstance(service.primary, UnconfiguredProvider)
    assert isinstance(service.fallback, DuckDuckGoSearch)
    with pytest.raises(SearchConfigError, match="TAVILY_API_KEY"):
        service.primary.search("q", 1)


def test_fallback_can_be_disabled(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    service = build_web_search(settings(storage_dir=tmp_path, tavily_api_key="k", search_fallback="none"))
    assert isinstance(service.primary, TavilySearch) and service.fallback is None
