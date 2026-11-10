"""Safe download of web pages and PDFs.

Every hop is checked before it is requested: the initial URL and each redirect target
go through ``validate_url`` (no private or local addresses) and robots.txt, and the
connection itself only reaches a checked public address (``SafeTransport``, which defeats
DNS rebinding). Requests to one site are spaced out (a minimum delay, or the site's
robots.txt Crawl-delay, capped). Bodies are streamed and the download stops as soon as it
exceeds the size limit. Expected failures are returned as a status, never raised, so one bad
source does not stop a research.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

from app.tools.url_safety import (
    Resolver,
    SafeTransport,
    UnsafeURLError,
    system_resolver,
    validate_url,
)

ContentKind = Literal["html", "pdf", "text"]
HTML_TYPES = {"text/html", "application/xhtml+xml"}
REDIRECT_CODES = {301, 302, 303, 307, 308}


class FetchStatus(StrEnum):
    OK = "ok"
    UNSAFE_URL = "unsafe_url"
    ROBOTS_DISALLOWED = "robots_disallowed"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    TOO_MANY_REDIRECTS = "too_many_redirects"


class FetchResult(BaseModel):
    url: str
    final_url: str
    status: FetchStatus
    http_status: int | None = None
    content_type: str | None = None
    kind: ContentKind | None = None
    content: bytes | None = Field(default=None, repr=False)
    error: str | None = None
    fetched_at: datetime


def detect_kind(content_type: str | None, url: str, head: bytes) -> ContentKind | None:
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media == "application/pdf" or head.startswith(b"%PDF"):
        return "pdf"
    if media in HTML_TYPES:
        return "html"
    if media == "text/plain":
        return "text"
    if media in ("", "application/octet-stream") and urlsplit(url).path.lower().endswith(".pdf"):
        return "pdf"
    return None


class RobotsChecker:
    """robots.txt rules per site, fetched once and cached.

    If robots.txt is missing (4xx) everything is allowed; if it cannot be read (timeout,
    5xx) the page is allowed too, so a flaky robots.txt does not block research.
    """

    def __init__(self, client: httpx.Client, user_agent: str, resolver: Resolver):
        self._client, self._user_agent, self._resolver = client, user_agent, resolver
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}

    def _parser(self, url: str) -> robotparser.RobotFileParser | None:
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in self._cache:
            self._cache[site] = self._load(site)
        return self._cache[site]

    def allowed(self, url: str) -> bool:
        parser = self._parser(url)
        return True if parser is None else parser.can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float:
        parser = self._parser(url)
        delay = parser.crawl_delay(self._user_agent) if parser is not None else None
        try:
            return max(0.0, float(delay or 0))
        except (TypeError, ValueError):
            return 0.0

    def _load(self, site: str) -> robotparser.RobotFileParser | None:
        robots_url = f"{site}/robots.txt"
        try:
            validate_url(robots_url, self._resolver)
            response = self._client.get(robots_url, headers={"User-Agent": self._user_agent})
        except (UnsafeURLError, httpx.HTTPError):
            return None
        if response.status_code >= 400:
            return None
        parser = robotparser.RobotFileParser()
        parser.parse(response.text[:500_000].splitlines())
        return parser


class HostThrottle:
    """Spaces out requests to the same host: at least ``delay(host)`` seconds between them."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._clock, self._sleep = clock, sleep
        self._last: dict[str, float] = {}

    def wait(self, host: str, delay: float) -> float:
        """Sleep if needed before a request to ``host``; returns the seconds slept."""
        waited = 0.0
        if host in self._last:
            waited = max(0.0, self._last[host] + delay - self._clock())
            if waited:
                self._sleep(waited)
        self._last[host] = self._clock()
        return waited


class Fetcher:
    def __init__(
        self,
        timeout: float,
        max_bytes: int,
        max_redirects: int,
        user_agent: str,
        respect_robots: bool = True,
        client: httpx.Client | None = None,
        resolver: Resolver = system_resolver,
        min_delay_per_host: float = 1.0,
        max_crawl_delay: float = 10.0,
        throttle: HostThrottle | None = None,
    ):
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False,
                                              transport=SafeTransport(resolver))
        self._min_delay, self._max_crawl_delay = min_delay_per_host, max_crawl_delay
        self._throttle = throttle or HostThrottle()
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._user_agent = user_agent
        self._resolver = resolver
        self._robots = RobotsChecker(self._client, user_agent, resolver) if respect_robots else None

    def fetch(self, url: str) -> FetchResult:
        current = url

        def result(status: FetchStatus, **fields) -> FetchResult:
            return FetchResult(url=url, final_url=current, status=status,
                               fetched_at=datetime.now(UTC), **fields)

        for _ in range(self._max_redirects + 1):
            try:
                validate_url(current, self._resolver)
            except UnsafeURLError as exc:
                return result(FetchStatus.UNSAFE_URL, error=str(exc))
            if self._robots is not None and not self._robots.allowed(current):
                return result(FetchStatus.ROBOTS_DISALLOWED, error="disallowed by robots.txt")
            self._throttle.wait(urlsplit(current).hostname or "", self._delay(current))
            try:
                outcome = self._get(current)
            except httpx.TimeoutException as exc:
                return result(FetchStatus.TIMEOUT, error=f"timed out: {exc}")
            except httpx.HTTPError as exc:
                status = FetchStatus.UNSAFE_URL if str(exc).startswith("blocked:") else FetchStatus.NETWORK_ERROR
                return result(status, error=str(exc))
            if isinstance(outcome, str):  # redirect target
                current = urljoin(current, outcome)
                continue
            status, fields = outcome
            return result(status, **fields)
        return result(FetchStatus.TOO_MANY_REDIRECTS,
                      error=f"more than {self._max_redirects} redirects")

    def _delay(self, url: str) -> float:
        crawl_delay = self._robots.crawl_delay(url) if self._robots is not None else 0.0
        return max(self._min_delay, min(crawl_delay, self._max_crawl_delay))

    def _get(self, url: str) -> str | tuple[FetchStatus, dict]:
        headers = {"User-Agent": self._user_agent, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5"}
        with self._client.stream("GET", url, headers=headers) as response:
            code = response.status_code
            if code in REDIRECT_CODES and response.headers.get("Location"):
                return response.headers["Location"]
            content_type = response.headers.get("Content-Type")
            base = {"http_status": code, "content_type": content_type}
            if code >= 400:
                return FetchStatus.HTTP_ERROR, {**base, "error": f"HTTP {code}"}
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > self._max_bytes:
                return FetchStatus.TOO_LARGE, {**base, "error": f"declared size {declared} bytes"}

            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > self._max_bytes:
                    return FetchStatus.TOO_LARGE, {**base, "error": f"more than {self._max_bytes} bytes"}

        kind = detect_kind(content_type, url, bytes(body[:8]))
        if kind is None:
            return FetchStatus.UNSUPPORTED_TYPE, {**base, "error": f"unsupported type {content_type}"}
        return FetchStatus.OK, {**base, "kind": kind, "content": bytes(body)}
