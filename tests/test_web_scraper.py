import httpx
import pytest

from app.tools.web_scraper import Fetcher, FetchStatus, detect_kind

PUBLIC = {"site.org": ["93.184.215.14"], "other.org": ["93.184.215.15"], "intranet.org": ["10.0.0.7"]}

HTML = b"<html><body><p>hello</p></body></html>"


def make_fetcher(routes, max_bytes=10_000, max_redirects=3, robots=True):
    """routes: {(host, path): httpx.Response or callable}; unknown paths answer 404."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        route = routes.get((request.url.host, request.url.path))
        if route is None:
            return httpx.Response(404)
        return route(request) if callable(route) else route

    fetcher = Fetcher(timeout=5, max_bytes=max_bytes, max_redirects=max_redirects, user_agent="TestBot/1.0",
                      respect_robots=robots, client=httpx.Client(transport=httpx.MockTransport(handler)),
                      resolver=lambda host: PUBLIC.get(host, []))
    fetcher.calls = calls
    return fetcher


def html_response(body=HTML, **headers):
    return httpx.Response(200, content=body, headers={"Content-Type": "text/html; charset=utf-8", **headers})


def test_html_page_is_fetched():
    result = make_fetcher({("site.org", "/a"): html_response()}).fetch("https://site.org/a")
    assert result.status == FetchStatus.OK and result.kind == "html" and result.content == HTML
    assert result.http_status == 200


def test_pdf_is_detected_by_type_or_signature():
    pdf = b"%PDF-1.7 fake"
    fetcher = make_fetcher({
        ("site.org", "/doc"): httpx.Response(200, content=pdf, headers={"Content-Type": "application/pdf"}),
        ("site.org", "/download"): httpx.Response(200, content=pdf, headers={"Content-Type": "application/octet-stream"}),
    })
    assert fetcher.fetch("https://site.org/doc").kind == "pdf"
    assert fetcher.fetch("https://site.org/download").kind == "pdf"


def test_unsupported_type_is_reported():
    fetcher = make_fetcher({("site.org", "/img"): httpx.Response(200, content=b"\x89PNG", headers={"Content-Type": "image/png"})})
    result = fetcher.fetch("https://site.org/img")
    assert result.status == FetchStatus.UNSUPPORTED_TYPE and result.content is None


def test_declared_size_over_limit_is_refused_without_reading():
    big = httpx.Response(200, content=b"x" * 50, headers={"Content-Type": "text/html", "Content-Length": "999999"})
    result = make_fetcher({("site.org", "/big"): big}).fetch("https://site.org/big")
    assert result.status == FetchStatus.TOO_LARGE


def test_streamed_size_over_limit_is_cut():
    def stream(request):
        return httpx.Response(200, headers={"Content-Type": "text/html"},
                              stream=httpx.ByteStream(b"x" * 20_000))
    result = make_fetcher({("site.org", "/big"): stream}, max_bytes=10_000).fetch("https://site.org/big")
    assert result.status == FetchStatus.TOO_LARGE


def test_http_errors_are_reported_not_raised():
    result = make_fetcher({("site.org", "/gone"): httpx.Response(410)}).fetch("https://site.org/gone")
    assert result.status == FetchStatus.HTTP_ERROR and result.http_status == 410


def test_timeout_is_reported():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    result = make_fetcher({("site.org", "/slow"): slow}).fetch("https://site.org/slow")
    assert result.status == FetchStatus.TIMEOUT


def test_redirects_are_followed_and_recorded():
    fetcher = make_fetcher({
        ("site.org", "/old"): httpx.Response(301, headers={"Location": "/new"}),
        ("site.org", "/new"): httpx.Response(302, headers={"Location": "https://other.org/final"}),
        ("other.org", "/final"): html_response(),
    })
    result = fetcher.fetch("https://site.org/old")
    assert result.status == FetchStatus.OK
    assert result.url == "https://site.org/old" and result.final_url == "https://other.org/final"


def test_redirect_to_private_address_is_blocked():
    fetcher = make_fetcher({
        ("site.org", "/trap"): httpx.Response(302, headers={"Location": "http://intranet.org/admin"}),
        ("intranet.org", "/admin"): html_response(b"secret"),
    })
    result = fetcher.fetch("https://site.org/trap")
    assert result.status == FetchStatus.UNSAFE_URL
    assert not any("intranet.org" in c for c in fetcher.calls), "the private host must never be contacted"


def test_redirect_loop_stops():
    fetcher = make_fetcher({("site.org", "/loop"): httpx.Response(302, headers={"Location": "/loop"})},
                           max_redirects=3)
    assert fetcher.fetch("https://site.org/loop").status == FetchStatus.TOO_MANY_REDIRECTS


def test_unsafe_initial_url_is_never_requested():
    fetcher = make_fetcher({})
    assert fetcher.fetch("http://127.0.0.1:8000/").status == FetchStatus.UNSAFE_URL
    assert fetcher.calls == []


def test_robots_txt_is_respected_and_cached():
    robots = httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    fetcher = make_fetcher({
        ("site.org", "/robots.txt"): robots,
        ("site.org", "/private/page"): html_response(),
        ("site.org", "/public"): html_response(),
    })
    assert fetcher.fetch("https://site.org/private/page").status == FetchStatus.ROBOTS_DISALLOWED
    assert fetcher.fetch("https://site.org/public").status == FetchStatus.OK
    assert sum(c.endswith("/robots.txt") for c in fetcher.calls) == 1


def test_missing_robots_txt_allows_everything():
    result = make_fetcher({("site.org", "/a"): html_response()}).fetch("https://site.org/a")
    assert result.status == FetchStatus.OK


def test_robots_can_be_disabled():
    robots = httpx.Response(200, text="User-agent: *\nDisallow: /\n")
    fetcher = make_fetcher({("site.org", "/robots.txt"): robots, ("site.org", "/a"): html_response()}, robots=False)
    assert fetcher.fetch("https://site.org/a").status == FetchStatus.OK


@pytest.mark.parametrize(("content_type", "url", "head", "kind"), [
    ("text/html; charset=utf-8", "https://x.org/", b"<html", "html"),
    ("application/xhtml+xml", "https://x.org/", b"<?xml", "html"),
    ("text/plain", "https://x.org/a.txt", b"abc", "text"),
    (None, "https://x.org/paper.pdf", b"", "pdf"),
    ("application/json", "https://x.org/api", b"{", None),
])
def test_detect_kind(content_type, url, head, kind):
    assert detect_kind(content_type, url, head) == kind
