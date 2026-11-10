import http.server
import threading

import httpcore
import httpx
import pytest

from app.tools.url_safety import (
    PinnedBackend,
    SafeTransport,
    UnsafeURLError,
    is_public_ip,
    validate_url,
)
from app.tools.web_scraper import Fetcher, FetchStatus

DNS = {
    "example.org": ["93.184.215.14"],
    "ipv6.example.org": ["2606:2800:21f:cb07:6820:80da:af6b:8b2c"],
    "evil-rebind.example": ["127.0.0.1"],
    "mixed.example": ["93.184.215.14", "10.0.0.5"],
    "router.example": ["192.168.1.1"],
}


def resolver(host):
    return DNS.get(host, [])


@pytest.mark.parametrize("url", ["https://example.org/page", "http://example.org:8080/x?q=1",
                                 "https://ipv6.example.org/", "https://93.184.215.14/"])
def test_public_urls_are_allowed(url):
    assert validate_url(url, resolver) == url


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://example.org/file", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("https://user:pass@example.org/", "credentials"),
        ("http://localhost:8000/admin", "local"),
        ("http://LOCALHOST/", "local"),
        ("http://printer.local/", "local"),
        ("http://127.0.0.1/", "non-public"),
        ("http://[::1]/", "non-public"),
        ("http://10.1.2.3/", "non-public"),
        ("http://172.16.0.1/", "non-public"),
        ("http://192.168.0.10/", "non-public"),
        ("http://169.254.169.254/latest/meta-data/", "non-public"),  # cloud metadata endpoint
        ("http://0.0.0.0/", "non-public"),
        ("http://[::ffff:127.0.0.1]/", "non-public"),  # IPv4-mapped loopback
        ("http://evil-rebind.example/", "non-public"),  # public-looking name, local address
        ("http://mixed.example/", "non-public"),  # one private address is enough to refuse
        ("http://router.example/", "non-public"),
        ("http://unknown.example/", "no address"),
        ("https:///no-host", "no host"),
        ("http://example.org:99999/", "port"),
    ],
)
def test_unsafe_urls_are_refused(url, reason):
    with pytest.raises(UnsafeURLError, match=reason):
        validate_url(url, resolver)


def test_is_public_ip():
    assert is_public_ip("8.8.8.8")
    assert not is_public_ip("100.64.0.1")  # carrier-grade NAT
    assert not is_public_ip("224.0.0.1")  # multicast
    assert not is_public_ip("fe80::1%eth0")  # link-local with zone index


# --- Connect-time check (DNS rebinding) -----------------------------------------------------

class RecordingBackend(httpcore.NetworkBackend):
    def __init__(self):
        self.connected = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connected.append((host, port))
        raise httpcore.ConnectError("test backend: no network")

    def sleep(self, seconds):
        pass


def test_connection_goes_to_the_checked_address_not_to_the_name():
    backend = RecordingBackend()
    with pytest.raises(httpcore.ConnectError):
        PinnedBackend(resolver, backend).connect_tcp("example.org", 443)
    assert backend.connected == [("93.184.215.14", 443)]


@pytest.mark.parametrize("host", ["evil-rebind.example", "mixed.example", "127.0.0.1", "[::1]", "unknown.example"])
def test_connection_to_a_non_public_address_is_refused(host):
    backend = RecordingBackend()
    with pytest.raises(httpcore.ConnectError, match="blocked"):
        PinnedBackend(resolver, backend).connect_tcp(host, 80)
    assert backend.connected == []


def test_dns_rebinding_cannot_reach_a_local_server():
    """The name resolves to a public address for the URL check, then to 127.0.0.1."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"secret")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    answers = iter([["93.184.215.14"], ["127.0.0.1"], ["127.0.0.1"]])
    try:
        fetcher = Fetcher(timeout=5, max_bytes=10_000, max_redirects=0, user_agent="TestBot/1.0", respect_robots=False,
                          resolver=lambda host: next(answers), min_delay_per_host=0)
        result = fetcher.fetch(f"http://rebind.example:{server.server_port}/admin")
    finally:
        server.shutdown()
    assert result.status == FetchStatus.UNSAFE_URL and "non-public" in result.error
    assert hits == []


def test_safe_transport_ignores_proxy_settings(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = httpx.Client(transport=SafeTransport(resolver), trust_env=False)
    assert client._mounts == {}  # no proxy transport mounted from the environment
