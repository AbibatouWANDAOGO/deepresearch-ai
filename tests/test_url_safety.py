import pytest

from app.tools.url_safety import UnsafeURLError, is_public_ip, validate_url

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
