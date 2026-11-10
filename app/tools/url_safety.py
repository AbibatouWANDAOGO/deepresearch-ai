"""Checks run on every URL before the application downloads it.

A web page or a search result can point anywhere, including the user's own machine or
local network (server-side request forgery, SSRF). Only http(s) URLs whose host resolves
exclusively to public addresses are allowed.

``validate_url`` rejects a URL early (scheme, credentials, local names, addresses). The
connection itself goes through ``PinnedBackend``: the host is resolved once, every address is
checked, and the socket is opened to a checked address. A hostile DNS server that answers
with a public address for the check and a private one for the connection (DNS rebinding)
therefore gets nowhere: there is no second resolution. TLS still verifies the certificate
against the host name.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

import httpcore
import httpx

Resolver = Callable[[str], list[str]]

BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


class UnsafeURLError(ValueError):
    """The URL must not be fetched."""


def system_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"cannot resolve host {host!r}") from exc
    return sorted({info[4][0] for info in infos})


def is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone index
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def public_addresses(host: str, resolver: Resolver = system_resolver) -> list[str]:
    """Addresses of ``host`` (a name or a literal IP); raises if any of them is not public."""
    host = host.strip("[]").rstrip(".").lower()
    try:
        addresses = [str(ipaddress.ip_address(host))]  # literal IP
    except ValueError:
        addresses = resolver(host)
    if not addresses:
        raise UnsafeURLError(f"host {host!r} has no address")
    blocked = [a for a in addresses if not is_public_ip(a)]
    if blocked:
        raise UnsafeURLError(f"host {host!r} resolves to a non-public address ({blocked[0]})")
    return addresses


class PinnedBackend(httpcore.NetworkBackend):
    """Opens TCP connections only to public addresses, resolved and checked at connect time."""

    def __init__(self, resolver: Resolver = system_resolver, backend: httpcore.NetworkBackend | None = None):
        self._resolver = resolver
        self._backend = backend or httpcore.SyncBackend()

    def connect_tcp(self, host: str, port: int, timeout: float | None = None, local_address: str | None = None,
                    socket_options: Iterable | None = None) -> httpcore.NetworkStream:
        try:
            addresses = public_addresses(host, self._resolver)
        except UnsafeURLError as exc:
            raise httpcore.ConnectError(f"blocked: {exc}") from exc
        error: Exception | None = None
        for address in addresses:
            try:
                return self._backend.connect_tcp(address, port, timeout=timeout, local_address=local_address,
                                                 socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                error = exc
        raise error or httpcore.ConnectError(f"cannot connect to {host!r}")

    def connect_unix_socket(self, *args, **kwargs) -> httpcore.NetworkStream:
        raise httpcore.ConnectError("unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class SafeTransport(httpx.HTTPTransport):
    """httpx transport whose connections go through ``PinnedBackend``, without proxies from
    the environment (a proxy would make the check apply to the proxy, not to the site)."""

    def __init__(self, resolver: Resolver = system_resolver, backend: httpcore.NetworkBackend | None = None):
        super().__init__(trust_env=False)
        self._pool = httpcore.ConnectionPool(ssl_context=httpx.create_ssl_context(trust_env=False),
                                             network_backend=PinnedBackend(resolver, backend))


def validate_url(url: str, resolver: Resolver = system_resolver) -> str:
    """Return the URL unchanged if it is safe to fetch, otherwise raise UnsafeURLError."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"scheme {parts.scheme or '(none)'!r} is not allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError("the URL has no host")
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeURLError(f"host {host!r} is local")
    try:
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError as exc:
        raise UnsafeURLError("invalid port") from exc

    public_addresses(host, resolver)
    return url.strip()
