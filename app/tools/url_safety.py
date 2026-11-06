"""Checks run on every URL before the application downloads it.

A web page or a search result can point anywhere, including the user's own machine or
local network (server-side request forgery, SSRF). Only http(s) URLs whose host resolves
exclusively to public addresses are allowed.

Known limit, addressed in the security phase: the address is resolved here and again by
the HTTP client, so a hostile DNS server could answer differently the second time
(DNS rebinding).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlsplit

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

    try:
        addresses = [str(ipaddress.ip_address(host))]  # literal IP in the URL
    except ValueError:
        addresses = resolver(host)
    if not addresses:
        raise UnsafeURLError(f"host {host!r} has no address")
    blocked = [a for a in addresses if not is_public_ip(a)]
    if blocked:
        raise UnsafeURLError(f"host {host!r} resolves to a non-public address ({blocked[0]})")
    return url.strip()
