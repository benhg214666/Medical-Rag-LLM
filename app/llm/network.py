"""Local-only network policy for the configured inference endpoint."""

import ipaddress
from urllib.parse import urlparse

_ALLOWED_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


def _is_allowed_private_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return any(address in network for network in _ALLOWED_PRIVATE_NETWORKS)


def validate_local_llm_base_url(
    base_url: str, *, allow_private_network: bool = False
) -> str:
    """Validate loopback by default, with explicit opt-in for private IPs."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must not contain a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("LLM_BASE_URL contains an invalid port") from exc

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost":
        return base_url.rstrip("/")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise ValueError(
            "LLM_BASE_URL must use loopback; private-network hosts require "
            "an explicit IP address and LLM_ALLOW_PRIVATE_NETWORK=true"
        ) from exc

    if address.is_loopback:
        return base_url.rstrip("/")
    if allow_private_network and _is_allowed_private_address(address):
        return base_url.rstrip("/")
    raise ValueError(
        "LLM_BASE_URL must use loopback unless LLM_ALLOW_PRIVATE_NETWORK=true "
        "for a private IP address"
    )
