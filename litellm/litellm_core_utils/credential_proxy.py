import ipaddress
import re
from typing import Final
from urllib.parse import unquote, urlsplit, urlunsplit

from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

_CREDENTIAL_PROXY_KEY: Final = "litellm_internal_proxy_url"
_SUPPORTED_SCHEMES: Final = frozenset({"http", "https", "socks5", "socks5h"})
_BAD_PERCENT: Final = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SUPPORTED_PROXY_PROVIDERS: Final = frozenset({"openai", "custom_openai", "openrouter", "anthropic", "chatgpt", "azure"})


def require_proxy_provider_support(provider: str | None) -> None:
    if provider not in _SUPPORTED_PROXY_PROVIDERS:
        raise ValueError("Connection proxies are not supported for this provider transport")


def _invalid() -> ValueError:
    return ValueError("Invalid credential proxy URL; expected an absolute http, https, socks5, or socks5h URL")


def validate_proxy_url(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or value != value.strip():
        raise _invalid()
    if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in value) or "\\" in value or _BAD_PERCENT.search(value):
        raise _invalid()
    try:
        parsed: Final = urlsplit(value)
        scheme, hostname, port = parsed.scheme.lower(), parsed.hostname, parsed.port
    except (TypeError, ValueError):
        raise _invalid() from None
    if scheme not in _SUPPORTED_SCHEMES or not parsed.netloc or not hostname:
        raise _invalid()
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise _invalid()
    if port is not None and not 1 <= port <= 65535:
        raise _invalid()
    if "@" in parsed.netloc and parsed.username == "":
        raise _invalid()
    try:
        authority: Final = unquote(parsed.netloc, errors="strict")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in authority):
            raise _invalid()
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
        else:
            if any(char in hostname for char in "%/@"):
                raise _invalid()
            hostname.encode("idna")
    except (UnicodeError, ValueError):
        raise _invalid() from None
    return urlunsplit((scheme, parsed.netloc, parsed.path, "", ""))


def get_credential_proxy_url(credential_name: str | None) -> str | None:
    if not credential_name:
        return None
    credential: Final = CredentialAccessor.find_credential(credential_name)
    if credential is None:
        return None
    value: Final = credential.credential_values.get(_CREDENTIAL_PROXY_KEY)
    if value is None or value == "":
        if credential.credential_info.get("proxy_configured"):
            raise ValueError("Credential proxy configuration is unavailable")
        return None
    if not isinstance(value, str):
        raise _invalid()
    provider: Final = credential.credential_info.get("provider")
    if isinstance(provider, str):
        require_proxy_provider_support(provider)
    return validate_proxy_url(value)


def pop_request_proxy_url(kwargs: dict[str, object]) -> str | None:
    from litellm.llms.custom_httpx.http_handler import CREDENTIAL_PROXY_TRUSTED

    injected: Final = kwargs.pop("_credential_proxy_url", None)
    trusted: Final = kwargs.pop("_credential_proxy_trusted", None)
    kwargs.pop("litellm_internal_proxy_url", None)
    credential_name: Final = kwargs.get("litellm_credential_name")
    if isinstance(credential_name, str) and credential_name:
        return get_credential_proxy_url(credential_name)
    if injected is not None and trusted is CREDENTIAL_PROXY_TRUSTED:
        if not isinstance(injected, str):
            raise _invalid()
        return validate_proxy_url(injected)
    return None


__all__ = ["get_credential_proxy_url", "pop_request_proxy_url", "validate_proxy_url"]
