import asyncio
import base64
import hashlib
import math
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import timezone
from email.utils import format_datetime, parsedate_to_datetime
from typing import Final, Protocol, TypeGuard, cast, runtime_checkable
from urllib.parse import urlencode

import httpx
from pydantic import TypeAdapter, ValidationError

_OAUTH_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_ACCESS_TOKEN_PATTERN: Final = re.compile(r"sk-ant-oat[-A-Za-z0-9._~+/]+=*")

ANTHROPIC_OAUTH_CLIENT_ID: Final = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_OAUTH_AUTHORIZE_URL: Final = "https://claude.com/cai/oauth/authorize"
ANTHROPIC_OAUTH_TOKEN_URL: Final = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_OAUTH_REDIRECT_URI: Final = "https://platform.claude.com/oauth/code/callback"
ANTHROPIC_OAUTH_REFRESH_SCOPES: Final = (
    "user:profile",
    "user:inference",
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
    "user:plugins",
)
ANTHROPIC_OAUTH_SCOPES: Final = ("org:create_api_key", *ANTHROPIC_OAUTH_REFRESH_SCOPES)
ANTHROPIC_OAUTH_TIMEOUT_SECONDS: Final = 30.0


class AsyncHTTPClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: Mapping[str, str],
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response: ...


class AnthropicOAuthError(RuntimeError):
    def __init__(self, operation: str, status_code: int = 400, retry_after: str | None = None) -> None:
        super().__init__(f"Anthropic OAuth {operation} failed")
        self.status_code = status_code
        self.retryable = status_code == 429 or status_code >= 500
        self.retry_after = retry_after if self.retryable else None

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        return {"Retry-After": self.retry_after} if self.retry_after is not None else None

    @property
    def headers(self) -> dict[str, str]:
        return dict(self.response_headers or {})


@dataclass(frozen=True, slots=True, repr=False)
class AnthropicAuthorization:
    authorization_url: str
    state: str
    code_verifier: str


@dataclass(frozen=True, slots=True, repr=False)
class AnthropicOAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str | None = None
    device_id: str = field(default_factory=lambda: secrets.token_hex(32))

    def __repr__(self) -> str:
        return f"AnthropicOAuthTokens(expires_at={self.expires_at!r}, account_id={self.account_id!r})"

    def to_json(self) -> str:
        import json

        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "AnthropicOAuthTokens":
        parsed: Final = _OAUTH_OBJECT_ADAPTER.validate_json(value)
        access_token: Final = parsed.get("access_token")
        refresh_token: Final = parsed.get("refresh_token")
        expires_at: Final = parsed.get("expires_at")
        account_id: Final = parsed.get("account_id")
        device_id: Final = parsed.get("device_id")
        if not _is_access_token(access_token):
            raise ValueError("Anthropic OAuth access token is invalid")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ValueError("Anthropic OAuth refresh token is missing")
        if (
            not isinstance(expires_at, (int, float))
            or isinstance(expires_at, bool)
            or not math.isfinite(expires_at)
            or expires_at <= 0
        ):
            raise ValueError("Anthropic OAuth token expiry is invalid")
        if device_id is not None and (
            not isinstance(device_id, str)
            or len(device_id) != 64
            or any(character not in "0123456789abcdef" for character in device_id)
        ):
            raise ValueError("Anthropic OAuth device identity is invalid")
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at),
            account_id=account_id if isinstance(account_id, str) and account_id else None,
            device_id=device_id or hashlib.sha256(f"litellm-anthropic-device:{refresh_token}".encode()).hexdigest(),
        )


class AnthropicOAuthClient:
    def __init__(self, http_client: AsyncHTTPClient | None = None) -> None:
        self._http_client = http_client

    def begin_authorization(self) -> AnthropicAuthorization:
        verifier: Final = _base64url(secrets.token_bytes(64))
        challenge: Final = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
        state: Final = secrets.token_hex(16)
        query: Final = urlencode(
            {
                "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                "response_type": "code",
                "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
                "scope": " ".join(ANTHROPIC_OAUTH_SCOPES),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "code": "true",
            }
        )
        return AnthropicAuthorization(
            authorization_url=f"{ANTHROPIC_OAUTH_AUTHORIZE_URL}?{query}",
            state=state,
            code_verifier=verifier,
        )

    async def exchange_code(self, code: str, state: str, code_verifier: str) -> AnthropicOAuthTokens:
        return await self._request_tokens(
            {
                "grant_type": "authorization_code",
                "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
                "code_verifier": code_verifier,
            },
            operation="token exchange",
        )

    async def refresh(self, previous: AnthropicOAuthTokens) -> AnthropicOAuthTokens:
        return await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                "refresh_token": previous.refresh_token,
                "scope": " ".join(ANTHROPIC_OAUTH_REFRESH_SCOPES),
            },
            operation="token refresh",
            previous=previous,
        )

    async def _request_tokens(
        self,
        body: Mapping[str, str],
        *,
        operation: str,
        previous: AnthropicOAuthTokens | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AnthropicOAuthTokens:
        try:
            async with asyncio.timeout(ANTHROPIC_OAUTH_TIMEOUT_SECONDS):
                response: Final = await self._post(body, headers)
            response.raise_for_status()
            payload: Final = _OAUTH_OBJECT_ADAPTER.validate_json(response.content)
        except httpx.HTTPStatusError as error:
            retry_after: Final = sanitize_retry_after(cast(str | None, error.response.headers.get("retry-after")))
            raise AnthropicOAuthError(operation, error.response.status_code, retry_after) from None
        except (httpx.TimeoutException, TimeoutError):
            raise AnthropicOAuthError(operation, 504) from None
        except httpx.HTTPError:
            raise AnthropicOAuthError(operation, 503) from None
        except ValueError:
            raise AnthropicOAuthError(operation) from None
        return self._parse_tokens(payload, previous, operation)

    async def _post(self, body: Mapping[str, str], headers: Mapping[str, str] | None) -> httpx.Response:
        if self._http_client is not None:
            return await self._http_client.post(ANTHROPIC_OAUTH_TOKEN_URL, json=body, headers=headers)
        transport: Final = httpx.AsyncHTTPTransport(retries=0)
        async with httpx.AsyncClient(timeout=ANTHROPIC_OAUTH_TIMEOUT_SECONDS, transport=transport) as client:
            return await client.post(ANTHROPIC_OAUTH_TOKEN_URL, json=body, headers=headers)

    @staticmethod
    def _parse_tokens(
        payload: Mapping[str, object],
        previous: AnthropicOAuthTokens | None,
        operation: str,
    ) -> AnthropicOAuthTokens:
        access_token: Final = payload.get("access_token")
        rotated_refresh_token: Final = payload.get("refresh_token")
        refresh_token: Final = (
            rotated_refresh_token
            if isinstance(rotated_refresh_token, str) and rotated_refresh_token
            else previous.refresh_token
            if previous is not None
            else None
        )
        expires_in: Final = payload.get("expires_in")
        account_id_value: Final = (
            _nested_id(payload.get("account"))
            or payload.get("account_id")
            or payload.get("user_id")
            or _nested_id(payload.get("user"))
        )
        account_id: Final = (
            account_id_value
            if isinstance(account_id_value, str) and account_id_value
            else previous.account_id
            if previous is not None
            else None
        )
        if not _is_access_token(access_token):
            raise AnthropicOAuthError(operation)
        if not isinstance(refresh_token, str) or not refresh_token:
            raise AnthropicOAuthError(operation)
        if (
            not isinstance(expires_in, (int, float))
            or isinstance(expires_in, bool)
            or not math.isfinite(expires_in)
            or expires_in <= 0
        ):
            raise AnthropicOAuthError(operation)
        return AnthropicOAuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + float(expires_in),
            account_id=account_id,
            device_id=previous.device_id if previous is not None else secrets.token_hex(32),
        )


@runtime_checkable
class _TokenRecoveryHook(Protocol):
    async def recover_rejected_token(self, credential_name: str, rejected_access_token: str) -> str | None: ...


async def recover_managed_anthropic_oauth_headers(
    *, headers: Mapping[str, object], litellm_params: Mapping[str, object]
) -> dict[str, object] | None:
    import litellm

    credential_name: Final = litellm_params.get("litellm_credential_name")
    authorization_name, authorization = next(
        ((name, value) for name, value in headers.items() if name.lower() == "authorization"), ("", None)
    )
    if not isinstance(credential_name, str) or not credential_name or not isinstance(authorization, str):
        return None
    scheme, separator, rejected_token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not rejected_token.startswith("sk-ant-oat"):
        return None
    for callback in cast(list[object], litellm.callbacks):
        if not isinstance(callback, _TokenRecoveryHook):
            continue
        try:
            if (token := await callback.recover_rejected_token(credential_name, rejected_token)) is not None:
                if not _is_access_token(token):
                    raise AnthropicOAuthError("token refresh", 502)
                return {
                    **{name: value for name, value in headers.items() if name.lower() != "authorization"},
                    authorization_name: f"Bearer {token}",
                }
        except Exception as error:  # noqa: BLE001  # credential callback failures must not expose token bundles
            raise _recovery_error(error) from None
    return None


def _recovery_error(error: Exception) -> AnthropicOAuthError:
    status: Final = cast(object, getattr(error, "status_code", 503))
    status_code: Final = (
        status if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599 else 503
    )
    response_headers: Final = cast(object, getattr(error, "response_headers", None))
    retry_after: Final = (
        next(
            (
                value
                for name, value in cast(Mapping[str, object], response_headers).items()
                if name.lower() == "retry-after"
            ),
            None,
        )
        if isinstance(response_headers, Mapping)
        else None
    )
    return AnthropicOAuthError(
        "token refresh", status_code, sanitize_retry_after(retry_after) if isinstance(retry_after, str) else None
    )


def sanitize_retry_after(value: str | None) -> str | None:
    if value is None or len(value) > 128:
        return None
    if value.isascii() and value.isdigit() and len(value) <= 10:
        return value
    try:
        parsed: Final = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return None
        return format_datetime(parsed.astimezone(timezone.utc), usegmt=True)
    except (TypeError, ValueError, OverflowError):
        return None


def _nested_id(value: object) -> str | None:
    try:
        fields: Final = _OAUTH_OBJECT_ADAPTER.validate_python(value)
    except ValidationError:
        return None
    nested_id: Final = fields.get("uuid") or fields.get("id")
    return nested_id if isinstance(nested_id, str) and nested_id else None


def _is_access_token(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _ACCESS_TOKEN_PATTERN.fullmatch(value) is not None


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
