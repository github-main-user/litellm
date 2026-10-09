import asyncio
import base64
import binascii
import json
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, Protocol
from urllib.parse import urlencode

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.credential_proxy import validate_proxy_url
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client

from .common_utils import (
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_OAUTH_TOKEN_URL,
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAccessTokenError,
)

CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS: Final = 30.0
RefreshFailureReason = Literal["invalid_grant", "credential_revoked"]
_OBJECT: Final = TypeAdapter(dict[str, object])


class ChatGPTRefreshError(RefreshAccessTokenError):
    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        reason: RefreshFailureReason | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code)
        self.reason = reason


class _ChatGPTRefreshBody(TypedDict):
    client_id: ReadOnly[str]
    grant_type: ReadOnly[str]
    refresh_token: ReadOnly[str]
    scope: ReadOnly[str]


class AsyncRefreshHTTPClient(Protocol):
    async def post(self, url: str, *, json: _ChatGPTRefreshBody) -> httpx.Response: ...


class SyncHTTPClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class ChatGPTDeviceCode:
    device_auth_id: str
    user_code: str
    interval_seconds: int


@dataclass(frozen=True, slots=True)
class ChatGPTAuthorizationCode:
    authorization_code: str
    code_verifier: str


@dataclass(frozen=True, slots=True, repr=False)
class ChatGPTTokens:
    access_token: str
    refresh_token: str
    id_token: str
    expires_at: int | None
    account_id: str | None

    def __repr__(self) -> str:
        return f"ChatGPTTokens(expires_at={self.expires_at!r}, account_id={self.account_id!r})"

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ChatGPTTokens":
        access_token: Final = value.get("access_token")
        refresh_token: Final = value.get("refresh_token")
        id_token: Final = value.get("id_token")
        expires_at: Final = value.get("expires_at")
        account_id: Final = value.get("account_id")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("ChatGPT access token is missing")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ValueError("ChatGPT refresh token is missing")
        if not isinstance(id_token, str):
            raise ValueError("ChatGPT ID token is missing")
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=int(expires_at) if isinstance(expires_at, (int, float, str)) else None,
            account_id=account_id if isinstance(account_id, str) and account_id else None,
        )

    @classmethod
    def from_json(cls, value: str) -> "ChatGPTTokens":
        parsed: Final = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("ChatGPT token bundle must be an object")
        return cls.from_mapping(parsed)


class ChatGPTRefreshClient(Protocol):
    async def async_refresh(self, previous: ChatGPTTokens) -> ChatGPTTokens: ...


class ManagedChatGPTAccessToken(str):
    account_id: str | None
    credential_name: str | None

    def __new__(
        cls, value: str, *, account_id: str | None = None, credential_name: str | None = None
    ) -> "ManagedChatGPTAccessToken":
        token: Final = super().__new__(cls, value)
        token.account_id = account_id
        token.credential_name = credential_name
        return token


def require_managed_chatgpt_access_token(model: str, credential_name: object, access_token: str | None) -> str:
    if isinstance(access_token, ManagedChatGPTAccessToken) and isinstance(credential_name, str) and credential_name:
        return access_token
    raise AuthenticationError(
        model=model,
        llm_provider="chatgpt",
        message="ChatGPT requires a managed OAuth credential. Connect an account and set litellm_credential_name.",
    )


class ChatGPTOAuthClient:
    def __init__(
        self,
        http_client: SyncHTTPClient | HTTPHandler | None = None,
        *,
        proxy_url: str | None = None,
        async_http_client: AsyncRefreshHTTPClient | None = None,
    ) -> None:
        if (http_client is not None or async_http_client is not None) and proxy_url is not None:
            raise ValueError("http_client and proxy_url cannot be used together")
        self._async_http_client = async_http_client
        self._http_client = http_client
        self._proxy_url = validate_proxy_url(proxy_url) if proxy_url is not None else None

    def request_device_code(self) -> ChatGPTDeviceCode:
        try:
            response: Final = self._post(
                CHATGPT_DEVICE_CODE_URL,
                json={"client_id": CHATGPT_CLIENT_ID},
            )
            response.raise_for_status()
            data: Final = response.json()
        except httpx.HTTPStatusError as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from None
        except Exception:  # noqa: BLE001  # OAuth failures must not expose proxy or token details
            raise GetDeviceCodeError(
                message="Failed to request device code",
                status_code=400,
            ) from None

        device_auth_id: Final = data.get("device_auth_id")
        user_code: Final = data.get("user_code") or data.get("usercode")
        interval: Final = data.get("interval")
        if not isinstance(device_auth_id, str) or not device_auth_id or not isinstance(user_code, str) or not user_code:
            raise GetDeviceCodeError(
                message="Device code response missing required fields",
                status_code=400,
            )
        interval_seconds: Final = self._parse_interval(interval)
        return ChatGPTDeviceCode(
            device_auth_id=device_auth_id,
            user_code=user_code,
            interval_seconds=interval_seconds,
        )

    def poll_authorization(self, device_code: ChatGPTDeviceCode) -> ChatGPTAuthorizationCode | None:
        try:
            response: Final = self._post(
                CHATGPT_DEVICE_TOKEN_URL,
                json={
                    "device_auth_id": device_code.device_auth_id,
                    "user_code": device_code.user_code,
                },
            )
            if response.status_code in (403, 404):
                return None
            response.raise_for_status()
            data: Final = response.json()
        except httpx.HTTPStatusError as exc:
            raise GetAccessTokenError(
                message=f"Polling failed: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from None
        except Exception:  # noqa: BLE001  # OAuth failures must not expose proxy or token details
            raise GetAccessTokenError(
                message="Polling failed",
                status_code=400,
            ) from None

        authorization_code: Final = data.get("authorization_code")
        code_verifier: Final = data.get("code_verifier")
        if not isinstance(authorization_code, str) or not authorization_code:
            raise GetAccessTokenError(
                message="Authorization response missing authorization code",
                status_code=400,
            )
        if not isinstance(code_verifier, str) or not code_verifier:
            raise GetAccessTokenError(
                message="Authorization response missing code verifier",
                status_code=400,
            )
        return ChatGPTAuthorizationCode(
            authorization_code=authorization_code,
            code_verifier=code_verifier,
        )

    def exchange_code(self, code: ChatGPTAuthorizationCode) -> ChatGPTTokens:
        redirect_uri: Final = f"{CHATGPT_AUTH_BASE}/deviceauth/callback"
        body: Final = urlencode(
            {
                "grant_type": "authorization_code",
                "code": code.authorization_code,
                "redirect_uri": redirect_uri,
                "client_id": CHATGPT_CLIENT_ID,
                "code_verifier": code.code_verifier,
            }
        )
        try:
            response: Final = self._post(
                CHATGPT_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=body,
            )
            response.raise_for_status()
            data: Final = response.json()
        except httpx.HTTPStatusError as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from None
        except Exception:  # noqa: BLE001  # OAuth failures must not expose proxy or token details
            raise GetAccessTokenError(
                message="Token exchange failed",
                status_code=400,
            ) from None
        return self._tokens_from_response(data=data, error_type=GetAccessTokenError)

    def refresh(self, refresh_token: str | ChatGPTTokens) -> ChatGPTTokens:
        previous: Final = refresh_token if isinstance(refresh_token, ChatGPTTokens) else None
        token: Final = refresh_token.refresh_token if isinstance(refresh_token, ChatGPTTokens) else refresh_token
        try:
            response: Final = self._post(CHATGPT_OAUTH_TOKEN_URL, json=self._refresh_body(token))
            return self._refresh_response(response, token, previous)
        except httpx.HTTPStatusError as exc:
            raise self._refresh_status_error(exc.response) from None
        except RefreshAccessTokenError:
            raise
        except Exception:  # noqa: BLE001  # OAuth failures must not expose proxy or token details
            raise ChatGPTRefreshError(message="Refresh token failed", status_code=503) from None

    async def async_refresh(self, previous: ChatGPTTokens) -> ChatGPTTokens:
        try:
            response: Final = await asyncio.wait_for(
                self._post_refresh(previous.refresh_token),
                timeout=CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS,
            )
            return self._refresh_response(response, previous.refresh_token, previous)
        except httpx.HTTPStatusError as exc:
            raise self._refresh_status_error(exc.response) from None
        except (httpx.TimeoutException, asyncio.TimeoutError):
            raise ChatGPTRefreshError(message="Refresh token timed out", status_code=504) from None
        except RefreshAccessTokenError:
            raise
        except Exception:  # noqa: BLE001  # OAuth failures must not expose proxy or token details
            raise ChatGPTRefreshError(message="Refresh token failed", status_code=503) from None

    async def _post_refresh(self, refresh_token: str) -> httpx.Response:
        body: Final = self._refresh_body(refresh_token)
        if self._async_http_client is not None:
            return await self._async_http_client.post(CHATGPT_OAUTH_TOKEN_URL, json=body)
        async with httpx.AsyncClient(
            proxy=self._proxy_url,
            trust_env=False,
            timeout=CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS,
        ) as client:
            return await client.post(CHATGPT_OAUTH_TOKEN_URL, json=body)

    @staticmethod
    def _refresh_body(refresh_token: str) -> _ChatGPTRefreshBody:
        return {
            "client_id": CHATGPT_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "openid profile email",
        }

    @classmethod
    def _refresh_response(
        cls, response: httpx.Response, refresh_token: str, previous: ChatGPTTokens | None
    ) -> ChatGPTTokens:
        response.raise_for_status()
        return cls._tokens_from_response(
            data=_OBJECT.validate_json(response.content),
            fallback_refresh_token=refresh_token,
            fallback_id_token=previous.id_token if previous is not None else "",
            fallback_account_id=previous.account_id if previous is not None else None,
            error_type=ChatGPTRefreshError,
        )

    @staticmethod
    def _refresh_status_error(response: httpx.Response) -> ChatGPTRefreshError:
        return ChatGPTRefreshError(
            message=f"Refresh token failed: HTTP {response.status_code}",
            status_code=response.status_code,
            reason=ChatGPTOAuthClient._refresh_failure_reason(response),
        )

    @staticmethod
    def _refresh_failure_reason(response: httpx.Response) -> RefreshFailureReason | None:
        if response.status_code not in (400, 401, 403):
            return None
        try:
            payload: Final = _OBJECT.validate_json(response.content)
            error: Final = payload.get("error")
            detail: Final = _OBJECT.validate_python(error) if isinstance(error, dict) else {}
        except ValueError:
            return None
        code: Final = (detail.get("code") or detail.get("type")) if isinstance(error, dict) else error
        if code == "invalid_grant":
            return "invalid_grant"
        if code in (
            "token_revoked", "refresh_token_revoked", "refresh_token_expired",
            "refresh_token_reused", "refresh_token_invalidated",
        ):
            return "credential_revoked"
        return None

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        if self._http_client is not None:
            client = self._http_client.client if isinstance(self._http_client, HTTPHandler) else self._http_client
            return client.post(url, **kwargs)
        if self._proxy_url is not None:
            # A dedicated client prevents provider credentials from inheriting ambient
            # proxy settings and is closed after each synchronous OAuth operation.
            with httpx.Client(proxy=self._proxy_url, trust_env=False) as client:
                return client.post(url, **kwargs)
        client = _get_httpx_client()
        return (client.client if isinstance(client, HTTPHandler) else client).post(url, **kwargs)

    @staticmethod
    def _parse_interval(value: object) -> int:
        try:
            return int(value or 5)
        except (TypeError, ValueError):
            return 5

    @classmethod
    def _tokens_from_response(
        cls,
        data: dict[str, object],
        error_type: type[GetAccessTokenError] | type[RefreshAccessTokenError],
        fallback_refresh_token: str | None = None,
        fallback_id_token: str | None = None,
        fallback_account_id: str | None = None,
    ) -> ChatGPTTokens:
        access_token: Final = data.get("access_token")
        refresh_token: Final = data.get("refresh_token") or fallback_refresh_token
        id_token: Final = data.get("id_token") or fallback_id_token
        if not isinstance(access_token, str) or not access_token:
            raise error_type(message="Token response missing access token", status_code=400)
        if not isinstance(refresh_token, str) or not refresh_token:
            raise error_type(message="Token response missing refresh token", status_code=400)
        if not isinstance(id_token, str) or (not id_token and fallback_id_token is None):
            raise error_type(message="Token response missing ID token", status_code=400)
        return ChatGPTTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=cls.get_expires_at(access_token),
            account_id=cls.extract_account_id(id_token or access_token) or fallback_account_id,
        )

    @staticmethod
    def decode_jwt_claims(token: str) -> dict[str, Any]:
        try:
            parts: Final = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64: Final = parts[1] + "=" * (-len(parts[1]) % 4)
            payload_bytes: Final = base64.urlsafe_b64decode(payload_b64)
            payload: Final = json.loads(payload_bytes.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    @classmethod
    def get_expires_at(cls, token: str) -> int | None:
        exp: Final = cls.decode_jwt_claims(token).get("exp")
        return int(exp) if isinstance(exp, (int, float)) else None

    @classmethod
    def extract_account_id(cls, token: str | None) -> str | None:
        if not token:
            return None
        auth_claims: Final = cls.decode_jwt_claims(token).get("https://api.openai.com/auth")
        if not isinstance(auth_claims, dict):
            return None
        account_id: Final = auth_claims.get("chatgpt_account_id")
        return account_id if isinstance(account_id, str) and account_id else None
