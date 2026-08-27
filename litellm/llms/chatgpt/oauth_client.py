import base64
import binascii
import json
from dataclasses import asdict, dataclass
from typing import Any, Final, Protocol
from urllib.parse import urlencode

import httpx

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
        if not isinstance(id_token, str) or not id_token:
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


class ChatGPTOAuthClient:
    def __init__(self, http_client: SyncHTTPClient | HTTPHandler | None = None) -> None:
        self._http_client = http_client

    def request_device_code(self) -> ChatGPTDeviceCode:
        try:
            response: Final = self._client().post(
                CHATGPT_DEVICE_CODE_URL,
                json={"client_id": CHATGPT_CLIENT_ID},
            )
            response.raise_for_status()
            data: Final = response.json()
        except httpx.HTTPStatusError as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise GetDeviceCodeError(
                message="Failed to request device code",
                status_code=400,
            ) from exc

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
            response: Final = self._client().post(
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
            ) from exc
        except Exception as exc:
            raise GetAccessTokenError(
                message="Polling failed",
                status_code=400,
            ) from exc

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
            response: Final = self._client().post(
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
            ) from exc
        except Exception as exc:
            raise GetAccessTokenError(
                message="Token exchange failed",
                status_code=400,
            ) from exc
        return self._tokens_from_response(data=data, error_type=GetAccessTokenError)

    def refresh(self, refresh_token: str) -> ChatGPTTokens:
        try:
            response: Final = self._client().post(
                CHATGPT_OAUTH_TOKEN_URL,
                json={
                    "client_id": CHATGPT_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
            )
            response.raise_for_status()
            data: Final = response.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise RefreshAccessTokenError(
                message="Refresh token failed",
                status_code=400,
            ) from exc
        return self._tokens_from_response(
            data=data,
            fallback_refresh_token=refresh_token,
            error_type=RefreshAccessTokenError,
        )

    def _client(self) -> SyncHTTPClient:
        client = self._http_client or _get_httpx_client()
        return client.client if isinstance(client, HTTPHandler) else client

    @staticmethod
    def _parse_interval(value: object) -> int:
        try:
            return int(value or 5)
        except (TypeError, ValueError):
            return 5

    @classmethod
    def _tokens_from_response(
        cls,
        data: dict[str, Any],
        error_type: type[GetAccessTokenError] | type[RefreshAccessTokenError],
        fallback_refresh_token: str | None = None,
    ) -> ChatGPTTokens:
        access_token: Final = data.get("access_token")
        refresh_token: Final = data.get("refresh_token") or fallback_refresh_token
        id_token: Final = data.get("id_token")
        if not isinstance(access_token, str) or not access_token:
            raise error_type(message="Token response missing access token", status_code=400)
        if not isinstance(refresh_token, str) or not refresh_token:
            raise error_type(message="Token response missing refresh token", status_code=400)
        if not isinstance(id_token, str) or not id_token:
            raise error_type(message="Token response missing ID token", status_code=400)
        return ChatGPTTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=cls.get_expires_at(access_token),
            account_id=cls.extract_account_id(id_token or access_token),
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
