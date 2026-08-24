import base64
import json
import time

import httpx

from litellm.llms.chatgpt.common_utils import (
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_OAUTH_TOKEN_URL,
)
from litellm.llms.chatgpt.oauth_client import (
    ChatGPTAuthorizationCode,
    ChatGPTDeviceCode,
    ChatGPTOAuthClient,
    ChatGPTTokens,
)


def _jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}."


def test_device_code_and_token_exchange() -> None:
    expires_at = int(time.time()) + 3600
    access_token = _jwt({"exp": expires_at})
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account-a"}})

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_DEVICE_CODE_URL:
            return httpx.Response(
                200,
                json={"device_auth_id": "device-a", "user_code": "CODE-A", "interval": 2},
            )
        if str(request.url) == CHATGPT_DEVICE_TOKEN_URL:
            return httpx.Response(
                200,
                json={"authorization_code": "authorization-a", "code_verifier": "verifier-a"},
            )
        if str(request.url) == CHATGPT_OAUTH_TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "refresh_token": "refresh-a",
                    "id_token": id_token,
                },
            )
        return httpx.Response(404)

    client = ChatGPTOAuthClient(httpx.Client(transport=httpx.MockTransport(handler)))
    device_code = client.request_device_code()
    authorization = client.poll_authorization(device_code)

    assert device_code == ChatGPTDeviceCode("device-a", "CODE-A", 2)
    assert authorization == ChatGPTAuthorizationCode("authorization-a", "verifier-a")
    assert authorization is not None
    tokens = client.exchange_code(authorization)
    assert tokens.access_token == access_token
    assert tokens.expires_at == expires_at
    assert tokens.account_id == "account-a"


def test_pending_device_authorization_returns_none() -> None:
    client = ChatGPTOAuthClient(httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403))))

    assert client.poll_authorization(ChatGPTDeviceCode("device", "CODE", 5)) is None


def test_refresh_preserves_rotating_credentials() -> None:
    access_token = _jwt({"exp": int(time.time()) + 3600})
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account-b"}})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": access_token,
                "refresh_token": "refresh-new",
                "id_token": id_token,
            },
        )

    client = ChatGPTOAuthClient(httpx.Client(transport=httpx.MockTransport(handler)))
    tokens = client.refresh("refresh-old")

    assert tokens.refresh_token == "refresh-new"
    assert tokens.account_id == "account-b"
    assert "refresh-new" not in repr(tokens)


def test_token_bundle_round_trip() -> None:
    tokens = ChatGPTTokens(
        access_token="access",
        refresh_token="refresh",
        id_token="id",
        expires_at=123,
        account_id="account",
    )

    assert ChatGPTTokens.from_json(tokens.to_json()) == tokens
