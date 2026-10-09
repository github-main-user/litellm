import base64
import json
import time

import httpx
import pytest

import litellm
from litellm.exceptions import AuthenticationError
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
    require_managed_chatgpt_access_token,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.models.credentials import CredentialItem


@pytest.mark.parametrize("provider,auth_type,bundle", [
    ("anthropic", "oauth", ChatGPTTokens("token", "refresh", "id", 9999999999, "account").to_json()),
    ("chatgpt", "api_key", ChatGPTTokens("token", "refresh", "id", 9999999999, "account").to_json()),
    ("chatgpt", "oauth", "invalid-json"),
])
def test_rejects_credentials_not_managed_chatgpt_oauth(
    monkeypatch: pytest.MonkeyPatch, provider: str, auth_type: str, bundle: str
) -> None:
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="subscription",
        credential_info={"provider": provider, "auth_type": auth_type},
        credential_values={"litellm_internal_chatgpt_auth_token": bundle},
    )])
    with pytest.raises(AuthenticationError, match="managed OAuth credential"):
        require_managed_chatgpt_access_token("test-model", "subscription", "token")


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


def test_pending_device_authorization_with_http_handler_returns_none() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    client = ChatGPTOAuthClient(HTTPHandler(client=http_client))

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


def test_client_rejects_injected_transport_with_proxy() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    try:
        with pytest.raises(ValueError, match="cannot be used together"):
            ChatGPTOAuthClient(http_client, proxy_url="http://proxy.example:8080")
    finally:
        http_client.close()


def test_invalid_proxy_performs_no_http_operation(monkeypatch) -> None:
    opened = False

    def fail_client(**kwargs: object) -> None:
        nonlocal opened
        opened = True

    monkeypatch.setattr(httpx, "Client", fail_client)
    secret_url = "http://proxy-secret.example:70000"
    with pytest.raises(ValueError) as caught:
        ChatGPTOAuthClient(proxy_url=secret_url)

    assert not opened
    assert secret_url not in str(caught.value)


def test_dedicated_proxy_client_disables_environment_and_closes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            captured["closed"] = True

        def post(self, url: str, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                200,
                json={"device_auth_id": "device", "user_code": "CODE"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "Client", Client)
    ChatGPTOAuthClient(proxy_url="http://proxy.example:8080").request_device_code()

    assert captured["proxy"] == "http://proxy.example:8080"
    assert captured["trust_env"] is False
    assert captured["closed"] is True


def test_token_bundle_round_trip() -> None:
    tokens = ChatGPTTokens(
        access_token="access",
        refresh_token="refresh",
        id_token="id",
        expires_at=123,
        account_id="account",
    )

    assert ChatGPTTokens.from_json(tokens.to_json()) == tokens


@pytest.mark.asyncio
async def test_refresh_without_id_token_preserves_identity_and_rotates_refresh_token() -> None:
    previous = ChatGPTTokens("access-old", "refresh-old", "id-old", 1, "account-old")
    access = _jwt({"exp": int(time.time()) + 3600})
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"access_token": access, "refresh_token": "rotated"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        tokens = await ChatGPTOAuthClient(async_http_client=transport).async_refresh(previous)

    assert tokens == ChatGPTTokens(access, "rotated", previous.id_token, ChatGPTOAuthClient.get_expires_at(access), "account-old")
    assert requests[0]["refresh_token"] == previous.refresh_token


@pytest.mark.parametrize("status,payload,reason", [
    (400, {"error": "invalid_grant"}, "invalid_grant"),
    (401, {"error": {"code": "refresh_token_reused", "message": "secret"}}, "credential_revoked"),
    (403, {"error": {"type": "refresh_token_revoked"}}, "credential_revoked"),
    (400, {"error": "invalid_request", "error_description": "invalid_grant secret"}, None),
    (401, {"error": "unauthorized"}, None),
    (429, {"error": "invalid_grant"}, None),
    (503, {"error": "invalid_grant"}, None),
    (400, ["invalid_grant"], None),
])
def test_refresh_failure_classification_uses_only_explicit_oauth_error_codes(status, payload, reason) -> None:
    from litellm.llms.chatgpt.oauth_client import ChatGPTRefreshError

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload))) as transport:
        with pytest.raises(ChatGPTRefreshError) as caught:
            ChatGPTOAuthClient(transport).refresh("refresh-secret")

    assert caught.value.status_code == status
    assert caught.value.reason == reason
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_async_refresh_total_deadline_bounds_multiple_http_waits(monkeypatch) -> None:
    import asyncio

    from litellm.llms.chatgpt import oauth_client

    stopped = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.sleep(0.03)
            await asyncio.sleep(0.03)
            return httpx.Response(200, json={"access_token": "too-late"})
        finally:
            stopped.set()

    monkeypatch.setattr(oauth_client, "CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS", 0.05)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(oauth_client.ChatGPTRefreshError) as caught:
            await ChatGPTOAuthClient(async_http_client=transport).async_refresh(
                ChatGPTTokens("access", "refresh", "id", 1, "account")
            )

    assert caught.value.status_code == 504
    assert caught.value.reason is None
    assert stopped.is_set()


def test_sync_refresh_can_retain_previous_id_and_refresh_token() -> None:
    previous = ChatGPTTokens("old-access", "old-refresh", "old-id", 1, "account")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"access_token": "new-access"}))) as transport:
        refreshed = ChatGPTOAuthClient(transport).refresh(previous)
    assert refreshed == ChatGPTTokens("new-access", previous.refresh_token, previous.id_token, None, previous.account_id)


def test_sync_refresh_without_previous_id_preserves_rotated_bundle_round_trip() -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"access_token": "new-access", "refresh_token": "rotated-refresh"}
    ))) as transport:
        refreshed = ChatGPTOAuthClient(transport).refresh("old-refresh")
    assert refreshed.refresh_token == "rotated-refresh"
    assert ChatGPTTokens.from_json(refreshed.to_json()) == refreshed
