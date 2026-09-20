import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from litellm.llms.anthropic.oauth_client import (
    ANTHROPIC_OAUTH_AUTHORIZE_URL,
    ANTHROPIC_OAUTH_REDIRECT_URI,
    ANTHROPIC_OAUTH_REFRESH_SCOPES,
    ANTHROPIC_OAUTH_TOKEN_URL,
    AnthropicOAuthClient,
    AnthropicOAuthError,
    AnthropicOAuthTokens,
    recover_managed_anthropic_oauth_headers,
)


def test_begin_authorization_has_state_and_pkce() -> None:
    authorization = AnthropicOAuthClient().begin_authorization()
    query = parse_qs(urlparse(authorization.authorization_url).query)

    assert query["state"] == [authorization.state]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code"] == ["true"]
    # Claude Code 2.1.278 public binary constants inspected 2026-09-20.
    assert ANTHROPIC_OAUTH_AUTHORIZE_URL == "https://claude.com/cai/oauth/authorize"
    assert ANTHROPIC_OAUTH_REDIRECT_URI == "https://platform.claude.com/oauth/code/callback"
    assert query["redirect_uri"] == [ANTHROPIC_OAUTH_REDIRECT_URI]
    assert authorization.authorization_url.startswith(f"{ANTHROPIC_OAUTH_AUTHORIZE_URL}?")
    assert authorization.code_verifier not in authorization.authorization_url


@pytest.mark.asyncio
async def test_exchange_reads_nested_account_uuid_and_refresh_preserves_it() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body["grant_type"] == "authorization_code":
            return httpx.Response(
                200,
                json={
                    "access_token": "sk-ant-oat-initial",
                    "refresh_token": "refresh-initial",
                    "expires_in": 3600,
                    "account": {"uuid": "account-a"},
                    "organization": {"id": "organization-must-not-be-the-account"},
                },
            )
        return httpx.Response(200, json={"access_token": "sk-ant-oat-refreshed", "expires_in": 3600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AnthropicOAuthClient(http_client)
        initial = await client.exchange_code("code", "state", "verifier")
        refreshed = await client.refresh(initial)

    assert initial.account_id == "account-a"
    # Claude Code 2.1.278 public token exchange inspected 2026-09-20.
    assert requests[0] == {
        "grant_type": "authorization_code",
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        "code": "code",
        "state": "state",
        "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
        "code_verifier": "verifier",
    }
    assert refreshed.refresh_token == "refresh-initial"
    assert refreshed.account_id == "account-a"
    assert requests[1]["refresh_token"] == "refresh-initial"
    assert requests[1]["scope"] == " ".join(ANTHROPIC_OAUTH_REFRESH_SCOPES)
    assert "refresh-initial" not in repr(refreshed)


@pytest.mark.asyncio
async def test_refresh_replaces_rotated_token() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        # Claude Code 2.1.278 public binary constant inspected 2026-09-20.
        assert ANTHROPIC_OAUTH_TOKEN_URL == "https://platform.claude.com/v1/oauth/token"
        assert str(request.url) == ANTHROPIC_OAUTH_TOKEN_URL
        assert "anthropic-sdk-typescript" not in request.headers.get("user-agent", "")
        return httpx.Response(
            200,
            json={
                "access_token": "sk-ant-oat-next",
                "refresh_token": "refresh-next",
                "expires_in": 300,
                "account_id": "account-a",
            },
        )

    previous = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", 1, "account-a")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        refreshed = await AnthropicOAuthClient(http_client).refresh(previous)

    assert refreshed.refresh_token == "refresh-next"


@pytest.mark.asyncio
async def test_organization_id_is_not_used_as_account_identity() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "sk-ant-oat-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "organization_id": "organization-a",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        tokens = await AnthropicOAuthClient(http_client).exchange_code("code", "state", "verifier")

    assert tokens.account_id is None


@pytest.mark.asyncio
async def test_provider_error_is_sanitized_without_secret_bearing_cause() -> None:
    response_secret = "sk-ant-oat-provider-secret-body"
    code_secret = "authorization-code-secret"
    verifier_secret = "pkce-verifier-secret"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert code_secret in request.content.decode()
        return httpx.Response(
            401,
            json={"error_description": response_secret, "access_token": response_secret},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(AnthropicOAuthError) as caught:
            await AnthropicOAuthClient(http_client).exchange_code(code_secret, "state", verifier_secret)

    assert caught.value.status_code == 401
    assert caught.value.__cause__ is None
    error_text = f"{caught.value!s} {caught.value!r}"
    assert response_secret not in error_text
    assert code_secret not in error_text
    assert verifier_secret not in error_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "retry_after", "expected_retry_after"),
    [(429, "17", "17"), (503, "not-a-delay", None)],
)
async def test_transient_provider_error_preserves_retry_metadata_safely(
    status_code: int,
    retry_after: str,
    expected_retry_after: str | None,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers={"Retry-After": retry_after}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(AnthropicOAuthError) as caught:
            await AnthropicOAuthClient(http_client).exchange_code("code", "state", "verifier")

    assert caught.value.status_code == status_code
    assert caught.value.retryable
    assert caught.value.retry_after == expected_retry_after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status_code"),
    [(httpx.ConnectError("private network detail"), 503), (httpx.ReadTimeout("private timeout detail"), 504)],
)
async def test_transient_transport_error_is_sanitized_and_retryable(
    failure: httpx.HTTPError,
    status_code: int,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(AnthropicOAuthError) as caught:
            await AnthropicOAuthClient(http_client).exchange_code("code", "state", "verifier")

    assert caught.value.status_code == status_code
    assert caught.value.retryable
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_token_bundle_validation_and_log_safety() -> None:
    tokens = AnthropicOAuthTokens("sk-ant-oat-access", "refresh", 123.0, "account")

    assert AnthropicOAuthTokens.from_json(tokens.to_json()) == tokens
    assert "sk-ant-oat-access" not in repr(tokens)
    assert "refresh" not in repr(tokens)
    with pytest.raises(ValueError):
        AnthropicOAuthTokens.from_json('{"access_token":"not-oauth","refresh_token":"refresh","expires_at":1}')


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["\r\nsecret", " secret", "\x00secret", "\x7fsecret", "ésecret"])
async def test_unsafe_access_tokens_are_rejected_before_header_construction(suffix: str) -> None:
    token = f"sk-ant-oat-{suffix}"
    payload = {"access_token": token, "refresh_token": "refresh", "expires_in": 3600}
    with pytest.raises(ValueError) as stored_error:
        AnthropicOAuthTokens.from_json(json.dumps({**payload, "expires_at": 3600}))
    assert token not in str(stored_error.value)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(AnthropicOAuthError) as upstream_error:
            await AnthropicOAuthClient(http_client).exchange_code("code", "state", "verifier")
    assert token not in str(upstream_error.value)


@pytest.mark.asyncio
async def test_recovery_does_not_return_unsafe_headers(monkeypatch) -> None:
    import litellm

    secret = "sk-ant-oat-private\r\nvalue"
    callback = SimpleNamespace(recover_rejected_token=AsyncMock(return_value=secret))
    monkeypatch.setattr(litellm, "callbacks", [callback])
    with pytest.raises(AnthropicOAuthError) as error:
        await recover_managed_anthropic_oauth_headers(
            headers={"Authorization": "Bearer sk-ant-oat-rejected"},
            litellm_params={"litellm_credential_name": "subscription"},
        )
    assert error.value.status_code == 502
    assert secret not in str(error.value)
    callback.recover_rejected_token.assert_awaited_once_with("subscription", "sk-ant-oat-rejected")


@pytest.mark.parametrize("expires_at", [math.inf, -math.inf, math.nan])
def test_stored_token_bundle_rejects_nonfinite_expiry(expires_at: float) -> None:
    serialized = json.dumps(
        {
            "access_token": "sk-ant-oat-access",
            "refresh_token": "refresh",
            "expires_at": expires_at,
        }
    )

    with pytest.raises(ValueError):
        AnthropicOAuthTokens.from_json(serialized)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", [math.inf, -math.inf, math.nan])
async def test_token_response_rejects_nonfinite_expiry(expires_in: float) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "sk-ant-oat-access",
                "refresh_token": "refresh",
                "expires_in": expires_in,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(AnthropicOAuthError):
            await AnthropicOAuthClient(http_client).exchange_code("code", "state", "verifier")
