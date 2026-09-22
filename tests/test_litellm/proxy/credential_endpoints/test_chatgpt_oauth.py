import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.chatgpt.oauth_client import (
    ChatGPTAuthorizationCode,
    ChatGPTDeviceCode,
    ChatGPTTokens,
)
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    CREDENTIAL_PROXY_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
    ChatGPTOAuthPollRequest,
    ChatGPTOAuthStartRequest,
    poll_chatgpt_oauth,
    start_chatgpt_oauth,
)
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.utils import load_credentials_from_list


def _credential(name: str, account_id: str, access_token: str) -> CredentialItem:
    tokens = ChatGPTTokens(
        access_token=access_token,
        refresh_token=f"refresh-{account_id}",
        id_token=f"id-{account_id}",
        expires_at=int(time.time()) + 3600,
        account_id=account_id,
    )
    return CredentialItem(
        credential_name=name,
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


@pytest.mark.asyncio
async def test_hook_resolves_each_deployment_subscription_independently() -> None:
    previous = litellm.credential_list
    litellm.credential_list = [
        _credential("subscription-a", "account-a", "access-a"),
        _credential("subscription-b", "account-b", "access-b"),
    ]
    try:
        hook = ChatGPTOAuthCredentialHook()
        resolved_a = await hook.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "litellm_credential_name": "subscription-a",
            },
            None,
        )
        resolved_b = await hook.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "litellm_credential_name": "subscription-b",
            },
            None,
        )
    finally:
        litellm.credential_list = previous

    assert resolved_a is not None
    assert resolved_a["api_key"] == "access-a"
    assert resolved_a["chatgpt_auth_account_id"] == "account-a"
    assert resolved_b is not None
    assert resolved_b["api_key"] == "access-b"
    assert resolved_b["chatgpt_auth_account_id"] == "account-b"


@pytest.mark.asyncio
async def test_hook_ignores_non_oauth_chatgpt_credential() -> None:
    previous = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="chatgpt-api-key",
            credential_info={"provider": "chatgpt", "auth_type": "api_key"},
            credential_values={"api_key": "regular-api-key"},
        )
    ]
    try:
        resolved = await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "litellm_credential_name": "chatgpt-api-key",
            },
            None,
        )
    finally:
        litellm.credential_list = previous

    assert resolved is None


@pytest.mark.asyncio
async def test_hook_loads_oauth_credential_from_database_on_cache_miss() -> None:
    previous = litellm.credential_list
    litellm.credential_list = []
    stored = _credential("subscription-a", "account-a", "access-a")
    prisma_client = MagicMock()
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch.object(
                CredentialsRepository,
                "find_by_name",
                AsyncMock(return_value=stored),
            ),
            patch(
                "litellm.proxy.credential_endpoints.chatgpt_oauth.decrypt_value_helper",
                side_effect=lambda value, key: value,
            ),
        ):
            resolved = await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
                {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "subscription-a",
                },
                None,
            )
    finally:
        litellm.credential_list = previous

    assert resolved is not None
    assert resolved["api_key"] == "access-a"
    assert resolved["chatgpt_auth_account_id"] == "account-a"


def test_internal_token_bundle_is_not_copied_into_request_kwargs() -> None:
    previous = litellm.credential_list
    litellm.credential_list = [_credential("subscription-a", "account-a", "access-a")]
    kwargs = {"litellm_credential_name": "subscription-a"}
    try:
        load_credentials_from_list(kwargs)
    finally:
        litellm.credential_list = previous

    assert CHATGPT_CREDENTIAL_VALUE_KEY not in kwargs


@pytest.mark.asyncio
async def test_concurrent_expired_requests_share_one_refresh() -> None:
    expired = ChatGPTTokens(
        access_token="access-old",
        refresh_token="refresh-old",
        id_token="id-old",
        expires_at=int(time.time()) - 60,
        account_id="account-a",
    )
    fresh = ChatGPTTokens(
        access_token="access-new",
        refresh_token="refresh-new",
        id_token="id-new",
        expires_at=int(time.time()) + 3600,
        account_id="account-a",
    )
    previous = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="subscription-a",
            credential_info={"provider": "chatgpt", "auth_type": "oauth"},
            credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: expired.to_json()},
        )
    ]
    transaction = MagicMock()
    transaction.execute_raw = AsyncMock()
    transaction.litellm_credentialstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: expired.to_json()},
            credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        )
    )
    transaction.litellm_credentialstable.update = AsyncMock()
    transaction_context = MagicMock()
    transaction_context.__aenter__ = AsyncMock(return_value=transaction)
    transaction_context.__aexit__ = AsyncMock(return_value=None)
    prisma_client = MagicMock()
    prisma_client.db.tx.return_value = transaction_context
    oauth_client = MagicMock()
    oauth_client.refresh.return_value = fresh
    hook = ChatGPTOAuthCredentialHook(oauth_client=oauth_client)

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch(
                "litellm.proxy.credential_endpoints.chatgpt_oauth.decrypt_value_helper",
                side_effect=lambda value, key: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.chatgpt_oauth.encrypt_value_helper",
                side_effect=lambda value: value,
            ),
        ):
            first, second = await asyncio.gather(
                hook._get_tokens("subscription-a"),
                hook._get_tokens("subscription-a"),
            )
    finally:
        litellm.credential_list = previous

    assert first == fresh
    assert second == fresh
    oauth_client.refresh.assert_called_once_with("refresh-old")
    transaction.litellm_credentialstable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_poll_store_reconnect_uses_override_and_encrypts_until_success() -> None:
    old_proxy = "http://old-proxy.example:8080"
    new_proxy = "socks5://new-proxy.example:1080"
    old_tokens = ChatGPTTokens("old-access", "old-refresh", "old-id", 1, "account-a")
    new_tokens = ChatGPTTokens("new-access", "new-refresh", "new-id", int(time.time()) + 3600, "account-a")
    calls: list[str | None] = []
    updated: list[dict[str, object]] = []

    class OAuthClient:
        def __init__(self, *, proxy_url=None):
            calls.append(proxy_url)

        def request_device_code(self):
            return ChatGPTDeviceCode("device", "CODE", 1)

        def poll_authorization(self, device_code):
            return ChatGPTAuthorizationCode("code", "verifier")

        def exchange_code(self, authorization):
            return new_tokens

    class Table:
        async def find_unique(self, *, where):
            return row

        async def update(self, *, where, data):
            updated.append(data)

    class Transaction:
        litellm_credentialstable = Table()

        async def execute_raw(self, query, key):
            return None

    class Database:
        def tx(self):
            @asynccontextmanager
            async def context():
                yield Transaction()

            return context()

    previous = litellm.credential_list
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "chatgpt-proxy-lifecycle-key"}):
        row = SimpleNamespace(
            credential_info={"provider": "chatgpt", "auth_type": "oauth", "custom": "preserved"},
            credential_values={
                CHATGPT_CREDENTIAL_VALUE_KEY: encrypt_value_helper(old_tokens.to_json()),
                CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(old_proxy),
                "other": "preserved",
            },
        )
        litellm.credential_list = [
            CredentialItem(
                credential_name="subscription",
                credential_info={"provider": "chatgpt", "auth_type": "oauth", "proxy_configured": True},
                credential_values={
                    CHATGPT_CREDENTIAL_VALUE_KEY: old_tokens.to_json(),
                    CREDENTIAL_PROXY_VALUE_KEY: old_proxy,
                },
            )
        ]
        try:
            with (
                patch("litellm.proxy.credential_endpoints.chatgpt_oauth.ChatGPTOAuthClient", OAuthClient),
                patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database())),
                patch(
                    "litellm.proxy.credential_endpoints.chatgpt_oauth.publish_config_change_for_object_type",
                    AsyncMock(),
                ),
            ):
                started = await start_chatgpt_oauth(
                    ChatGPTOAuthStartRequest(credential_name="subscription", proxy_url=new_proxy),
                    UserAPIKeyAuth(user_id="actor-a"),
                )
                assert old_proxy not in started.attempt_token
                assert new_proxy not in started.attempt_token
                connected = await poll_chatgpt_oauth(
                    ChatGPTOAuthPollRequest(attempt_token=started.attempt_token),
                    UserAPIKeyAuth(user_id="actor-a"),
                )
        finally:
            litellm.credential_list = previous

    assert connected.status == "connected"
    assert calls == [new_proxy, new_proxy]
    values = updated[0]["credential_values"]
    if isinstance(values, str):
        values = json.loads(values)
    assert isinstance(values, dict)
    assert values["other"] == "preserved"
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "chatgpt-proxy-lifecycle-key"}):
        assert decrypt_value_helper(values[CREDENTIAL_PROXY_VALUE_KEY], CREDENTIAL_PROXY_VALUE_KEY) == new_proxy
        assert decrypt_value_helper(values[CHATGPT_CREDENTIAL_VALUE_KEY], CHATGPT_CREDENTIAL_VALUE_KEY) == new_tokens.to_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"credential_name": "subscription"}, {"credential_name": "subscription", "proxy_url": None}])
async def test_start_reuses_saved_proxy_when_omitted_or_null(payload: dict[str, object]) -> None:
    saved_proxy = "http://saved-proxy.example:8080"
    seen: list[str | None] = []

    class OAuthClient:
        def __init__(self, *, proxy_url=None):
            seen.append(proxy_url)

        def request_device_code(self):
            return ChatGPTDeviceCode("device", "CODE", 1)

    previous = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="subscription",
            credential_info={"provider": "chatgpt", "auth_type": "oauth", "proxy_configured": True},
            credential_values={CREDENTIAL_PROXY_VALUE_KEY: saved_proxy},
        )
    ]
    try:
        with (
            patch.dict("os.environ", {"LITELLM_SALT_KEY": "chatgpt-proxy-reuse-key"}),
            patch("litellm.proxy.credential_endpoints.chatgpt_oauth.ChatGPTOAuthClient", OAuthClient),
        ):
            await start_chatgpt_oauth(
                ChatGPTOAuthStartRequest.model_validate(payload), UserAPIKeyAuth(user_id="actor-a")
            )
    finally:
        litellm.credential_list = previous

    assert seen == [saved_proxy]
