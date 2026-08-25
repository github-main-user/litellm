import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens
from litellm.models.credentials import CredentialItem
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
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
