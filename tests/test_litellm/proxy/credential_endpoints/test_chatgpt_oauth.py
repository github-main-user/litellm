import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.oauth_client import (
    ChatGPTAuthorizationCode,
    ChatGPTDeviceCode,
    ChatGPTTokens,
)
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    CREDENTIAL_PROXY_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
    ChatGPTOAuthPollRequest,
    ChatGPTOAuthStartRequest,
    _DatabaseChatGPTCredential,
    poll_chatgpt_oauth,
    start_chatgpt_oauth,
)
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import load_credentials_from_list


def _credential(name: str, account_id: str, access_token: str) -> CredentialItem:
    tokens = ChatGPTTokens(
        access_token=access_token,
        refresh_token=f"refresh-{account_id}",
        id_token=f"id-{account_id}",
        expires_at=int(time.time()) + 3600,
        account_id=account_id,
    )
    return _DatabaseChatGPTCredential(
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
async def test_hook_rejects_non_oauth_chatgpt_credential() -> None:
    previous = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="chatgpt-api-key",
            credential_info={"provider": "chatgpt", "auth_type": "api_key"},
            credential_values={"api_key": "regular-api-key"},
        )
    ]
    try:
        with pytest.raises(AuthenticationError, match="unavailable"):
            await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
                {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "chatgpt-api-key",
                },
                None,
            )
    finally:
        litellm.credential_list = previous


@pytest.mark.asyncio
async def test_hook_rejects_configured_oauth_tokens_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = CredentialItem.model_validate(_credential("subscription-a", "account-a", "access-a").model_dump())
    monkeypatch.setattr(litellm, "credential_list", [configured])
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        with pytest.raises(AuthenticationError, match="unavailable"):
            await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
                {"model": "chatgpt/test-model", "litellm_credential_name": "subscription-a"}, None
            )


@pytest.mark.asyncio
async def test_inflight_auth_survives_cache_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "credential_list", [_credential("subscription-a", "account-a", "access-a")])
    resolved = await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
        {"model": "chatgpt/test-model", "litellm_credential_name": "subscription-a"}, None
    )
    assert resolved is not None
    monkeypatch.setattr(litellm, "credential_list", [_credential("subscription-a", "account-a", "access-new")])
    headers = ChatGPTResponsesAPIConfig().validate_environment(
        headers={}, model="test-model", litellm_params=GenericLiteLLMParams(**resolved)
    )
    assert headers["Authorization"] == "Bearer access-a"
    assert headers["ChatGPT-Account-Id"] == "account-a"


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
                side_effect=lambda value, key, **kwargs: value,
            ),
        ):
            resolved = await ChatGPTOAuthCredentialHook().async_pre_call_deployment_hook(
                {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "subscription-a",
                },
                None,
            )
            assert resolved is not None
            headers = ChatGPTResponsesAPIConfig().validate_environment(
                headers={}, model="gpt-5.4", litellm_params=GenericLiteLLMParams(**resolved)
            )
    finally:
        litellm.credential_list = previous

    assert resolved["api_key"] == "access-a"
    assert resolved["chatgpt_auth_account_id"] == "account-a"
    assert headers["Authorization"] == "Bearer access-a"
    assert headers["ChatGPT-Account-Id"] == "account-a"


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
        _DatabaseChatGPTCredential(
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
    oauth_client.async_refresh = AsyncMock(return_value=fresh)
    hook = ChatGPTOAuthCredentialHook(oauth_client=oauth_client)

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch(
                "litellm.proxy.credential_endpoints.chatgpt_oauth.decrypt_value_helper",
                side_effect=lambda value, key, **kwargs: value,
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
    oauth_client.async_refresh.assert_awaited_once_with(expired)
    transaction.litellm_credentialstable.update.assert_awaited_once()
    persisted = transaction.litellm_credentialstable.update.call_args.kwargs["data"]
    assert json.loads(persisted["credential_values"]) == {CHATGPT_CREDENTIAL_VALUE_KEY: fresh.to_json()}
    assert json.loads(persisted["credential_info"]) == {
        "provider": "chatgpt", "auth_type": "oauth", "proxy_configured": False,
    }


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

        async def execute_raw(self, query, key=None):
            return None

    class Database:
        def tx(self, **kwargs):
            @asynccontextmanager
            async def context():
                yield Transaction()

            return context()

    previous = litellm.credential_list
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "chatgpt-proxy-lifecycle-key"}):
        row = SimpleNamespace(
            credential_info={
                "provider": "chatgpt", "auth_type": "oauth", "custom": "preserved",
                "chatgpt_reauth_required_token": "old-block", "chatgpt_reauth_required_reason": "invalid_grant",
            },
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
    info = json.loads(updated[0]["credential_info"])
    assert info == {"provider": "chatgpt", "auth_type": "oauth", "custom": "preserved", "proxy_configured": True}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "chatgpt"])
@pytest.mark.parametrize("lookup", ["saved", "new", "failure", "corrupt"])
async def test_oauth_start_cold_cache_checks_database_before_network(monkeypatch, provider, lookup):
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.credential_endpoints import anthropic_oauth, chatgpt_oauth

    module = anthropic_oauth if provider == "anthropic" else chatgpt_oauth
    request_type = module.AnthropicOAuthStartRequest if provider == "anthropic" else module.ChatGPTOAuthStartRequest
    start = module.start_anthropic_oauth if provider == "anthropic" else module.start_chatgpt_oauth
    client_name = "AnthropicOAuthClient" if provider == "anthropic" else "ChatGPTOAuthClient"
    monkeypatch.setenv("LITELLM_SALT_KEY", "cold-cache-key")
    monkeypatch.setattr(litellm, "credential_list", [])
    proxy = "socks5h://user:secret@proxy.invalid:1080"
    row = CredentialItem(
        credential_name="cold-subscription",
        credential_info={"provider": provider, "auth_type": "oauth", "proxy_configured": True},
        credential_values={CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(proxy)} if lookup == "saved" else {},
    )
    repository_lookup = AsyncMock(return_value=None if lookup == "new" else row)
    if lookup == "failure":
        repository_lookup.side_effect = RuntimeError("secret database connection")
    client = MagicMock()
    client.return_value.begin_authorization.return_value = SimpleNamespace(
        state="state", code_verifier="verifier", authorization_url="https://example.com/auth"
    )
    client.return_value.request_device_code.return_value = ChatGPTDeviceCode(
        device_auth_id="device", user_code="code", interval_seconds=1
    )
    prisma = SimpleNamespace(
        db=SimpleNamespace(litellm_credentialstable=SimpleNamespace(find_unique=repository_lookup))
    )
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch.object(module, client_name, client),
    ):
        if lookup in {"failure", "corrupt"}:
            with pytest.raises(HTTPException) as error:
                await start(request_type(credential_name="cold-subscription"), UserAPIKeyAuth(
                    user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
                ))
            assert "secret" not in str(error.value.detail)
            client.assert_not_called()
        else:
            await start(request_type(credential_name="cold-subscription"), UserAPIKeyAuth(
                user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
            ))
            client.assert_called_once_with(proxy_url=proxy if lookup == "saved" else None)
        repository_lookup.assert_awaited_once_with(where={"credential_name": "cold-subscription"})


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["load", "refresh-expired", "refresh-fresh", "store", "refresh-latest", "store-latest"])
@pytest.mark.parametrize("proxy_value", [None, "", 17, False, "undecryptable", "encrypted-empty", "encrypted-invalid"])
async def test_chatgpt_database_required_proxy_fails_closed(monkeypatch, operation, proxy_value):
    from litellm.proxy.credential_endpoints.chatgpt_oauth import _store_tokens

    monkeypatch.setenv("LITELLM_SALT_KEY", "chatgpt-required-proxy-key")
    monkeypatch.setattr(litellm, "credential_list", [])
    tokens = ChatGPTTokens(
        access_token="access", refresh_token="refresh", id_token="id",
        expires_at=int(time.time()) + (3600 if operation == "refresh-fresh" else -1), account_id="account",
    )
    if proxy_value == "encrypted-empty":
        proxy_value = encrypt_value_helper("")
    elif proxy_value == "encrypted-invalid":
        proxy_value = encrypt_value_helper("not-a-proxy")
    row = SimpleNamespace(
        credential_name="subscription",
        credential_info={"provider": "chatgpt", "auth_type": "oauth", "proxy_configured": True},
        credential_values={
            CHATGPT_CREDENTIAL_VALUE_KEY: encrypt_value_helper(tokens.to_json()),
            **({CREDENTIAL_PROXY_VALUE_KEY: proxy_value} if proxy_value is not None else {}),
        },
    )
    transaction = MagicMock()
    transaction.execute_raw = AsyncMock()
    valid_proxy = "http://proxy.invalid:8080"
    valid_row = SimpleNamespace(
        credential_name=row.credential_name, credential_info=row.credential_info,
        credential_values={**row.credential_values, CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(valid_proxy)},
    )
    transaction.litellm_credentialstable.find_unique = AsyncMock(
        side_effect=[valid_row, row] if operation.endswith("latest") else None,
        return_value=row,
    )
    transaction.litellm_credentialstable.update = AsyncMock()
    transaction.litellm_credentialstable.create = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=transaction)
    context.__aexit__ = AsyncMock(return_value=None)
    prisma = MagicMock()
    prisma.db.tx.return_value = context
    client = MagicMock()
    client.async_refresh = AsyncMock(return_value=tokens)
    hook = ChatGPTOAuthCredentialHook(oauth_client=client)
    prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=vars(row))
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.credential_endpoints.chatgpt_oauth.ChatGPTOAuthClient", return_value=client),
    ):
        with pytest.raises((ValueError, TypeError)):
            if operation == "load":
                await hook._find_or_load_credential("subscription")
            elif operation.startswith("store"):
                await _store_tokens("subscription", tokens, "admin", None,
                    valid_proxy if operation.endswith("latest") else None)
            else:
                await hook._refresh_tokens("subscription")
    if operation != "refresh-latest":
        client.async_refresh.assert_not_awaited()
    transaction.litellm_credentialstable.update.assert_not_awaited()
    transaction.litellm_credentialstable.create.assert_not_awaited()
    assert litellm.credential_list == []
    assert row.credential_info["proxy_configured"] is True


def _refresh_database(monkeypatch, tokens, info=None):
    from copy import deepcopy

    monkeypatch.setenv("LITELLM_SALT_KEY", "chatgpt-refresh-regression-key")
    monkeypatch.setattr(litellm, "credential_list", [_DatabaseChatGPTCredential(
        credential_name="subscription",
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )])

    class Database:
        def __init__(self):
            self.row = CredentialItem(
                credential_name="subscription",
                credential_info={"provider": "chatgpt", "auth_type": "oauth", **(info or {})},
                credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: encrypt_value_helper(tokens.to_json()), "other": "untouched"},
            )
            self.lock = asyncio.Lock()
            self.commits = 0
            self.options = []
            self.queries = []
            self.reads = 0

        def tx(self, **kwargs):
            self.options.append(kwargs)

            @asynccontextmanager
            async def context():
                async with self.lock:
                    database = self
                    snapshot = deepcopy(self.row)
                    self.active_row = snapshot

                    class Table:
                        async def find_unique(self, *, where):
                            database.reads += 1
                            return snapshot

                        async def update(self, *, where, data):
                            for key in ("credential_info", "credential_values"):
                                if key in data:
                                    setattr(snapshot, key, json.loads(data[key]))

                    class Transaction:
                        litellm_credentialstable = Table()

                        async def execute_raw(self, query, *args):
                            database.queries.append((query, args))

                    yield Transaction()
                    self.row = snapshot
                    self.commits += 1

            return context()

    database = Database()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database))
    monkeypatch.setattr(
        "litellm.proxy.credential_endpoints.chatgpt_oauth.publish_config_change_for_object_type", AsyncMock()
    )
    return database


def _persisted_tokens(database):
    return ChatGPTTokens.from_json(decrypt_value_helper(
        database.row.credential_values[CHATGPT_CREDENTIAL_VALUE_KEY], CHATGPT_CREDENTIAL_VALUE_KEY
    ))


def test_recovery_singleflight_can_be_reused_across_sync_event_loops(monkeypatch):
    old = ChatGPTTokens("old-access", "old-refresh", "old-id", 1, "account")
    fresh = ChatGPTTokens("new-access", "new-refresh", "old-id", int(time.time()) + 3600, "account")

    class OAuthClient:
        async def async_refresh(self, previous):
            assert previous == old
            await asyncio.sleep(0)
            return fresh

    hook = ChatGPTOAuthCredentialHook(OAuthClient())

    async def wave():
        database = _refresh_database(monkeypatch, old)
        recovered = await asyncio.gather(*(
            hook.recover_after_unauthorized("subscription", old.access_token) for _ in range(3)
        ))
        assert recovered == [fresh] * 3
        assert _persisted_tokens(database) == fresh
        assert database.commits == 1

    asyncio.run(wave())
    asyncio.run(wave())


@pytest.mark.asyncio
async def test_unauthorized_recovery_forces_unexpired_refresh_and_singleflights_across_hooks(monkeypatch):
    import httpx

    from litellm.llms.chatgpt.oauth_client import CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS, ChatGPTOAuthClient
    from litellm.proxy.credential_endpoints.chatgpt_oauth import CHATGPT_REFRESH_LOCK_WAIT_SECONDS

    old = ChatGPTTokens("old-access", "old-refresh", "old-id", int(time.time()) + 3600, "account")
    database = _refresh_database(monkeypatch, old)
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        await asyncio.sleep(0)
        return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = ChatGPTOAuthClient(async_http_client=transport)
        first = ChatGPTOAuthCredentialHook(client)
        second = ChatGPTOAuthCredentialHook(client)
        recovered = await asyncio.gather(
            first.recover_after_unauthorized("subscription", old.access_token),
            first.recover_after_unauthorized("subscription", old.access_token),
            second.recover_after_unauthorized("subscription", old.access_token),
        )

    expected = ChatGPTTokens("new-access", "new-refresh", old.id_token, None, old.account_id)
    assert recovered == [expected] * 3
    assert _persisted_tokens(database) == expected
    assert len(requests) == 1
    assert requests[0]["refresh_token"] == old.refresh_token
    assert database.row.credential_values["other"] == "untouched"
    assert all(
        option["timeout"].total_seconds() > CHATGPT_REFRESH_HTTP_TIMEOUT_SECONDS + CHATGPT_REFRESH_LOCK_WAIT_SECONDS
        for option in database.options
    )
    assert any("lock_timeout" in query for query, args in database.queries)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error,permanent", [
    (400, "invalid_grant", True), (401, "refresh_token_reused", True),
    (401, "unauthorized", False), (429, "invalid_grant", False), (503, "invalid_grant", False),
])
async def test_refresh_rejection_commits_token_scoped_block_but_transient_errors_retry(monkeypatch, status, error, permanent):
    import hashlib

    import httpx

    from litellm.llms.chatgpt.oauth_client import ChatGPTOAuthClient, ChatGPTRefreshError
    from litellm.proxy.credential_endpoints.chatgpt_oauth import CHATGPT_REAUTH_REQUIRED_TOKEN_KEY

    old = ChatGPTTokens("old-access", "old-refresh", "old-id", int(time.time()) + 3600, "account")
    database = _refresh_database(monkeypatch, old)
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": error, "error_description": "never expose this secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = ChatGPTOAuthClient(async_http_client=transport)
        for _ in range(2):
            hook = ChatGPTOAuthCredentialHook(client)
            monkeypatch.setattr(litellm, "credential_list", [])
            if permanent:
                assert await hook.recover_after_unauthorized("subscription", old.access_token) is None
            else:
                with pytest.raises(ChatGPTRefreshError) as caught:
                    await hook.recover_after_unauthorized("subscription", old.access_token)
                assert caught.value.reason is None
                assert "secret" not in str(caught.value)

        assert len(requests) == (1 if permanent else 2)
        assert _persisted_tokens(database) == old
        if permanent:
            assert database.commits == 1
            assert database.row.credential_info[CHATGPT_REAUTH_REQUIRED_TOKEN_KEY] == hashlib.sha256(old.refresh_token.encode()).hexdigest()
            assert "disabled" not in database.row.credential_info
            new = ChatGPTTokens("other-access", "other-refresh", old.id_token, old.expires_at, old.account_id)
            database.row.credential_values[CHATGPT_CREDENTIAL_VALUE_KEY] = encrypt_value_helper(new.to_json())
            assert await ChatGPTOAuthCredentialHook(client).recover_after_unauthorized("subscription", new.access_token) is None
            assert len(requests) == 2
        else:
            assert CHATGPT_REAUTH_REQUIRED_TOKEN_KEY not in database.row.credential_info
            assert database.commits == 0


@pytest.mark.asyncio
async def test_cancelled_recovery_still_persists_rotated_tokens_before_next_recovery(monkeypatch):
    import httpx

    from litellm.llms.chatgpt.oauth_client import ChatGPTOAuthClient

    old = ChatGPTTokens("old-access", "old-refresh", "old-id", int(time.time()) + 3600, "account")
    database = _refresh_database(monkeypatch, old)
    started = asyncio.Event()
    finish = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        started.set()
        await finish.wait()
        return httpx.Response(200, json={"access_token": "rotated-access", "refresh_token": "rotated-refresh"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        hook = ChatGPTOAuthCredentialHook(ChatGPTOAuthClient(async_http_client=transport))
        caller = asyncio.create_task(hook.recover_after_unauthorized("subscription", old.access_token))
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        finish.set()
        recovered = await hook.recover_after_unauthorized("subscription", old.access_token)

    assert recovered == ChatGPTTokens("rotated-access", "rotated-refresh", old.id_token, None, old.account_id)
    assert _persisted_tokens(database) == recovered
    assert len(requests) == 1
    assert database.commits == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_rejected", [False, True])
@pytest.mark.parametrize("concurrent_change", ["reconnect", "proxy-edit", "auth-type"])
async def test_refresh_preserves_concurrent_reconnect_proxy_and_auth_type_changes(monkeypatch, refresh_rejected, concurrent_change):
    import httpx

    from litellm.llms.chatgpt.oauth_client import ChatGPTOAuthClient
    from litellm.proxy.credential_endpoints.chatgpt_oauth import CHATGPT_REAUTH_REQUIRED_TOKEN_KEY

    old = ChatGPTTokens("old-access", "old-refresh", "old-id", 1, "account")
    newer = ChatGPTTokens("reconnected-access", "reconnected-refresh", "reconnected-id", int(time.time()) + 3600, "account")
    database = _refresh_database(monkeypatch, old)
    saved_proxy = "http://saved-proxy.invalid:8080"

    async def handler(request):
        database.active_row.credential_values["other"] = "concurrent-edit"
        database.active_row.credential_info["custom"] = "concurrent-edit"
        if concurrent_change == "reconnect":
            database.active_row.credential_values[CHATGPT_CREDENTIAL_VALUE_KEY] = encrypt_value_helper(newer.to_json())
        elif concurrent_change == "proxy-edit":
            database.active_row.credential_values[CREDENTIAL_PROXY_VALUE_KEY] = encrypt_value_helper(saved_proxy)
            database.active_row.credential_info["proxy_configured"] = True
        else:
            database.active_row.credential_info["auth_type"] = "api_key"
        return httpx.Response(400, json={"error": "invalid_grant"}) if refresh_rejected else httpx.Response(
            200, json={"access_token": "refreshed-access", "refresh_token": "refreshed-refresh"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        hook = ChatGPTOAuthCredentialHook(ChatGPTOAuthClient(async_http_client=transport))
        if concurrent_change == "auth-type":
            with pytest.raises(ValueError, match="Credential changed"):
                await hook.recover_after_unauthorized("subscription", old.access_token)
            assert database.commits == 0
            assert _persisted_tokens(database) == old
            return
        recovered = await hook.recover_after_unauthorized("subscription", old.access_token)

    assert database.row.credential_values["other"] == "concurrent-edit"
    assert database.row.credential_info["custom"] == "concurrent-edit"
    if concurrent_change == "reconnect":
        assert recovered == newer
        assert _persisted_tokens(database) == newer
        assert CHATGPT_REAUTH_REQUIRED_TOKEN_KEY not in database.row.credential_info
    else:
        assert decrypt_value_helper(database.row.credential_values[CREDENTIAL_PROXY_VALUE_KEY], CREDENTIAL_PROXY_VALUE_KEY) == saved_proxy
        assert database.row.credential_info["proxy_configured"] is True
        if refresh_rejected:
            assert recovered is None
            assert CHATGPT_REAUTH_REQUIRED_TOKEN_KEY in database.row.credential_info
        else:
            assert recovered == ChatGPTTokens("refreshed-access", "refreshed-refresh", old.id_token, None, old.account_id)
            assert _persisted_tokens(database) == recovered
