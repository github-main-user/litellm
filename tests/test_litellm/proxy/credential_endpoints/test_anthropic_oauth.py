import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.oauth_client import (
    ANTHROPIC_OAUTH_REDIRECT_URI,
    AnthropicOAuthError,
    AnthropicOAuthTokens,
)
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.credential_endpoints.anthropic_oauth import (
    ANTHROPIC_CREDENTIAL_VALUE_KEY,
    AnthropicOAuthAttempt,
    AnthropicOAuthCompleteRequest,
    AnthropicOAuthCredentialHook,
    AnthropicOAuthStartRequest,
    ParsedAuthorizationCode,
    _complete_and_store,
    _decode_attempt,
    _encode_attempt,
    _parse_authorization_code,
    complete_anthropic_oauth,
    start_anthropic_oauth,
)


def _credential(name: str, tokens: AnthropicOAuthTokens, auth_type: str = "oauth") -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_info={"provider": "anthropic", "auth_type": auth_type},
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def test_attempt_is_actor_bound_and_expiring() -> None:
    attempt = AnthropicOAuthAttempt(
        credential_name="account-a",
        actor="admin-a",
        state="state-a",
        code_verifier="verifier-a",
        expires_at=time.time() + 60,
    )
    with (
        patch("litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper", side_effect=lambda value: value),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
            side_effect=lambda value, *args, **kwargs: value,
        ),
    ):
        token = _encode_attempt(attempt)
        assert _decode_attempt(token, "admin-a") == attempt
        with pytest.raises(HTTPException) as wrong_actor:
            _decode_attempt(token, "admin-b")
        expired = attempt.model_copy(update={"expires_at": time.time() - 1})
        with pytest.raises(HTTPException) as expired_error:
            _decode_attempt(_encode_attempt(expired), "admin-a")

    assert wrong_actor.value.status_code == 400
    assert expired_error.value.status_code == 400


def test_authorization_response_requires_matching_state() -> None:
    assert _parse_authorization_code("code-a#state-a", "state-a").code == "code-a"
    assert _parse_authorization_code(
        f"{ANTHROPIC_OAUTH_REDIRECT_URI}?code=code-b&state=state-a", "state-a"
    ).code == "code-b"
    for invalid in (
        "bare-code",
        "code-a#wrong",
        "http://localhost:54545/callback?code=code-a&state=state-a",
        f"{ANTHROPIC_OAUTH_REDIRECT_URI}?code=code-a",
    ):
        with pytest.raises(HTTPException):
            _parse_authorization_code(invalid, "state-a")


@pytest.mark.parametrize(
    "value",
    (
        "code#státe",
        "code#\ud800",
        "https://[invalid",
        f"{ANTHROPIC_OAUTH_REDIRECT_URI};ignored?code=code&state=state",
        f"{ANTHROPIC_OAUTH_REDIRECT_URI}?code=code&code=&state=state",
        f"{ANTHROPIC_OAUTH_REDIRECT_URI}?code=code&state=wrong#state=state",
        f"{ANTHROPIC_OAUTH_REDIRECT_URI}?code=old&state=state#code=code",
    ),
)
def test_authorization_response_rejects_ambiguous_or_malformed_values(value: str) -> None:
    with pytest.raises(HTTPException) as error:
        _parse_authorization_code(value, "state")
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_start_is_admin_only_and_validates_name() -> None:
    with pytest.raises(HTTPException) as denied:
        await start_anthropic_oauth(
            AnthropicOAuthStartRequest(credential_name="account-a"),
            UserAPIKeyAuth(user_id="user-a", user_role=LitellmUserRoles.INTERNAL_USER),
        )
    with pytest.raises(HTTPException) as invalid_name:
        await start_anthropic_oauth(
            AnthropicOAuthStartRequest(credential_name="bad/name"),
            UserAPIKeyAuth(user_id="admin-a", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert denied.value.status_code == 403
    assert invalid_name.value.status_code == 422


@pytest.mark.asyncio
async def test_complete_is_admin_only_and_rejects_malformed_or_expired_attempts() -> None:
    admin = UserAPIKeyAuth(user_id="admin-a", user_role=LitellmUserRoles.PROXY_ADMIN)
    non_admin = UserAPIKeyAuth(user_id="user-a", user_role=LitellmUserRoles.INTERNAL_USER)
    payload = AnthropicOAuthCompleteRequest(attempt_token="not-ciphertext", authorization_code="code#state")

    with pytest.raises(HTTPException) as denied:
        await complete_anthropic_oauth(payload, non_admin)
    with pytest.raises(HTTPException) as malformed:
        await complete_anthropic_oauth(payload, admin)

    expired = AnthropicOAuthAttempt(
        credential_name="subscription",
        actor="admin-a",
        state="state",
        code_verifier="verifier",
        expires_at=time.time() - 1,
    )
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "test-only-anthropic-state-key"}):
        expired_payload = payload.model_copy(update={"attempt_token": _encode_attempt(expired)})
        with pytest.raises(HTTPException) as expired_error:
            await complete_anthropic_oauth(expired_payload, admin)

    assert denied.value.status_code == 403
    assert malformed.value.status_code == 400
    assert expired_error.value.status_code == 400


@pytest.mark.asyncio
async def test_completion_encrypts_storage_and_rejects_collisions() -> None:
    tokens = AnthropicOAuthTokens("sk-ant-oat-new", "refresh-new", time.time() + 3600, "account-a")
    created: list[dict[str, object]] = []

    class Table:
        def __init__(self, row):
            self.row = row

        async def find_unique(self, where):
            return self.row

        async def create(self, data):
            created.append(data)

        async def update(self, where, data):
            created.append(data)

    class Transaction:
        def __init__(self, row):
            self.litellm_credentialstable = Table(row)

        async def execute_raw(self, query, key):
            return None

    class Database:
        def __init__(self, row):
            self.row = row

        def tx(self, timeout):
            @asynccontextmanager
            async def context():
                yield Transaction(self.row)

            return context()

    attempt = AnthropicOAuthAttempt(
        credential_name="subscription",
        actor="admin-a",
        state="state-a",
        code_verifier="verifier-a",
        expires_at=time.time() + 60,
    )
    oauth_client = MagicMock()
    oauth_client.exchange_code = AsyncMock(return_value=tokens)
    previous = litellm.credential_list
    litellm.credential_list = []
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database(None))),
            patch.dict("os.environ", {"LITELLM_SALT_KEY": "test-only-anthropic-storage-key"}),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                AsyncMock(),
            ),
        ):
            await _complete_and_store(
                attempt,
                ParsedAuthorizationCode(code="code-a", state="state-a"),
                oauth_client,
            )
    finally:
        litellm.credential_list = previous

    stored_values = created[0]["credential_values"]
    assert isinstance(stored_values, str)
    ciphertext = json.loads(stored_values)[ANTHROPIC_CREDENTIAL_VALUE_KEY]
    assert isinstance(ciphertext, str)
    assert tokens.access_token not in ciphertext
    assert tokens.refresh_token not in ciphertext
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "test-only-anthropic-storage-key"}):
        assert decrypt_value_helper(ciphertext, ANTHROPIC_CREDENTIAL_VALUE_KEY) == tokens.to_json()

    collision = SimpleNamespace(
        credential_values={},
        credential_info={"provider": "openai", "auth_type": "oauth"},
    )
    oauth_client.exchange_code.reset_mock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database(collision))),
        pytest.raises(HTTPException) as conflict,
    ):
        await _complete_and_store(
            attempt,
            ParsedAuthorizationCode(code="code-a", state="state-a"),
            oauth_client,
        )
    assert conflict.value.status_code == 409
    oauth_client.exchange_code.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_rejects_account_mismatch() -> None:
    old = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", time.time() + 3600, "account-a")
    new = AnthropicOAuthTokens("sk-ant-oat-new", "refresh-new", time.time() + 3600, "account-b")
    row = SimpleNamespace(
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: old.to_json()},
        credential_info={"provider": "anthropic", "auth_type": "oauth"},
    )

    class Database:
        def tx(self, timeout):
            @asynccontextmanager
            async def context():
                transaction = MagicMock()
                transaction.execute_raw = AsyncMock()
                transaction.litellm_credentialstable.find_unique = AsyncMock(return_value=row)
                yield transaction

            return context()

    oauth_client = MagicMock()
    oauth_client.exchange_code = AsyncMock(return_value=new)
    attempt = AnthropicOAuthAttempt(
        credential_name="subscription",
        actor="admin-a",
        state="state-a",
        code_verifier="verifier-a",
        expires_at=time.time() + 60,
    )
    with (
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database())),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
            side_effect=lambda value, *args, **kwargs: value,
        ),
        pytest.raises(HTTPException) as conflict,
    ):
        await _complete_and_store(
            attempt,
            ParsedAuthorizationCode(code="code-a", state="state-a"),
            oauth_client,
        )

    assert conflict.value.status_code == 409


@pytest.mark.asyncio
async def test_hook_selects_managed_accounts_and_bypasses_regular_api_keys() -> None:
    now = time.time() + 3600
    previous = litellm.credential_list
    litellm.credential_list = [
        _credential("account-a", AnthropicOAuthTokens("sk-ant-oat-a", "refresh-a", now, "id-a")),
        _credential("account-b", AnthropicOAuthTokens("sk-ant-oat-b", "refresh-b", now, "id-b")),
        CredentialItem(
            credential_name="api-key",
            credential_info={"provider": "anthropic", "auth_type": "api_key"},
            credential_values={"api_key": "sk-ant-api-key"},
        ),
    ]
    try:
        hook = AnthropicOAuthCredentialHook()
        result = await hook.async_pre_call_deployment_hook(
            {"model": "anthropic/claude", "litellm_credential_name": "account-b"}, None
        )
        bypassed = await hook.async_pre_call_deployment_hook(
            {"model": "anthropic/claude", "litellm_credential_name": "api-key"}, None
        )
        with pytest.raises(AuthenticationError):
            await hook.async_pre_call_deployment_hook(
                {"model": "anthropic/claude", "litellm_credential_name": "missing"}, None
            )
    finally:
        litellm.credential_list = previous

    assert result is not None
    assert result["api_key"] == "sk-ant-oat-b"
    assert result["api_base"] == "https://api.anthropic.com"
    assert bypassed is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authorization_input",
    [
        {"extra_headers": {"Authorization": "Bearer caller-token"}},
        {"headers": {"authorization": "Bearer caller-token"}},
        {"additional_headers": {"AUTHORIZATION": "Bearer caller-token"}},
        {
            "provider_specific_header": {
                "custom_llm_provider": "anthropic",
                "extra_headers": {"Authorization": "Bearer caller-token"},
            }
        },
    ],
)
async def test_hook_rejects_caller_authorization_from_every_header_shape(
    authorization_input: dict[str, object],
) -> None:
    tokens = AnthropicOAuthTokens("sk-ant-oat-a", "refresh-a", time.time() + 3600, "id-a")
    previous = litellm.credential_list
    litellm.credential_list = [_credential("account-a", tokens)]
    try:
        with pytest.raises(AuthenticationError):
            await AnthropicOAuthCredentialHook().async_pre_call_deployment_hook(
                {
                    "model": "anthropic/claude",
                    "litellm_credential_name": "account-a",
                    **authorization_input,
                },
                None,
            )
    finally:
        litellm.credential_list = previous


@pytest.mark.asyncio
async def test_hook_rejects_wrong_provider_and_nonofficial_api_base() -> None:
    tokens = AnthropicOAuthTokens("sk-ant-oat-a", "refresh-a", time.time() + 3600, "id-a")
    previous = litellm.credential_list
    litellm.credential_list = [_credential("account-a", tokens)]
    hook = AnthropicOAuthCredentialHook()
    try:
        with pytest.raises(AuthenticationError):
            await hook.async_pre_call_deployment_hook(
                {
                    "model": "anthropic/claude",
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "account-a",
                },
                None,
            )
        with pytest.raises(AuthenticationError):
            await hook.async_pre_call_deployment_hook(
                {
                    "model": "anthropic/claude",
                    "custom_llm_provider": "anthropic",
                    "litellm_credential_name": "account-a",
                    "api_base": "https://example.invalid",
                },
                None,
            )
    finally:
        litellm.credential_list = previous


class _AdvisoryLockDatabase:
    """only execute_raw acquires shared DB locks, not transaction entry."""

    def __init__(self, row: SimpleNamespace) -> None:
        self.row = row
        self.locks: dict[int, asyncio.Lock] = {}

    def tx(self, timeout):
        database = self

        class Table:
            async def find_unique(self, where):
                # return a snapshot, as independent database sessions do.
                return SimpleNamespace(
                    credential_values=dict(database.row.credential_values),
                    credential_info=dict(database.row.credential_info),
                )

            async def update(self, where, data):
                database.row.credential_values = json.loads(data["credential_values"])
                database.row.credential_info = json.loads(data["credential_info"])

            async def create(self, data):
                database.row = SimpleNamespace(
                    credential_values=json.loads(data["credential_values"]),
                    credential_info=json.loads(data["credential_info"]),
                )

        class Transaction:
            def __init__(self) -> None:
                self.litellm_credentialstable = Table()
                self.acquired: list[asyncio.Lock] = []
                self.acquired_keys: set[int] = set()

            async def execute_raw(self, query, key):
                if key in self.acquired_keys:
                    return
                lock = database.locks.setdefault(key, asyncio.Lock())
                await lock.acquire()
                self.acquired.append(lock)
                self.acquired_keys.add(key)

        @asynccontextmanager
        async def context():
            transaction = Transaction()
            try:
                yield transaction
            finally:
                for lock in reversed(transaction.acquired):
                    lock.release()

        return context()


@pytest.mark.asyncio
async def test_shared_database_lock_prevents_cross_worker_refresh_race() -> None:
    expired = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", time.time() - 1, "account-a")
    fresh = AnthropicOAuthTokens("sk-ant-oat-new", "refresh-new", time.time() + 3600, "account-a")
    row = SimpleNamespace(
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: expired.to_json()},
        credential_info={"provider": "anthropic", "auth_type": "oauth"},
    )
    database = _AdvisoryLockDatabase(row)

    async def refresh(tokens: AnthropicOAuthTokens) -> AnthropicOAuthTokens:
        # ensure an implementation without the advisory lock lets both workers
        # read the expired snapshot before either one writes.
        await asyncio.sleep(0.01)
        return fresh

    oauth_client = MagicMock()
    oauth_client.refresh = AsyncMock(side_effect=refresh)
    prisma_client = SimpleNamespace(db=database)
    first_hook = AnthropicOAuthCredentialHook(oauth_client)
    second_hook = AnthropicOAuthCredentialHook(oauth_client)
    previous = litellm.credential_list
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper",
                side_effect=lambda value: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                AsyncMock(),
            ),
        ):
            first, second = await asyncio.gather(
                first_hook._refresh_tokens("subscription", "account-a"),
                second_hook._refresh_tokens("subscription", "account-a"),
            )
    finally:
        litellm.credential_list = previous

    assert first == fresh
    assert second == fresh
    oauth_client.refresh.assert_awaited_once_with(expired)


@pytest.mark.asyncio
async def test_reconnect_and_refresh_share_the_same_database_lock() -> None:
    expired = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", time.time() - 1, "account-a")
    refreshed = AnthropicOAuthTokens("sk-ant-oat-refreshed", "refresh-new", time.time() + 3600, "account-a")
    reconnected = AnthropicOAuthTokens("sk-ant-oat-reconnected", "refresh-reconnected", time.time() + 3600, "account-a")
    database = _AdvisoryLockDatabase(
        SimpleNamespace(
            credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: expired.to_json()},
            credential_info={"provider": "anthropic", "auth_type": "oauth"},
        )
    )
    active_requests = 0
    maximum_active_requests = 0

    async def network_result(result: AnthropicOAuthTokens) -> AnthropicOAuthTokens:
        nonlocal active_requests, maximum_active_requests
        active_requests += 1
        maximum_active_requests = max(maximum_active_requests, active_requests)
        await asyncio.sleep(0.01)
        active_requests -= 1
        return result

    async def refresh_request(_: AnthropicOAuthTokens) -> AnthropicOAuthTokens:
        return await network_result(refreshed)

    async def exchange_request(*args: object) -> AnthropicOAuthTokens:
        return await network_result(reconnected)

    client = MagicMock()
    client.refresh = AsyncMock(side_effect=refresh_request)
    client.exchange_code = AsyncMock(side_effect=exchange_request)
    attempt = AnthropicOAuthAttempt(
        credential_name="subscription",
        actor="admin-a",
        state="state-a",
        code_verifier="verifier-a",
        expires_at=time.time() + 60,
    )
    previous = litellm.credential_list
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database)),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper",
                side_effect=lambda value: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                AsyncMock(),
            ),
        ):
            await asyncio.gather(
                AnthropicOAuthCredentialHook(client)._refresh_tokens("subscription", "account-a"),
                _complete_and_store(
                    attempt,
                    ParsedAuthorizationCode(code="code-a", state="state-a"),
                    client,
                ),
            )
    finally:
        litellm.credential_list = previous

    assert maximum_active_requests == 1


@pytest.mark.asyncio
async def test_recovery_refreshes_rejected_fresh_token_once_across_workers() -> None:
    rejected = AnthropicOAuthTokens("sk-ant-oat-rejected", "refresh-old", time.time() + 3600, "account-a")
    refreshed = AnthropicOAuthTokens("sk-ant-oat-recovered", "refresh-new", time.time() + 3600, "account-a")
    database = _AdvisoryLockDatabase(
        SimpleNamespace(
            credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: rejected.to_json()},
            credential_info={"provider": "anthropic", "auth_type": "oauth"},
        )
    )

    async def refresh_request(_: AnthropicOAuthTokens) -> AnthropicOAuthTokens:
        await asyncio.sleep(0.01)
        return refreshed

    client = MagicMock()
    client.refresh = AsyncMock(side_effect=refresh_request)
    previous = litellm.credential_list
    litellm.credential_list = [_credential("subscription", rejected)]
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database)),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper",
                side_effect=lambda value: value,
            ),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                AsyncMock(),
            ),
        ):
            recovered = await asyncio.gather(
                AnthropicOAuthCredentialHook(client).recover_rejected_token(
                    "subscription", rejected.access_token
                ),
                AnthropicOAuthCredentialHook(client).recover_rejected_token(
                    "subscription", rejected.access_token
                ),
            )
    finally:
        litellm.credential_list = previous

    assert recovered == [refreshed.access_token, refreshed.access_token]
    client.refresh.assert_awaited_once_with(rejected)


@pytest.mark.asyncio
async def test_recovery_ignores_api_keys_and_preserves_transient_refresh_error() -> None:
    rejected = AnthropicOAuthTokens("sk-ant-oat-rejected", "refresh-old", time.time() + 3600, "account-a")
    row = SimpleNamespace(
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: rejected.to_json()},
        credential_info={"provider": "anthropic", "auth_type": "oauth"},
    )
    client = MagicMock()
    transient = AnthropicOAuthError("token refresh", 503, "9")
    client.refresh = AsyncMock(side_effect=transient)
    previous = litellm.credential_list
    litellm.credential_list = [
        _credential("subscription", rejected),
        CredentialItem(
            credential_name="api-key",
            credential_info={"provider": "anthropic", "auth_type": "api_key"},
            credential_values={"api_key": "sk-ant-api-key"},
        ),
    ]
    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_AdvisoryLockDatabase(row))),
            patch(
                "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
                side_effect=lambda value, *args, **kwargs: value,
            ),
        ):
            hook = AnthropicOAuthCredentialHook(client)
            assert await hook.recover_rejected_token("api-key", "sk-ant-oat-unrelated") is None
            with pytest.raises(AnthropicOAuthError) as caught:
                await hook.recover_rejected_token("subscription", rejected.access_token)
    finally:
        litellm.credential_list = previous

    assert caught.value is transient
    assert caught.value.response_headers == {"Retry-After": "9"}
    client.refresh.assert_awaited_once_with(rejected)
