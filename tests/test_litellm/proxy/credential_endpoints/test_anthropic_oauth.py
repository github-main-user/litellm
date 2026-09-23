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
    AnthropicAuthorization,
    AnthropicOAuthError,
    AnthropicOAuthTokens,
)
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.anthropic_oauth import (
    ANTHROPIC_CREDENTIAL_VALUE_KEY,
    ANTHROPIC_REFRESH_BLOCKED_TOKEN_KEY,
    ANTHROPIC_REFRESH_BLOCKED_UNTIL_KEY,
    CREDENTIAL_PROXY_VALUE_KEY,
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
async def test_start_complete_reconnect_supports_maximum_proxy_snapshots() -> None:
    old_prefix, old_suffix = "http://user:", "@old.example"
    new_prefix, new_suffix = "socks5://user:", "@new.example"
    old_proxy = old_prefix + "a" * (4096 - len(old_prefix) - len(old_suffix)) + old_suffix
    new_proxy = new_prefix + "b" * (4096 - len(new_prefix) - len(new_suffix)) + new_suffix
    old_tokens = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", time.time() + 3600, "account-a")
    new_tokens = AnthropicOAuthTokens("sk-ant-oat-new", "refresh-new", time.time() + 3600, "account-a")
    proxies: list[str | None] = []
    updated: list[dict[str, object]] = []

    class OAuthClient:
        def __init__(self, *, proxy_url=None):
            proxies.append(proxy_url)

        def begin_authorization(self):
            return AnthropicAuthorization("https://claude.com/authorize", "state", "verifier")

        async def exchange_code(self, code, state, code_verifier):
            return new_tokens

    class Table:
        async def find_unique(self, *, where):
            return row

        async def update(self, *, where, data):
            updated.append(data)

    class Transaction:
        litellm_credentialstable = Table()

        async def execute_raw(self, query, *parameters):
            return None

    class Database:
        def tx(self, *, timeout):
            @asynccontextmanager
            async def context():
                yield Transaction()

            return context()

    admin = UserAPIKeyAuth(user_id="admin-a", user_role=LitellmUserRoles.PROXY_ADMIN)
    previous = litellm.credential_list
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "anthropic-proxy-lifecycle-key"}):
        row = SimpleNamespace(
            credential_info={"provider": "anthropic", "auth_type": "oauth", "custom": "preserved"},
            credential_values={
                ANTHROPIC_CREDENTIAL_VALUE_KEY: encrypt_value_helper(old_tokens.to_json()),
                CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(old_proxy),
                "other": "preserved",
            },
        )
        litellm.credential_list = [
            CredentialItem(
                credential_name="subscription",
                credential_info={"provider": "anthropic", "auth_type": "oauth", "proxy_configured": True},
                credential_values={
                    ANTHROPIC_CREDENTIAL_VALUE_KEY: old_tokens.to_json(),
                    CREDENTIAL_PROXY_VALUE_KEY: old_proxy,
                },
            )
        ]
        try:
            with (
                patch("litellm.proxy.credential_endpoints.anthropic_oauth.AnthropicOAuthClient", OAuthClient),
                patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database())),
                patch(
                    "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                    AsyncMock(),
                ),
            ):
                started = await start_anthropic_oauth(
                    AnthropicOAuthStartRequest(credential_name="subscription", proxy_url=new_proxy), admin
                )
                assert old_proxy not in started.attempt_token
                assert new_proxy not in started.attempt_token
                request = AnthropicOAuthCompleteRequest(
                    attempt_token=started.attempt_token,
                    authorization_code="code#state",
                )
                completed = await complete_anthropic_oauth(request, admin)
        finally:
            litellm.credential_list = previous

        values = updated[0]["credential_values"]
        if isinstance(values, str):
            values = json.loads(values)
        assert isinstance(values, dict)
        assert values["other"] == "preserved"
        assert decrypt_value_helper(values[CREDENTIAL_PROXY_VALUE_KEY], CREDENTIAL_PROXY_VALUE_KEY) == new_proxy
        stored_tokens = AnthropicOAuthTokens.from_json(
            decrypt_value_helper(values[ANTHROPIC_CREDENTIAL_VALUE_KEY], ANTHROPIC_CREDENTIAL_VALUE_KEY)
        )
        assert stored_tokens.access_token == new_tokens.access_token
        assert stored_tokens.refresh_token == new_tokens.refresh_token
        assert stored_tokens.device_id == old_tokens.device_id

    assert completed.status == "connected"
    assert proxies == [new_proxy, new_proxy]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{"credential_name": "subscription"}, {"credential_name": "subscription", "proxy_url": None}],
)
async def test_start_reuses_saved_proxy_for_omitted_or_null(payload: dict[str, object]) -> None:
    saved_proxy = "http://saved.example:8080"
    seen: list[str | None] = []

    class OAuthClient:
        def __init__(self, *, proxy_url=None):
            seen.append(proxy_url)

        def begin_authorization(self):
            return AnthropicAuthorization("https://claude.com/authorize", "state", "verifier")

    previous = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="subscription",
            credential_info={"provider": "anthropic", "auth_type": "oauth", "proxy_configured": True},
            credential_values={CREDENTIAL_PROXY_VALUE_KEY: saved_proxy},
        )
    ]
    try:
        with (
            patch.dict("os.environ", {"LITELLM_SALT_KEY": "anthropic-proxy-reuse-key"}),
            patch("litellm.proxy.credential_endpoints.anthropic_oauth.AnthropicOAuthClient", OAuthClient),
        ):
            await start_anthropic_oauth(
                AnthropicOAuthStartRequest.model_validate(payload),
                UserAPIKeyAuth(user_id="admin-a", user_role=LitellmUserRoles.PROXY_ADMIN),
            )
    finally:
        litellm.credential_list = previous

    assert seen == [saved_proxy]


@pytest.mark.asyncio
async def test_completion_rejects_current_database_proxy_change_before_exchange() -> None:
    attempt = AnthropicOAuthAttempt(
        credential_name="subscription",
        actor="admin-a",
        proxy_url="http://attempt.example:8080",
        previous_proxy_url="http://old.example:8080",
        state="state",
        code_verifier="verifier",
        expires_at=time.time() + 60,
    )
    row = SimpleNamespace(
        credential_info={"provider": "anthropic", "auth_type": "oauth"},
        credential_values={CREDENTIAL_PROXY_VALUE_KEY: "http://changed.example:8080"},
    )

    class Database:
        def tx(self, *, timeout):
            @asynccontextmanager
            async def context():
                transaction = MagicMock()
                transaction.execute_raw = AsyncMock()
                transaction.litellm_credentialstable.find_unique = AsyncMock(return_value=row)
                yield transaction

            return context()

    oauth_client = MagicMock()
    oauth_client.exchange_code = AsyncMock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database())),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
            side_effect=lambda value, *args, **kwargs: value,
        ),
        pytest.raises(HTTPException) as caught,
    ):
        await _complete_and_store(
            attempt,
            ParsedAuthorizationCode(code="code", state="state"),
            oauth_client,
        )

    assert caught.value.status_code == 409
    oauth_client.exchange_code.assert_not_awaited()


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
async def test_rejected_token_recovery_uses_stored_proxy_and_preserves_concurrent_proxy_edit() -> None:
    old_proxy = "http://old.example:8080"
    new_proxy = "socks5://new.example:1080"
    rejected = AnthropicOAuthTokens("sk-ant-oat-rejected", "refresh-old", time.time() + 3600, "account-a")
    refreshed = AnthropicOAuthTokens("sk-ant-oat-refreshed", "refresh-new", time.time() + 3600, "account-a")
    seen_proxies: list[str | None] = []
    updates: list[dict[str, object]] = []

    class OAuthClient:
        def __init__(self, *, proxy_url=None):
            seen_proxies.append(proxy_url)

        async def refresh(self, previous):
            return refreshed

    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "anthropic-refresh-proxy-key"}):
        old_row = SimpleNamespace(
            credential_values={
                ANTHROPIC_CREDENTIAL_VALUE_KEY: encrypt_value_helper(rejected.to_json()),
                CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(old_proxy),
                "other": "preserved",
            },
            credential_info={"provider": "anthropic", "auth_type": "oauth"},
        )
        new_row = SimpleNamespace(
            credential_values={
                **old_row.credential_values,
                CREDENTIAL_PROXY_VALUE_KEY: encrypt_value_helper(new_proxy),
            },
            credential_info={"provider": "anthropic", "auth_type": "oauth", "edited": True},
        )

        class Table:
            calls = 0

            async def find_unique(self, *, where):
                self.calls += 1
                return old_row if self.calls == 1 else new_row

            async def update(self, *, where, data):
                updates.append(data)

        class Transaction:
            litellm_credentialstable = Table()

            async def execute_raw(self, query, *parameters):
                return None

        class Database:
            def tx(self, *, timeout):
                @asynccontextmanager
                async def context():
                    yield Transaction()

                return context()

        previous_cache = litellm.credential_list
        litellm.credential_list = [_credential("subscription", rejected)]
        try:
            with (
                patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=Database())),
                patch("litellm.proxy.credential_endpoints.anthropic_oauth.AnthropicOAuthClient", OAuthClient),
                patch(
                    "litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type",
                    AsyncMock(),
                ),
            ):
                recovered = await AnthropicOAuthCredentialHook(MagicMock()).recover_rejected_token(
                    "subscription", rejected.access_token
                )
        finally:
            litellm.credential_list = previous_cache

        values = json.loads(updates[0]["credential_values"])
        assert values[CREDENTIAL_PROXY_VALUE_KEY] == new_row.credential_values[CREDENTIAL_PROXY_VALUE_KEY]
        assert values["other"] == "preserved"
        assert json.loads(updates[0]["credential_info"])["edited"] is True

    assert recovered == refreshed.access_token
    assert seen_proxies == [old_proxy]


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


@pytest.mark.asyncio
async def test_refresh_429_blocks_same_refresh_token_across_requests() -> None:
    rejected = AnthropicOAuthTokens("sk-ant-oat-rejected", "refresh-old", time.time() + 3600, "account-a")
    database = _AdvisoryLockDatabase(
        SimpleNamespace(
            credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: rejected.to_json()},
            credential_info={"provider": "anthropic", "auth_type": "oauth"},
        )
    )
    client = MagicMock()
    client.refresh = AsyncMock(side_effect=AnthropicOAuthError("token refresh", 429, "120"))
    with (
        patch.object(litellm, "credential_list", [_credential("subscription", rejected)]),
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database)),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
            side_effect=lambda value, *args, **kwargs: value,
        ),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper", side_effect=lambda value: value
        ),
        patch("litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type", AsyncMock()),
    ):
        hook = AnthropicOAuthCredentialHook(client)
        with pytest.raises(AnthropicOAuthError) as first:
            await hook.recover_rejected_token("subscription", rejected.access_token)
        with pytest.raises(AnthropicOAuthError) as blocked:
            await hook.recover_rejected_token("subscription", rejected.access_token)

    assert first.value.retry_after == "120"
    assert 1 <= int(blocked.value.retry_after or "0") <= 120
    assert ANTHROPIC_REFRESH_BLOCKED_TOKEN_KEY in database.row.credential_info
    assert database.row.credential_info[ANTHROPIC_REFRESH_BLOCKED_UNTIL_KEY] > time.time()
    client.refresh.assert_awaited_once_with(rejected)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_account,expires_in", [("account-b", 3600), ("account-a", -1), ("account-a", 3600)])
async def test_recovery_rechecks_stored_account_and_expiry(stored_account: str, expires_in: int) -> None:
    rejected = AnthropicOAuthTokens("sk-ant-oat-rejected", "refresh-old", time.time() + 3600, "account-a")
    stored = AnthropicOAuthTokens("sk-ant-oat-rotated", "refresh-rotated", time.time() + expires_in, stored_account)
    refreshed = AnthropicOAuthTokens("sk-ant-oat-recovered", "refresh-new", time.time() + 3600, stored_account)
    database = _AdvisoryLockDatabase(
        SimpleNamespace(
            credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: stored.to_json()},
            credential_info={"provider": "anthropic", "auth_type": "oauth"},
        )
    )
    client = MagicMock()
    client.refresh = AsyncMock(return_value=refreshed)
    with (
        patch.object(litellm, "credential_list", [_credential("subscription", rejected)]),
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database)),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.decrypt_value_helper",
            side_effect=lambda value, *args, **kwargs: value,
        ),
        patch(
            "litellm.proxy.credential_endpoints.anthropic_oauth.encrypt_value_helper", side_effect=lambda value: value
        ),
        patch("litellm.proxy.credential_endpoints.anthropic_oauth.publish_config_change_for_object_type", AsyncMock()),
    ):
        hook = AnthropicOAuthCredentialHook(client)
        if stored_account != rejected.account_id:
            with pytest.raises(ValueError, match="account identity changed"):
                await hook.recover_rejected_token("subscription", rejected.access_token)
            client.refresh.assert_not_awaited()
        else:
            result = await hook.recover_rejected_token("subscription", rejected.access_token)
            if expires_in < 0:
                assert result == refreshed.access_token
                client.refresh.assert_awaited_once_with(stored)
            else:
                assert result == stored.access_token
                client.refresh.assert_not_awaited()
