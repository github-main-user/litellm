import asyncio
import json
from copy import deepcopy
from uuid import UUID, uuid4

import pytest

from litellm.llms.anthropic.common_utils import (
    ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT,
    prepare_anthropic_subscription_messages,
    prepare_anthropic_subscription_system,
)
from litellm.llms.anthropic.oauth_client import AnthropicOAuthClient, AnthropicOAuthTokens
from litellm.llms.anthropic.subscription_identity import SubscriptionIdentity, prepare_subscription_identity


@pytest.fixture(autouse=True)
async def flush_request_logging(isolate_litellm_state):
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    yield
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


def test_managed_identity_matches_session_header_and_preserves_customer_metadata():
    identity = SubscriptionIdentity(account_id="account-a", device_id="a" * 64)
    session_id = str(uuid4())
    body = {"metadata": {"user_id": json.dumps({"session_id": session_id, "customer": "alice"})}}
    headers = {"Authorization": "Bearer token"}
    original = deepcopy((body, headers))
    result, outgoing_headers = prepare_subscription_identity(body, headers, {}, identity)
    metadata = json.loads(result["metadata"]["user_id"])
    assert metadata == {
        "customer": "alice",
        "account_uuid": "account-a",
        "device_id": "a" * 64,
        "session_id": session_id,
    }
    assert outgoing_headers["x-claude-code-session-id"] == metadata["session_id"]
    assert outgoing_headers["Authorization"] == headers["Authorization"]
    assert (body, headers) == original
    repeated, repeated_headers = prepare_subscription_identity(result, outgoing_headers, {}, identity)
    assert (repeated, repeated_headers) == (result, outgoing_headers)


def test_identity_uses_selected_account_not_caller_supplied_account():
    identity = SubscriptionIdentity(account_id="selected-account", device_id="b" * 64)
    session_id = str(uuid4())
    body = {"metadata": {"user_id": json.dumps({"account_uuid": "spoofed", "device_id": "wrong"})}}
    result, headers = prepare_subscription_identity(body, {"X-Claude-Code-Session-Id": session_id}, {}, identity)
    metadata = json.loads(result["metadata"]["user_id"])
    assert metadata["account_uuid"] == identity.account_id
    assert metadata["device_id"] == identity.device_id
    assert metadata["session_id"] == session_id
    assert headers == {"x-claude-code-session-id": session_id}


def test_application_session_names_are_stable_and_scoped_to_device():
    params = {"metadata": {"session_id": "conversation-123"}}
    first = prepare_subscription_identity({}, {}, params, SubscriptionIdentity("a", "a" * 64))[1]
    repeated = prepare_subscription_identity({}, {}, params, SubscriptionIdentity("a", "a" * 64))[1]
    other_account = prepare_subscription_identity({}, {}, params, SubscriptionIdentity("b", "b" * 64))[1]
    assert first == repeated
    assert first != other_account
    UUID(first["x-claude-code-session-id"])


def test_trace_id_keeps_generated_session_stable_across_routed_attempts():
    identity = SubscriptionIdentity("account-a", "a" * 64)
    params = {"litellm_trace_id": "trace-for-one-logical-request"}

    first_body, first_headers = prepare_subscription_identity({}, {}, params, identity)
    second_body, second_headers = prepare_subscription_identity({}, {}, params, identity)

    assert first_headers == second_headers
    assert first_body == second_body
    assert json.loads(first_body["metadata"]["user_id"])["session_id"] == first_headers[
        "x-claude-code-session-id"
    ]


def test_identity_preserves_short_opaque_user_id_and_bounds_large_values():
    identity = SubscriptionIdentity("account-a", "a" * 64)
    short, _ = prepare_subscription_identity({"metadata": {"user_id": "customer-123"}}, {}, {}, identity)
    assert json.loads(short["metadata"]["user_id"])["client_user_id"] == "customer-123"
    oversized, _ = prepare_subscription_identity({"metadata": {"user_id": "x" * 500}}, {}, {}, identity)
    encoded = oversized["metadata"]["user_id"]
    assert len(encoded) <= 512
    assert json.loads(encoded)["account_uuid"] == identity.account_id
    assert "client_user_id_sha256" in json.loads(encoded)


def test_unmanaged_requests_do_not_gain_synthetic_identity():
    body = {"metadata": {"user_id": "original"}}
    headers = {"User-Agent": "original"}
    assert prepare_subscription_identity(body, headers, {}, None) == (body, headers)


def test_tokens_keep_device_identity_through_serialization_and_refresh():
    previous = AnthropicOAuthTokens("sk-ant-oat-old", "refresh-old", 123.0, "account-a")
    restored = AnthropicOAuthTokens.from_json(previous.to_json())
    refreshed = AnthropicOAuthClient._parse_tokens(
        {"access_token": "sk-ant-oat-new", "refresh_token": "refresh-new", "expires_in": 3600},
        restored,
        "refresh",
    )
    assert refreshed.device_id == restored.device_id == previous.device_id
    assert refreshed.account_id == previous.account_id
    assert refreshed.refresh_token == "refresh-new"
    assert len(previous.device_id) == 64


def test_legacy_credentials_acquire_repeatable_device_identity():
    encoded = json.dumps(
        {
            "access_token": "sk-ant-oat-legacy",
            "refresh_token": "refresh-legacy",
            "expires_at": 123.0,
        }
    )
    assert AnthropicOAuthTokens.from_json(encoded).device_id == AnthropicOAuthTokens.from_json(encoded).device_id


@pytest.mark.parametrize("device_id", ["", "x" * 64, "a" * 63, 123])
def test_malformed_stored_device_identity_is_rejected(device_id):
    encoded = json.dumps(
        {
            "access_token": "sk-ant-oat-legacy",
            "refresh_token": "refresh-legacy",
            "expires_at": 123.0,
            "device_id": device_id,
        }
    )
    with pytest.raises(ValueError, match="device identity"):
        AnthropicOAuthTokens.from_json(encoded)


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("oauth", [False, True])
def test_request_transformations_apply_identity_only_to_managed_oauth(monkeypatch, surface, oauth):
    import litellm
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
    from litellm.models.credentials import CredentialItem
    from litellm.types.router import GenericLiteLLMParams

    tokens = AnthropicOAuthTokens("sk-ant-oat-test", "refresh-test", 123.0, "selected-account")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="subscription",
                credential_info={"provider": "anthropic", "auth_type": "oauth"},
                credential_values={"litellm_internal_anthropic_auth_token": tokens.to_json()},
            )
        ],
    )
    headers = {"authorization": "Bearer sk-ant-oat-test"} if oauth else {"x-api-key": "sk-api-test"}
    optional = {"max_tokens": 32, "metadata": {"user_id": "customer-123"}}
    messages = [{"role": "user", "content": "Hello"}]
    if surface == "chat":
        body = AnthropicConfig().transform_request(
            model="claude-sonnet-5",
            messages=messages,
            optional_params=optional,
            litellm_params={"litellm_credential_name": "subscription"},
            headers=headers,
        )
    else:
        body = AnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-sonnet-5",
            messages=messages,
            anthropic_messages_optional_request_params=optional,
            litellm_params=GenericLiteLLMParams(litellm_credential_name="subscription"),
            headers=headers,
        )
    if oauth:
        metadata = json.loads(body["metadata"]["user_id"])
        assert metadata["account_uuid"] == tokens.account_id
        assert metadata["device_id"] == tokens.device_id
        assert metadata["client_user_id"] == "customer-123"
        assert metadata["session_id"] == headers["x-claude-code-session-id"]
    else:
        assert body["metadata"]["user_id"] == "customer-123"
        assert "x-claude-code-session-id" not in headers


def test_existing_billing_attribution_is_preserved_and_client_system_becomes_a_reminder():
    billing = {"type": "text", "text": "x-anthropic-billing-header: original-attribution;"}
    instruction = {"type": "text", "text": "Keep this instruction", "cache_control": {"type": "ephemeral"}}
    original = [instruction, billing]
    prepared = prepare_anthropic_subscription_system(original)
    messages = prepare_anthropic_subscription_messages([{"role": "user", "content": "Hello"}], original)
    assert prepared == [billing, {"type": "text", "text": ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT}]
    assert messages[0]["content"] == [
        {"type": "text", "text": "<system-reminder>"},
        instruction,
        {"type": "text", "text": "</system-reminder>"},
        {"type": "text", "text": "Hello"},
    ]
    assert prepare_anthropic_subscription_system(prepared) == prepared
    assert original == [instruction, billing]
    assert prepare_anthropic_subscription_system(None)[1:] == [
        {"type": "text", "text": ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT}
    ]


def test_coding_agent_signature_moves_from_system_to_user_reminder():
    signature = (
        "custom providers (docs/custom-provider.md), adding models (docs/models.md), "
        "pi packages (docs/packages.md), environment variables (docs/environment-variables.md)"
    )
    system = prepare_anthropic_subscription_system(signature)
    messages = prepare_anthropic_subscription_messages([{"role": "user", "content": "Hello"}], signature)
    assert system == prepare_anthropic_subscription_system(None)
    assert signature not in json.dumps(system)
    assert signature in json.dumps(messages)


@pytest.mark.parametrize("session", ["e96634a3-fa28-4083-b354-55542e2dca01", "conversation-123"])
@pytest.mark.parametrize("owner_field", ["user_id", "team_id", "org_id", "project_id", "api_key"])
def test_gateway_sessions_are_isolated_by_authenticated_owner(session, owner_field):
    from litellm.proxy._types import UserAPIKeyAuth

    identity = SubscriptionIdentity("account-a", "a" * 64)
    body = {"metadata": {"user_id": json.dumps({"session_id": session, "customer": "same"})}}

    def prepare(owner):
        params = {
            "litellm_credential_name": "subscription",
            "metadata": {"session_id": session, "user_api_key_auth": UserAPIKeyAuth(**{owner_field: owner})},
        }
        return prepare_subscription_identity(body, {}, params, identity)

    first = prepare("owner-a")
    assert first == prepare("owner-a")
    assert first[1] != prepare("owner-b")[1]
    assert first[1]["x-claude-code-session-id"] != session
    UUID(first[1]["x-claude-code-session-id"])


def test_gateway_selected_session_wins_and_is_stable_when_headers_are_reused():
    from litellm.proxy._types import UserAPIKeyAuth

    identity = SubscriptionIdentity("account-a", "a" * 64)
    params = {"metadata": {"session_id": "gateway-session", "user_api_key_auth": UserAPIKeyAuth(user_id="alice")}}
    body = {"metadata": {"user_id": json.dumps({"session_id": "body-session", "customer": "alice"})}}
    result = prepare_subscription_identity(body, {"X-Claude-Code-Session-Id": "native-session"}, params, identity)
    assert result == prepare_subscription_identity(body, {}, params, identity)
    assert prepare_subscription_identity(*result, params, identity) == result
    assert json.loads(result[0]["metadata"]["user_id"])["customer"] == "alice"


@pytest.mark.parametrize("session", ["e96634a3-fa28-4083-b354-55542e2dca01", "conversation-123"])
def test_gateway_sessions_are_scoped_to_selected_credential(session):
    from litellm.proxy._types import UserAPIKeyAuth

    identity = SubscriptionIdentity("account-a", "a" * 64)
    metadata = {"session_id": session, "user_api_key_auth": UserAPIKeyAuth(user_id="alice")}
    first = prepare_subscription_identity({}, {}, {"metadata": metadata, "litellm_credential_name": "one"}, identity)
    second = prepare_subscription_identity({}, {}, {"metadata": metadata, "litellm_credential_name": "two"}, identity)
    assert first[1] != second[1]


def test_client_user_fields_and_serialized_auth_do_not_establish_gateway_scope():
    session = "e96634a3-fa28-4083-b354-55542e2dca01"
    identity = SubscriptionIdentity("account-a", "a" * 64)
    body = {"metadata": {"user_id": json.dumps({"session_id": session})}}
    params = {
        "user": "spoof",
        "metadata": {"user_id": "spoof", "user_api_key_user_id": "spoof", "user_api_key_auth": {"user_id": "spoof"}},
    }
    assert prepare_subscription_identity(body, {}, params, identity)[1] == {"x-claude-code-session-id": session}


def test_unidentified_requests_generate_distinct_bounded_sessions():
    identity = SubscriptionIdentity("account-a", "a" * 64)
    first_body, first_headers = prepare_subscription_identity({}, {}, {}, identity)
    second_body, second_headers = prepare_subscription_identity({}, {}, {}, identity)
    assert first_headers != second_headers
    assert len(first_body["metadata"]["user_id"]) <= 512
    assert len(second_body["metadata"]["user_id"]) <= 512
    UUID(first_headers["x-claude-code-session-id"])


def test_gateway_generated_trace_session_survives_rewritten_headers_without_merging_requests():
    from litellm.proxy._types import UserAPIKeyAuth

    identity = SubscriptionIdentity("account-a", "a" * 64)
    params = {"litellm_trace_id": "request-one", "metadata": {"user_api_key_auth": UserAPIKeyAuth(user_id="alice")}}
    first = prepare_subscription_identity({}, {}, params, identity)
    assert prepare_subscription_identity(*first, params, identity) == first
    assert prepare_subscription_identity({}, first[1], params, identity) == first
    other = prepare_subscription_identity({}, {}, {**params, "litellm_trace_id": "request-two"}, identity)
    assert first[1] != other[1]
    assert len(first[0]["metadata"]["user_id"]) <= 512


def test_dict_user_metadata_retains_customer_fields_during_subscription_normalization():
    identity = SubscriptionIdentity("account-a", "a" * 64)
    session = "e96634a3-fa28-4083-b354-55542e2dca01"
    body = {"metadata": {"user_id": {"session_id": session, "customer": "alice"}, "custom": "retained"}}
    original = deepcopy(body)
    result, headers = prepare_subscription_identity(body, {}, {}, identity)
    assert result == {
        "metadata": {
            "custom": "retained",
            "user_id": json.dumps(
                {"session_id": session, "customer": "alice", "device_id": "a" * 64, "account_uuid": "account-a"},
                separators=(",", ":"),
            ),
        }
    }
    assert headers == {"x-claude-code-session-id": session}
    assert body == original


def test_client_end_user_id_does_not_change_authenticated_session_scope():
    from litellm.proxy._types import UserAPIKeyAuth

    identity = SubscriptionIdentity("account-a", "a" * 64)
    first = prepare_subscription_identity(
        {},
        {},
        {
            "metadata": {
                "session_id": "shared-conversation",
                "user_api_key_auth": UserAPIKeyAuth(user_id="alice", end_user_id="spoof-one"),
            }
        },
        identity,
    )
    second = prepare_subscription_identity(
        {},
        {},
        {
            "metadata": {
                "session_id": "shared-conversation",
                "user_api_key_auth": UserAPIKeyAuth(user_id="alice", end_user_id="spoof-two"),
            }
        },
        identity,
    )
    assert first == second


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("oauth", [False, True])
async def test_public_api_identity_isolates_gateway_users_without_changing_provider_payload(
    surface: str, oauth: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Final

    import httpx
    from pydantic import JsonValue, TypeAdapter

    import litellm
    from litellm.llms.anthropic.oauth_client import ManagedAnthropicOAuthToken
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.models.credentials import CredentialItem
    from litellm.proxy._types import UserAPIKeyAuth

    bodies: Final[list[dict[str, JsonValue]]] = []
    sessions: Final[list[str | None]] = []
    body_adapter: Final = TypeAdapter(dict[str, JsonValue])
    tokens: Final = AnthropicOAuthTokens("sk-ant-oat-test", "refresh-test", 2000000000, "account", "a" * 64)
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="subscription", credential_info={"provider": "anthropic", "auth_type": "oauth"},
        credential_values={"litellm_internal_anthropic_auth_token": tokens.to_json()},
    )])

    def upstream(request: httpx.Request) -> httpx.Response:
        bodies.append(body_adapter.validate_json(request.content))
        sessions.append(request.headers.get("x-claude-code-session-id"))
        return httpx.Response(200, json={
            "id": "msg_identity", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as raw:
        client.client = raw
        for user in ("alice", "bob", "alice"):
            metadata: Final = {"session_id": "same-conversation", "user_api_key_auth": UserAPIKeyAuth(user_id=user)}
            params: Final = {
                "model": "anthropic/claude-sonnet-5", "client": client, "litellm_credential_name": "subscription",
                "api_key": ManagedAnthropicOAuthToken(tokens.access_token) if oauth else "sk-ant-api-test",
            }
            if surface == "chat":
                await litellm.acompletion(
                    **params, messages=[{"role": "user", "content": "test"}], max_tokens=8, metadata=metadata,
                )
            elif surface == "responses":
                await litellm.aresponses(**params, input="test", max_output_tokens=8, litellm_metadata=metadata)
            else:
                await litellm.anthropic_messages(
                    **params, messages=[{"role": "user", "content": "test"}], max_tokens=8, litellm_metadata=metadata,
                )

    assert len(bodies) == len(sessions) == 3
    assert bodies[0] == bodies[2]
    if not oauth:
        assert sessions == [None, None, None]
        assert bodies[0] == bodies[1]
        assert "metadata" not in bodies[0]
        return
    from litellm.llms.anthropic.subscription_billing import serialize_anthropic_subscription_request

    assert sessions[0] == sessions[2]
    assert sessions[0] != sessions[1]
    assert bodies[0]["system"] != bodies[1]["system"]
    for body, session in zip(bodies, sessions, strict=True):
        assert body["metadata"] == {"user_id": json.dumps({
            "device_id": tokens.device_id, "account_uuid": tokens.account_id, "session_id": session,
        }, separators=(",", ":"))}
        assert body == json.loads(serialize_anthropic_subscription_request({**bodies[0], "metadata": body["metadata"]}))


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"x-litellm-session-id": "gateway-session"}])
async def test_proxy_dict_identity_reaches_native_messages_without_losing_customer_fields(headers) -> None:
    from typing import Final

    import httpx

    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    identity: Final = {"session_id": "conversation-123", "customer": "synthetic"}
    data: Final = {"metadata": {"user_id": identity}, "litellm_metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="litellm_metadata",
    )
    bodies: Final[list[bytes]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json={
            "id": "msg_metadata", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as raw:
        client.client = raw
        await litellm.anthropic_messages(
            model="anthropic/claude-sonnet-5", messages=[{"role": "user", "content": "test"}],
            max_tokens=8, api_key="sk-ant-api-test", client=client, **data,
        )
    assert len(bodies) == 1
    assert json.loads(bodies[0]) == {
        "model": "claude-sonnet-5", "max_tokens": 8, "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "test", "cache_control": {"type": "ephemeral"}},
        ]}],
        "metadata": {"user_id": json.dumps(identity, separators=(",", ":"))},
    }
    assert identity == {"session_id": "conversation-123", "customer": "synthetic"}
