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
    assert prepare_anthropic_subscription_system(None) == [
        {"type": "text", "text": ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT}
    ]


def test_coding_agent_signature_moves_from_system_to_user_reminder():
    signature = (
        "custom providers (docs/custom-provider.md), adding models (docs/models.md), "
        "pi packages (docs/packages.md), environment variables (docs/environment-variables.md)"
    )
    system = prepare_anthropic_subscription_system(signature)
    messages = prepare_anthropic_subscription_messages([{"role": "user", "content": "Hello"}], signature)
    assert system == [{"type": "text", "text": ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT}]
    assert signature not in json.dumps(system)
    assert signature in json.dumps(messages)
