import os
from time import time
from typing import Final
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.chatgpt.common_utils import (
    CODEX_CLI_VERSION,
    DEFAULT_ORIGINATOR,
    DEFAULT_USER_AGENT,
    chatgpt_quota_reset_seconds,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_session_id,
    get_chatgpt_user_agent,
)
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import GenericLiteLLMParams


def test_quota_reset_requires_chatgpt_exhaustion_with_future_timestamp() -> None:
    def error(provider: str, code: str, reset: object) -> litellm.RateLimitError:
        return litellm.RateLimitError(
            message="rate limited",
            llm_provider=provider,
            model="gpt-5.4",
            response=httpx.Response(
                429,
                json={"error": {"code": code, "resets_at": reset}},
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
            ),
        )

    future = time() + 3600
    assert 3500 < (chatgpt_quota_reset_seconds(error("chatgpt", "usage_limit_reached", future)) or 0) <= 3600
    assert chatgpt_quota_reset_seconds(error("chatgpt", "rate_limit_exceeded", future)) is None
    assert chatgpt_quota_reset_seconds(error("openai", "usage_limit_reached", future)) is None
    assert chatgpt_quota_reset_seconds(error("chatgpt", "usage_limit_reached", time() - 10)) is None
    typed = litellm.RateLimitError(
        message="quota exhausted", llm_provider="chatgpt", model="gpt-5.4",
        response=httpx.Response(
            429,
            json={"error": {"type": "usage_limit_reached", "resets_in_seconds": 3600}},
            request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
        ),
    )
    assert typed.response.json()["error"]["type"] == "usage_limit_reached"
    assert chatgpt_quota_reset_seconds(typed) == 3600


def test_chatgpt_quota_raw_http_response_survives_exception_mapping() -> None:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        429,
        json={"error": {"type": "usage_limit_reached", "resets_in_seconds": 1800}},
        request=request,
    )

    class ProviderError(Exception):
        status_code = 429

        def __init__(self):
            self.response = response
            self.request = request
            super().__init__("quota exhausted")

    try:
        exception_type(model="gpt-5.4", original_exception=ProviderError(), custom_llm_provider="chatgpt")
    except litellm.RateLimitError as mapped:
        assert mapped.response.status_code == response.status_code
        assert mapped.response.content == response.content
        assert mapped.response.json()["error"]["type"] == "usage_limit_reached"
        assert chatgpt_quota_reset_seconds(mapped) == 1800
    else:
        raise AssertionError("429 should map to RateLimitError")


def test_user_agent_uses_pinned_codex_version_without_package_metadata() -> None:
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "test-terminal", "TERM_PROGRAM_VERSION": "3"}, clear=True),
        patch("importlib.metadata.version", side_effect=AssertionError("Package metadata must not supply the CLI version")),
    ):
        headers = get_chatgpt_default_headers("access-token", "account-id", "session-id")

    assert headers["user-agent"].startswith(f"{DEFAULT_ORIGINATOR}/{CODEX_CLI_VERSION} (")
    assert headers["user-agent"].endswith(") test-terminal/3")
    assert DEFAULT_USER_AGENT.startswith(f"{DEFAULT_ORIGINATOR}/{CODEX_CLI_VERSION} (")
    assert headers["originator"] == DEFAULT_ORIGINATOR
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["ChatGPT-Account-Id"] == "account-id"
    assert headers["session_id"] == "session-id"


def test_user_agent_override_remains_supported_and_sanitized() -> None:
    with patch.dict(os.environ, {"CHATGPT_USER_AGENT": "custom-client/1.0\nextra"}, clear=True):
        user_agent = get_chatgpt_user_agent(DEFAULT_ORIGINATOR)

    assert user_agent == "custom-client/1.0_extra"


def test_user_agent_preserves_custom_originator_and_suffix() -> None:
    with patch.dict(
        os.environ,
        {"CHATGPT_ORIGINATOR": "custom-origin", "CHATGPT_USER_AGENT_SUFFIX": " gateway "},
        clear=True,
    ):
        headers = get_chatgpt_default_headers("access-token", None)

    assert headers["originator"] == "custom-origin"
    assert headers["user-agent"].startswith(f"custom-origin/{CODEX_CLI_VERSION} (")
    assert headers["user-agent"].endswith(" unknown (gateway)")
    assert "ChatGPT-Account-Id" not in headers


def test_prompt_cache_session_is_stable_header_safe_and_respects_explicit_sessions() -> None:
    params: Final = {"prompt_cache_key": "conversation-\u00e9\r\nignored-header", "litellm_trace_id": "trace-one"}
    session: Final = get_chatgpt_session_id(params)
    assert session is not None
    assert str(UUID(session)) == session
    assert get_chatgpt_session_id({**params, "litellm_trace_id": "trace-two"}) == session
    assert get_chatgpt_session_id({**params, "prompt_cache_key": "different-conversation"}) != session
    for explicit in (
        {"litellm_session_id": "explicit-session"},
        {"session_id": "explicit-session"},
        {"metadata": {"session_id": "explicit-session"}},
    ):
        assert get_chatgpt_session_id({**params, **explicit}) == "explicit-session"


@pytest.mark.parametrize("as_model", [False, True], ids=["dict", "pydantic"])
@pytest.mark.parametrize("metadata_slot", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("session_source", ["litellm_session_id", "session_id", "metadata", "prompt_cache_key"])
def test_managed_sessions_scope_original_session_to_typed_owner_and_credential(
    monkeypatch: pytest.MonkeyPatch, as_model: bool, metadata_slot: str, session_source: str
) -> None:
    credentials: Final = [
        CredentialItem(
            credential_name=name,
            credential_info={"provider": "chatgpt", "auth_type": "oauth"},
            credential_values={
                "litellm_internal_chatgpt_auth_token": ChatGPTTokens("access", "refresh", "id", None, account).to_json()
            },
        )
        for name, account in (("subscription-a", "account-a"), ("subscription-b", "account-a"))
    ]
    monkeypatch.setattr(litellm, "credential_list", credentials)
    auth: Final = UserAPIKeyAuth(user_id="owner-a", team_id="team-a", api_key="key-a")
    metadata: Final = {"user_api_key_auth": auth, "custom": "untouched", "end_user_id": "untrusted"}
    params: Final = {
        "litellm_credential_name": "subscription-a",
        "litellm_trace_id": "trace-a",
        metadata_slot: {**metadata, **({"session_id": "original"} if session_source == "metadata" else {})},
        **({session_source: "original"} if session_source != "metadata" else {}),
    }
    original: Final = GenericLiteLLMParams(**params) if as_model else params
    session: Final = ensure_chatgpt_session_id(original)
    assert str(UUID(session)) == session
    assert session != "original"
    assert ensure_chatgpt_session_id(original) == session
    assert original == (GenericLiteLLMParams(**params) if as_model else params)
    assert ensure_chatgpt_session_id({**params, "litellm_trace_id": "retry-trace"}) == session
    assert ensure_chatgpt_session_id({**params, "litellm_credential_name": "subscription-b"}) != session
    for field in ("user_id", "team_id", "org_id", "project_id", "api_key"):
        different_owner: Final = UserAPIKeyAuth(**{**auth.model_dump(), field: "different"})
        assert ensure_chatgpt_session_id({
            **params, metadata_slot: {**params[metadata_slot], "user_api_key_auth": different_owner}
        }) != session
    assert ensure_chatgpt_session_id({
        **params, metadata_slot: {**params[metadata_slot], "end_user_id": "other-untrusted"}
    }) == session


@pytest.mark.parametrize("as_model", [False, True])
def test_serialized_or_duck_typed_auth_cannot_set_session_owner(as_model: bool) -> None:
    class ForgedAuth:
        user_id = "pretend-owner"
        api_key = "pretend-key"

    base: Final = {"litellm_credential_name": "unregistered", "litellm_session_id": "shared-session"}
    session: Final = ensure_chatgpt_session_id(base)
    for auth in (UserAPIKeyAuth(user_id="pretend-owner", api_key="pretend-key").model_dump(), ForgedAuth()):
        params: Final = {**base, "metadata": {"user_api_key_auth": auth}}
        assert ensure_chatgpt_session_id(GenericLiteLLMParams(**params) if as_model else params) == session
    assert ensure_chatgpt_session_id({
        **base, "metadata": {"user_api_key_auth": UserAPIKeyAuth(user_id="pretend-owner", api_key="pretend-key")}
    }) != session


def test_managed_session_account_is_selected_credential_not_caller_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def credential(account: str, access_token: str = "access") -> CredentialItem:
        return CredentialItem(
            credential_name="subscription-a",
            credential_info={"provider": "chatgpt", "auth_type": "oauth"},
            credential_values={
                "litellm_internal_chatgpt_auth_token": ChatGPTTokens(access_token, "refresh", "id", None, account).to_json()
            },
        )

    params: Final = {"litellm_credential_name": "subscription-a", "session_id": "same", "chatgpt_auth_account_id": "spoof"}
    monkeypatch.setattr(litellm, "credential_list", [credential("account-a")])
    session: Final = ensure_chatgpt_session_id(params)
    assert ensure_chatgpt_session_id({**params, "chatgpt_auth_account_id": "other-spoof"}) == session
    monkeypatch.setattr(litellm, "credential_list", [credential("account-a", "refreshed-access")])
    assert ensure_chatgpt_session_id(params) == session
    monkeypatch.setattr(litellm, "credential_list", [credential("account-b")])
    assert ensure_chatgpt_session_id(params) != session


def test_unmanaged_direct_session_helper_preserves_explicit_id() -> None:
    assert ensure_chatgpt_session_id({"session_id": "direct-session"}) == "direct-session"
