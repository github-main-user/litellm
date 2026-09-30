import os
from time import time
from unittest.mock import patch

import httpx

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.chatgpt.common_utils import (
    CODEX_CLI_VERSION,
    DEFAULT_ORIGINATOR,
    DEFAULT_USER_AGENT,
    chatgpt_quota_reset_seconds,
    get_chatgpt_default_headers,
    get_chatgpt_user_agent,
)


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
