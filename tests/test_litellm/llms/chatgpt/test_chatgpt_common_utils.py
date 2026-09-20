import os
from unittest.mock import patch

from litellm.llms.chatgpt.common_utils import (
    CODEX_CLI_VERSION,
    DEFAULT_ORIGINATOR,
    DEFAULT_USER_AGENT,
    get_chatgpt_default_headers,
    get_chatgpt_user_agent,
)


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
