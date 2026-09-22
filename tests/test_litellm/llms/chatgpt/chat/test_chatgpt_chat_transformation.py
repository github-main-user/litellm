from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_chat_requires_explicit_credentials_when_file_auth_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_dir: Final = tmp_path / "chatgpt"
    monkeypatch.setenv("CHATGPT_ALLOW_FILE_AUTH", "false")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
    config: Final = ChatGPTConfig()

    with pytest.raises(AuthenticationError, match="file authentication is disabled"):
        config.validate_environment(
            headers={}, model="gpt-5.5", messages=[], optional_params={}, litellm_params={}
        )

    headers: Final = config.validate_environment(
        headers={},
        model="gpt-5.5",
        messages=[],
        optional_params={},
        litellm_params={"chatgpt_auth_account_id": "managed-account"},
        api_key="managed-token",
    )
    assert headers["Authorization"] == "Bearer managed-token"
    assert headers["ChatGPT-Account-Id"] == "managed-account"
    assert not token_dir.exists()


def test_provider_resolution_does_not_start_legacy_authentication() -> None:
    config = ChatGPTConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = "https://chatgpt.example.com"

    api_base, api_key, provider = config._get_openai_compatible_provider_info(
        model="gpt-5.4",
        api_base=None,
        api_key=None,
        custom_llm_provider="chatgpt",
    )

    assert api_base == "https://chatgpt.example.com"
    assert api_key is None
    assert provider == "chatgpt"
    config.authenticator.get_access_token.assert_not_called()


def test_chat_uses_selected_deployment_credentials() -> None:
    config = ChatGPTConfig()
    config.authenticator = MagicMock()

    headers = config.validate_environment(
        headers={},
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={"chatgpt_auth_account_id": "account-a"},
        api_key="access-a",
    )

    assert headers["Authorization"] == "Bearer access-a"
    assert headers["ChatGPT-Account-Id"] == "account-a"
    config.authenticator.get_access_token.assert_not_called()
    config.authenticator.get_account_id.assert_not_called()
