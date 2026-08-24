from unittest.mock import MagicMock

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


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
