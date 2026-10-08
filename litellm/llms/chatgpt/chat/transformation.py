from typing import Any, Final

from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    ensure_chatgpt_session_id,
    get_chatgpt_api_base,
    get_chatgpt_default_headers,
)
from ..oauth_client import require_managed_chatgpt_access_token
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str,
    ) -> tuple[str | None, str | None, str]:
        return api_base or get_chatgpt_api_base(), api_key, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        access_token: Final = require_managed_chatgpt_access_token(model, litellm_params.get("litellm_credential_name"), api_key)
        validated_headers: Final = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, access_token, api_base
        )

        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(access_token, litellm_params.get("chatgpt_auth_account_id"), session_id)
        return {
            **default_headers,
            **{
                name: value
                for name, value in validated_headers.items()
                if name.lower() not in {"authorization", "chatgpt-account-id"}
            },
        }

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        optional_params.setdefault("stream", False)
        return optional_params
