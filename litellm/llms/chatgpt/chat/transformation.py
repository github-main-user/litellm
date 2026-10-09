from typing import Any, Final

from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    ensure_chatgpt_session_id,
    get_chatgpt_api_base,
    get_chatgpt_default_headers,
    get_managed_chatgpt_account_id,
    merge_chatgpt_headers,
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
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        access_token: Final = require_managed_chatgpt_access_token(model, litellm_params.get("litellm_credential_name"), api_key)
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        account_id: Final = get_managed_chatgpt_account_id(litellm_params.get("litellm_credential_name"), access_token)
        default_headers: Final = get_chatgpt_default_headers(access_token, account_id, session_id)
        return merge_chatgpt_headers(headers, default_headers)

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
