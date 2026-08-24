from typing import Any, Final

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str,
    ) -> tuple[str | None, str | None, str]:
        return api_base or self.authenticator.get_api_base(), api_key, custom_llm_provider

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
        resolved_api_key: Final = api_key or self._get_legacy_access_token(
            model=model,
            custom_llm_provider="chatgpt",
        )
        validated_headers: Final = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, resolved_api_key, api_base
        )

        account_id: Final = (
            litellm_params.get("chatgpt_auth_account_id") if api_key else self.authenticator.get_account_id()
        )
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(resolved_api_key, account_id, session_id)
        return {**default_headers, **validated_headers}

    def _get_legacy_access_token(self, model: str, custom_llm_provider: str) -> str:
        try:
            return self.authenticator.get_access_token()
        except GetAccessTokenError as error:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(error),
            ) from error

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
