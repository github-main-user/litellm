"""
Anthropic Token Counter implementation using the CountTokens API.
"""

import os
from collections.abc import Mapping
from typing import Any, Final, Protocol

from pydantic import Field

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.credential_proxy import get_credential_proxy_url
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.oauth_client import (
    AnthropicOAuthError,
    recover_managed_anthropic_oauth_headers,
    sanitize_retry_after,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import CallTypes, LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
anthropic_count_tokens_handler: Final = AnthropicCountTokensHandler()


class _OAuthCredentialHook(Protocol):
    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object] | None: ...


class AnthropicTokenCountResponse(TokenCountResponse):
    retry_after: str | None = Field(default=None, exclude=True)


class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Anthropic provider using the CountTokens API."""

    def __init__(
        self,
        oauth_credential_hook: _OAuthCredentialHook | None = None,
        count_tokens_handler: AnthropicCountTokensHandler | None = None,
    ) -> None:
        self._oauth_credential_hook = oauth_credential_hook
        self._count_tokens_handler = count_tokens_handler

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.ANTHROPIC.value

    async def _resolve_named_credential(
        self,
        *,
        model: str,
        litellm_params: dict[str, Any],
    ) -> dict[str, object] | None:
        hook: Final = self._oauth_credential_hook or self._default_oauth_credential_hook()
        provider: Final = litellm_params.get("custom_llm_provider") or LlmProviders.ANTHROPIC.value
        hook_kwargs: Final[dict[str, object]] = {
            **litellm_params,
            "model": model,
            "custom_llm_provider": provider,
        }
        return await hook.async_pre_call_deployment_hook(hook_kwargs, None)

    @staticmethod
    def _default_oauth_credential_hook() -> _OAuthCredentialHook:
        from litellm.proxy.credential_endpoints.anthropic_oauth import AnthropicOAuthCredentialHook

        return AnthropicOAuthCredentialHook()

    @staticmethod
    def _retry_after(headers: object) -> str | None:
        if not isinstance(headers, Mapping):
            return None
        for name, value in headers.items():
            if isinstance(name, str) and name.lower() == "retry-after":
                return sanitize_retry_after(value) if isinstance(value, str) else None
        return None

    @staticmethod
    def _error_response(
        *,
        request_model: str,
        model_to_use: str,
        message: str,
        status_code: int,
        retry_after: str | None = None,
    ) -> TokenCountResponse:
        return AnthropicTokenCountResponse(
            total_tokens=0,
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type="anthropic_api",
            error=True,
            error_message=message,
            status_code=status_code,
            retry_after=retry_after,
        )

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, Any]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
    ) -> TokenCountResponse | None:
        """
        Count tokens using Anthropic's CountTokens API.

        Args:
            model_to_use: The model identifier
            messages: The messages to count tokens for
            contents: Alternative content format (not used for Anthropic)
            deployment: Deployment configuration containing litellm_params
            request_model: The original request model name

        Returns:
            TokenCountResponse with token count, or None if counting fails
        """
        from litellm.llms.anthropic.common_utils import AnthropicError

        if messages is None and system is None and tools is None:
            return None

        deployment_params: Final = (deployment or {}).get("litellm_params", {})
        litellm_params: Final[dict[str, Any]] = deployment_params if isinstance(deployment_params, dict) else {}
        credential_name: Final = litellm_params.get("litellm_credential_name")

        try:
            proxy_url: Final = get_credential_proxy_url(
                credential_name if isinstance(credential_name, str) else None
            )
            resolved: Final = (
                await self._resolve_named_credential(model=model_to_use, litellm_params=litellm_params)
                if isinstance(credential_name, str) and credential_name
                else None
            )
            # Get Anthropic API key from deployment config or environment
            resolved_api_key: Final = resolved.get("api_key") if resolved is not None else litellm_params.get("api_key")
            api_key: Final = (
                resolved_api_key
                if isinstance(resolved_api_key, str) and resolved_api_key
                else None
                if credential_name
                else os.getenv("ANTHROPIC_API_KEY")
            )
            if api_key is None:
                if not credential_name:
                    verbose_logger.warning("No Anthropic API key found for token counting")
                    return None
                message: Final = f"Anthropic credential '{credential_name}' is unavailable"
                verbose_logger.warning(message)
                return self._error_response(
                    request_model=request_model,
                    model_to_use=model_to_use,
                    message=message,
                    status_code=401,
                )

            handler: Final = self._count_tokens_handler or anthropic_count_tokens_handler
            try:
                result = await handler.handle_count_tokens_request(
                    model=model_to_use,
                    messages=messages or [],
                    api_key=api_key,
                    api_base=litellm_params.get("api_base"),
                    tools=tools,
                    system=system,
                    proxy_url=proxy_url,
                )
            except AnthropicError as error:
                if error.status_code != 401 or resolved is None:
                    raise
                recovered: Final = await recover_managed_anthropic_oauth_headers(
                    headers={"authorization": f"Bearer {api_key}"}, litellm_params=resolved
                )
                if recovered is None:
                    raise
                authorization: Final = recovered.get("authorization")
                if not isinstance(authorization, str):
                    raise
                result = await handler.handle_count_tokens_request(
                    model=model_to_use,
                    messages=messages or [],
                    api_key=authorization.removeprefix("Bearer "),
                    api_base=litellm_params.get("api_base"),
                    tools=tools,
                    system=system,
                    proxy_url=proxy_url,
                )
            input_tokens: Final = result.get("input_tokens")
            if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
                return self._error_response(
                    request_model=request_model,
                    model_to_use=model_to_use,
                    message="Anthropic CountTokens API returned an invalid input_tokens value",
                    status_code=502,
                )
            return TokenCountResponse(
                total_tokens=input_tokens,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="anthropic_api",
                original_response=result,
            )
        except AnthropicOAuthError as error:
            return self._error_response(
                request_model=request_model,
                model_to_use=model_to_use,
                message=str(error),
                status_code=error.status_code,
                retry_after=error.retry_after,
            )
        except AuthenticationError as error:
            verbose_logger.warning("Anthropic managed credential error: %s", error.message)
            return self._error_response(
                request_model=request_model,
                model_to_use=model_to_use,
                message=error.message,
                status_code=error.status_code,
                retry_after=self._retry_after(getattr(error.response, "headers", None)),
            )
        except AnthropicError as error:
            verbose_logger.warning(
                "Anthropic CountTokens API error: status=%s, message=%s", error.status_code, error.message
            )
            return self._error_response(
                request_model=request_model,
                model_to_use=model_to_use,
                message=error.message,
                status_code=error.status_code,
                retry_after=self._retry_after(error.headers),
            )
        except Exception as error:  # noqa: BLE001  # convert unexpected provider failures to the response contract
            verbose_logger.warning("Error calling Anthropic CountTokens API: %s", error)
            return self._error_response(
                request_model=request_model,
                model_to_use=model_to_use,
                message=str(error),
                status_code=500,
            )
