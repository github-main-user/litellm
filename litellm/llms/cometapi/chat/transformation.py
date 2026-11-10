"""
Support for CometAPI's `/v1/chat/completions` endpoint.

Based on OpenAI-compatible API interface implementation
Documentation: [CometAPI Documentation Link]
"""

from collections.abc import AsyncIterator, Iterator
from types import MappingProxyType
from typing import Any, Final, get_args

import httpx
from pydantic import TypeAdapter

from litellm.exceptions import BadRequestError, normalize_error_status_code
from litellm.llms.anthropic.chat.transformation import (
    REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT,
    AnthropicConfig,
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.anthropic import AnthropicOutputConfig, AnthropicThinkingParam
from litellm.types.llms.openai import REASONING_EFFORT, AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import CometAPIException


class CometAPIConfig(OpenAIGPTConfig):
    """
    CometAPI configuration class, inherits from OpenAIGPTConfig

    Since CometAPI is OpenAI-compatible API, we inherit from OpenAIGPTConfig
    and only need to override necessary methods to handle CometAPI-specific features
    """

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        supports_thinking: Final = model.removeprefix("cometapi/").startswith("claude-") and (
            AnthropicModelInfo._supports_model_capability(  # pyright: ignore[reportPrivateUsage]  # shared capability
                model, "supports_reasoning", "cometapi"
            )
        )
        return [  # mutable-ok: inherited contract
            *super().get_supported_openai_params(model),
            "reasoning_effort",
            *(("thinking",) if supports_thinking else ()),
        ]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        """
        Map OpenAI format parameters to CometAPI format
        """
        reasoning_effort: Final = non_default_params.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in get_args(REASONING_EFFORT):
            raise BadRequestError(
                message=f"Invalid reasoning_effort: {reasoning_effort!r}. Must be one of: {get_args(REASONING_EFFORT)}",
                model=model,
                llm_provider="cometapi",
            )
        mapped_openai_params: Final = TypeAdapter(dict[str, object]).validate_python(
            super().map_openai_params(non_default_params, optional_params, model, drop_params)
        )

        if "thinking" in self.get_supported_openai_params(model):
            thinking: Final = non_default_params.get("thinking")
            thinking_params: Final = (
                TypeAdapter(dict[str, object]).validate_python(thinking) if isinstance(thinking, dict) else None
            )
            if thinking_params is not None and thinking_params.get("type") == "disabled":
                disabled_params: Final = {  # mutable-ok: shared Anthropic normalization takes a mutable out-param
                    key: value for key, value in mapped_openai_params.items() if key != "reasoning_effort"
                }
                AnthropicModelInfo.maybe_drop_disabled_thinking(model, disabled_params, "cometapi")
                return disabled_params
            if isinstance(reasoning_effort, str) and (
                AnthropicModelInfo._is_adaptive_thinking_model(  # pyright: ignore[reportPrivateUsage]  # capability API
                    model, "cometapi"
                )
            ):
                wire: Final = AnthropicConfig._map_reasoning_effort(  # pyright: ignore[reportPrivateUsage]  # API reuse
                    reasoning_effort, model, "cometapi", llm_provider="cometapi"
                )
                adaptive_params: Final = MappingProxyType(
                    {key: value for key, value in mapped_openai_params.items() if key != "reasoning_effort"}
                )
                output_config: Final = (
                    TypeAdapter(AnthropicOutputConfig).validate_python(
                        MappingProxyType({"effort": REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT[reasoning_effort]})
                    )
                    if wire is not None
                    else None
                )
                reasoning_params: Final = {  # mutable-ok: shared Anthropic normalization takes a mutable out-param
                    **adaptive_params,
                    "thinking": wire or AnthropicThinkingParam(type="disabled"),
                    **(
                        MappingProxyType({"output_config": output_config})
                        if output_config is not None
                        else MappingProxyType({})
                    ),
                }
                AnthropicModelInfo.maybe_drop_disabled_thinking(model, reasoning_params, "cometapi")
                return reasoning_params
            AnthropicModelInfo.translate_legacy_thinking_for_adaptive_model(model, mapped_openai_params, "cometapi")

        # CometAPI-specific parameters (if any)
        extra_body: Final[dict[str, Any]] = {}
        # TODO: Add CometAPI-specific parameter handling here
        # Example:
        # custom_param = non_default_params.pop("custom_param", None)
        # if custom_param is not None:
        #     extra_body["custom_param"] = custom_param

        if extra_body:
            mapped_openai_params["extra_body"] = extra_body

        return mapped_openai_params

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list["ChatCompletionToolParam"] | None = None,
    ) -> tuple[list[AllMessageValues], list["ChatCompletionToolParam"] | None]:
        """
        Remove cache control flags from messages and tools if not supported
        """
        # For CometAPI, use default behavior (remove cache control)
        return super().remove_cache_control_flag_from_messages_and_tools(model, messages, tools)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        extra_body: Final = optional_params.pop("extra_body", {})
        response: Final = super().transform_request(model, messages, optional_params, litellm_params, headers)
        response.update(extra_body)
        return response

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for the CometAPI call.

        Returns:
            str: The complete URL for the API call.
        """
        # Default base
        if api_base is None:
            api_base = "https://api.cometapi.com/v1"
        endpoint: Final = "chat/completions"

        # Normalize
        api_base = api_base.rstrip("/")

        # If endpoint already present, return as-is
        if endpoint in api_base:
            return api_base

        # Ensure we include /v1 prefix when missing
        if api_base.endswith("/v1"):
            return f"{api_base}/{endpoint}"
        if api_base.endswith("/v1/"):
            return f"{api_base}{endpoint}"
        # If user provided https://api.cometapi.com, add /v1
        if api_base == "https://api.cometapi.com":
            return f"{api_base}/v1/{endpoint}"
        # Generic fallback: if '/v1' not in path, add it
        if "/v1" not in api_base.split("//", 1)[-1]:
            return f"{api_base}/v1/{endpoint}"
        return f"{api_base}/{endpoint}"

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """
        Return CometAPI-specific error class
        """
        return CometAPIException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
        """
        Get model response iterator for streaming responses
        """
        return CometAPIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class CometAPIChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Handler for CometAPI streaming chat completion responses
    """

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse individual chunks from streaming response
        """
        try:
            # Handle error in chunk
            if "error" in chunk:
                error_chunk: Final[dict[str, object]] = chunk["error"]  # mutable-ok: exception body contract
                error_message: Final = "CometAPI Error: {}".format(error_chunk.get("message", "Unknown error"))
                raise CometAPIException(
                    message=error_message,
                    status_code=normalize_error_status_code(error_chunk.get("code"), 500),
                    headers={"Content-Type": "application/json"},
                    body=error_chunk,
                )

            # Process choices
            new_choices: Final = []
            for choice in chunk["choices"]:
                # Handle reasoning content if present
                if "delta" in choice and "reasoning" in choice["delta"]:
                    choice["delta"]["reasoning_content"] = choice["delta"].get("reasoning")
                new_choices.append(choice)

            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=chunk.get("usage"),
                model=chunk["model"],
                choices=new_choices,
            )
        except KeyError as e:
            raise CometAPIException(
                message=f"KeyError: {e}, Got unexpected response from CometAPI: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
