"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final, cast

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.anthropic_cache_control_hook import (
    AllToolParamValues,
    AnthropicCacheControlHook,
)
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.llms.anthropic.common_utils import supports_anthropic_cache_control
from litellm.llms.anthropic.experimental_pass_through.messages.utils import anthropic_system_to_openai_message
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


def _get_min_token_count_for_deployments(healthy_deployments: list[dict]) -> int:
    """
    Returns the lowest minimum cacheable prefix across a model group.

    This gate only decides whether the cache lookup is worth doing. It cannot cause a wrong pin,
    because a deployment is only pinned when the cache already holds an entry for the prefix, and
    entries are written by `async_log_success_event` against the deployment's real model. A model
    that will not cache a prefix never records one, so there is nothing to pin it to.

    That makes the lowest minimum in the group the correct threshold rather than the highest.
    `model` here is the model-group alias the operator chose, not a model name, so the threshold
    has to come from the deployments themselves, and a group may mix models whose minimums differ.
    Taking the highest would skip the lookup for a prefix a lower-minimum member genuinely cached,
    losing a cache hit it had earned. The lowest can only cost a lookup that finds nothing.
    """
    return min(
        (
            get_prompt_cache_min_tokens(model=deployment["litellm_params"]["model"])
            for deployment in healthy_deployments
            if deployment.get("litellm_params", {}).get("model")
        ),
        default=DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    )


def _responses_prompt_caching_messages(
    request_kwargs: Mapping[str, object] | None,
) -> list[AllMessageValues] | None:
    if request_kwargs is None or request_kwargs.get("previous_response_id"):
        return None
    response_input: Final = request_kwargs.get("input")
    if not isinstance(response_input, (str, list)):
        return None

    from openai.types.responses.response_create_params import ResponseInputParam

    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    return cast(
        list[AllMessageValues],
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=cast(str | ResponseInputParam, response_input),
            responses_api_request=dict(request_kwargs),
            replay_reasoning=True,
        ),
    )


def _with_anthropic_system(messages: list[AllMessageValues], system: object) -> list[AllMessageValues]:
    system_message: Final = anthropic_system_to_openai_message(system)
    if system_message is None:
        return messages
    return [system_message, *(message for message in messages if message.get("role") != "system")]


def _routing_prompt_caching_messages(
    messages: list[AllMessageValues] | None,
    request_kwargs: Mapping[str, object] | None,
    models: Sequence[str],
) -> list[AllMessageValues] | None:
    cache_messages: Final = messages if messages is not None else _responses_prompt_caching_messages(request_kwargs)
    if cache_messages is None:
        return None
    tools: Final = request_kwargs.get("tools") if request_kwargs is not None else None
    if request_kwargs is None or request_kwargs.get("call_type") != CallTypes.anthropic_messages.value:
        return AnthropicCacheControlHook.messages_with_default_injections(
            messages=cache_messages,
            models=models,
            tools=cast(  # cast-ok: request_kwargs is untyped; the stand-down scan duck-types every tool it reads
                list[AllToolParamValues] | None, tools
            ),
            request_kwargs=request_kwargs,
        )

    model: Final = next((model for model in models if supports_anthropic_cache_control(model, None)), None)
    system: Final = request_kwargs.get("system")
    native_messages, native_system = AnthropicCacheControlHook.maybe_inject_cache_control(
        messages=cast(list[dict[str, object]], cache_messages),
        system=system if isinstance(system, (str, list)) else None,
        kwargs={key: value for key, value in request_kwargs.items() if key not in ("metadata", "litellm_metadata")},
        model=model,
        tools=tools if isinstance(tools, list) else None,
    )
    return _with_anthropic_system(cast(list[AllMessageValues], native_messages), native_system)


def _prompt_cache_affinity_key(model_group: str | None, request_kwargs: object) -> str | None:
    if not model_group or not isinstance(request_kwargs, Mapping):
        return None
    extra_body: Final = request_kwargs.get("extra_body")
    key: Final = request_kwargs.get("prompt_cache_key") or (
        extra_body.get("prompt_cache_key") if isinstance(extra_body, Mapping) else None
    )
    if not isinstance(key, str) or not key:
        return None
    digest: Final = hashlib.sha256(json.dumps((model_group, key)).encode()).hexdigest()
    return f"deployment:{digest}:prompt_cache_key"


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(self, cache: DualCache):
        self.cache = cache
        self.prompt_cache = PromptCachingCache(cache=cache)

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        if request_kwargs is not None and request_kwargs.get("_target_order") is not None:
            return healthy_deployments

        models: Final = tuple(
            deployment["litellm_params"]["model"]
            for deployment in healthy_deployments
            if isinstance(deployment.get("litellm_params"), dict) and deployment["litellm_params"].get("model")
        )
        cache_messages: Final = _routing_prompt_caching_messages(messages, request_kwargs, models)
        if (
            request_kwargs is not None
            and cache_messages
            and _prompt_cache_affinity_key(model, request_kwargs) is None
            and models
            and all(candidate.startswith("chatgpt/") for candidate in models)
        ):
            request_kwargs["prompt_cache_key"] = await asyncio.to_thread(
                self.prompt_cache.implicit_prompt_cache_key,
                cache_messages,
                cast(list[ChatCompletionToolParam] | None, request_kwargs.get("tools")),
            )
        request_key: Final = _prompt_cache_affinity_key(model, request_kwargs)
        if request_key is not None:
            cached_model_id: Final = await self.cache.async_get_cache(key=request_key)
            cached_deployment: Final = next(
                (deployment for deployment in healthy_deployments if deployment["model_info"]["id"] == cached_model_id),
                None,
            )
            if cached_deployment is not None:
                return [cached_deployment]

        if cache_messages is None or not await offload_token_count(is_prompt_caching_valid_prompt)(
            messages=cache_messages,
            model=model,
            min_token_count=_get_min_token_count_for_deployments(healthy_deployments),
        ):
            return healthy_deployments

        ## AUTO PROMPT CACHING - the breakpoints this request will carry are injected inside
        ## `litellm.acompletion`, after a deployment has been picked, so the affinity key has to
        ## be derived from the messages as they will be sent, not as they arrive here.
        model_id_dict: Final = await self.prompt_cache.async_get_model_id(
            messages=cache_messages,
            tools=None,
        )
        if model_id_dict is None:
            return healthy_deployments
        return next(
            (
                [deployment]
                for deployment in healthy_deployments
                if deployment["model_info"]["id"] == model_id_dict["model_id"]
            ),
            healthy_deployments,
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)

        if standard_logging_object is None:
            return

        call_type: Final = standard_logging_object["call_type"]

        if call_type not in (
            CallTypes.completion.value,
            CallTypes.acompletion.value,
            CallTypes.anthropic_messages.value,
            CallTypes.responses.value,
            CallTypes.aresponses.value,
        ):  # only use prompt caching for completion calls
            verbose_logger.debug("Skipping prompt cache affinity for unsupported call type: %s", call_type)
            return

        model: Final = standard_logging_object["model"]
        logged_messages: Final = standard_logging_object["messages"]
        model_id: Final = standard_logging_object["model_id"]
        request_key: Final = next(
            (
                key
                for params in (standard_logging_object.get("model_parameters"), kwargs.get("optional_params"), kwargs)
                if (key := _prompt_cache_affinity_key(standard_logging_object.get("model_group"), params)) is not None
            ),
            None,
        )
        prompt_tokens: Final = standard_logging_object.get("prompt_tokens")
        if (
            request_key is not None
            and model_id is not None
            and isinstance(prompt_tokens, int)
            and prompt_tokens >= get_prompt_cache_min_tokens(model)
        ):
            await self.cache.async_set_cache(request_key, model_id, ttl=300)

        if logged_messages is None or not isinstance(logged_messages, list):
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MESSAGES IS NOT A LIST"
            )
            return
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
            return

        native_messages: Final = kwargs.get("_prompt_caching_messages")
        messages: Final = (
            (
                native_messages
                if isinstance(native_messages, list)
                else _with_anthropic_system(cast(list[AllMessageValues], logged_messages), kwargs.get("system"))
            )
            if call_type == CallTypes.anthropic_messages.value
            else logged_messages
        )
        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if await offload_token_count(is_prompt_caching_valid_prompt)(
            model=model,
            messages=cast(list[AllMessageValues], messages),
        ):
            await self.prompt_cache.async_add_model_id(
                model_id=model_id,
                messages=messages,
                tools=None,  # [TODO]: add tools once standard_logging_object supports it
            )

        return
