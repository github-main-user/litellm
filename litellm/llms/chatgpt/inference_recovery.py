from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import httpx

from litellm.litellm_core_utils.asyncify import (
    run_async_function,  # pyright: ignore[reportUnknownVariableType]  # shared sync bridge is untyped
)
from litellm.litellm_core_utils.credential_proxy import validate_credential_proxy_route

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ChatGPTInferenceRecoveryError(Exception):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class SyncInferenceClient(Protocol):
    proxy_url: str | None

    def post(
        self,
        *,
        url: str,
        headers: dict[str, object],  # mutable-ok: legacy HTTP handlers require dict headers
        data: bytes,
        timeout: float | httpx.Timeout,
        stream: bool,
    ) -> httpx.Response: ...


class AsyncInferenceClient(Protocol):
    proxy_url: str | None

    async def post(
        self,
        *,
        url: str,
        headers: dict[str, object],  # mutable-ok: legacy HTTP handlers require dict headers
        data: bytes,
        timeout: float | httpx.Timeout,
        stream: bool,
        logging_obj: "LiteLLMLoggingObj",
    ) -> httpx.Response: ...


def _validate_proxy_route(proxy_url: str | None, credential_name: object) -> None:
    try:
        validate_credential_proxy_route(proxy_url, credential_name)
    except ValueError:
        raise ChatGPTInferenceRecoveryError(
            status_code=503, message="ChatGPT credential proxy routing validation failed"
        ) from None


async def _recover_headers(
    headers: Mapping[str, object], params: Mapping[str, object], proxy_url: str | None
) -> Mapping[str, object] | None:
    credential_name: Final = params.get("litellm_credential_name")
    authorization_name, authorization = next(
        ((name, value) for name, value in headers.items() if name.lower() == "authorization"), ("", None)
    )
    if not isinstance(credential_name, str) or not credential_name or not isinstance(authorization, str):
        return None
    scheme, separator, rejected_token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not rejected_token:
        return None

    try:
        from litellm.proxy.credential_endpoints.chatgpt_oauth import get_chatgpt_oauth_credential_hook

        tokens: Final = await get_chatgpt_oauth_credential_hook().recover_after_unauthorized(
            credential_name, rejected_token
        )
    except Exception as error:  # noqa: BLE001  # OAuth callback failures must not expose token bundles
        status: Final = getattr(error, "status_code", 503)
        raise ChatGPTInferenceRecoveryError(
            status_code=status
            if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599
            else 503,
            message="ChatGPT OAuth recovery failed",
        ) from None
    if tokens is None:
        return None
    account_id: Final = next((value for name, value in headers.items() if name.lower() == "chatgpt-account-id"), None)
    if tokens.account_id != account_id:
        raise ChatGPTInferenceRecoveryError(status_code=401, message="ChatGPT OAuth recovery account mismatch")
    if not tokens.access_token or any(ord(char) < 33 or ord(char) > 126 for char in tokens.access_token):
        raise ChatGPTInferenceRecoveryError(
            status_code=502, message="ChatGPT OAuth recovery returned an invalid access token"
        )
    _validate_proxy_route(proxy_url, credential_name)
    return MappingProxyType(
        {
            name: f"Bearer {tokens.access_token}" if name == authorization_name else value
            for name, value in headers.items()
        }
    )


def post_chatgpt_with_recovery(
    *,
    client: SyncInferenceClient,
    url: str,
    headers: Mapping[str, object],
    body: bytes,
    timeout: float | httpx.Timeout,
    stream: bool,
    params: Mapping[str, object],
) -> httpx.Response:
    _validate_proxy_route(client.proxy_url, params.get("litellm_credential_name"))
    try:
        response: Final = client.post(
            url=url,
            headers=dict(headers),  # mutable-ok: legacy HTTP handler boundary
            data=body,
            timeout=timeout,
            stream=stream,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 401:
            raise
        try:
            refreshed_headers: Final = run_async_function(_recover_headers, headers, params, client.proxy_url)
        finally:
            error.response.close()
        if refreshed_headers is None:
            raise
    else:
        return response
    replay: Final = client.post(
        url=url,
        headers=dict(refreshed_headers),  # mutable-ok: legacy HTTP handler boundary
        data=body,
        timeout=timeout,
        stream=stream,
    )
    replay.raise_for_status()
    return replay


async def async_post_chatgpt_with_recovery(
    *,
    client: AsyncInferenceClient,
    url: str,
    headers: Mapping[str, object],
    body: bytes,
    timeout: float | httpx.Timeout,
    stream: bool,
    params: Mapping[str, object],
    logging_obj: "LiteLLMLoggingObj",
) -> httpx.Response:
    _validate_proxy_route(client.proxy_url, params.get("litellm_credential_name"))
    try:
        response: Final = await client.post(
            url=url,
            headers=dict(headers),  # mutable-ok: legacy HTTP handler boundary
            data=body,
            timeout=timeout,
            stream=stream,
            logging_obj=logging_obj,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 401:
            raise
        try:
            refreshed_headers: Final = await _recover_headers(headers, params, client.proxy_url)
        finally:
            await error.response.aclose()
        if refreshed_headers is None:
            raise
    else:
        return response
    replay: Final = await client.post(
        url=url,
        headers=dict(refreshed_headers),  # mutable-ok: legacy HTTP handler boundary
        data=body,
        timeout=timeout,
        stream=stream,
        logging_obj=logging_obj,
    )
    replay.raise_for_status()
    return replay
