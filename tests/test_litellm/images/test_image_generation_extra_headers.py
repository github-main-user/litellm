"""
Unit test for https://github.com/BerriAI/litellm/issues/22285

Verifies that extra_headers passed to image_generation() are forwarded
to the OpenAI SDK on the openai/litellm_proxy/openai_compatible_providers
code paths.
"""

import json
import threading
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI


import litellm
from litellm.images.main import image_generation
from litellm.llms.custom_httpx.http_handler import CREDENTIAL_PROXY_TRUSTED, HTTPHandler
from litellm.types.utils import CredentialItem


class TestImageGenerationExtraHeaders:
    """Test that extra_headers are forwarded on the OpenAI code path."""

    @patch("litellm.images.main.openai_chat_completions")
    def test_extra_headers_forwarded_to_openai_image_generation(self, mock_openai_chat_completions):
        """
        extra_headers passed to image_generation() should appear in
        optional_params["extra_headers"] when the provider is openai.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        extra_headers = {"traceparent": "00-abc123-def456-01", "X-Custom": "value"}

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
            extra_headers=extra_headers,
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get("optional_params", call_kwargs[1].get("optional_params", {}))

        assert "extra_headers" in optional_params
        assert optional_params["extra_headers"] == extra_headers

    @patch("litellm.images.main.openai_chat_completions")
    def test_no_extra_headers_when_not_provided(self, mock_openai_chat_completions):
        """
        When extra_headers is not passed, optional_params should not
        contain extra_headers.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get("optional_params", call_kwargs[1].get("optional_params", {}))

        assert "extra_headers" not in optional_params


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("use_proxy", [False, True])
async def test_image_generation_uses_credential_proxy_without_serializing_it(is_async: bool, use_proxy: bool) -> None:
    proxy_url: Final = "http://proxy-user:proxy-password@proxy.invalid:8080"
    logging: Final = MagicMock()
    logging.async_failure_handler = AsyncMock()

    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"model": "dall-e-3", "prompt": "A red circle"}
        return httpx.Response(200, json={"created": 1234567890, "data": [{"b64_json": "aW1hZ2U="}]})

    transport: Final = httpx.MockTransport(respond)
    custom_client: Final = (
        AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        if is_async
        else OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    )
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPTransport"
        if is_async
        else "litellm.llms.custom_httpx.http_handler.HTTPTransport",
        return_value=transport,
    ) as proxy_transport:
        params: Final = {
            "model": "openai/dall-e-3",
            "prompt": "A red circle",
            "api_key": "test-key",
            "api_base": "https://upstream.invalid/v1",
            **(
                {"_credential_proxy_url": proxy_url, "_credential_proxy_trusted": CREDENTIAL_PROXY_TRUSTED}
                if use_proxy
                else {}
            ),
            "client": custom_client,
            "litellm_logging_obj": logging,
        }
        response: Final = await litellm.aimage_generation(**params) if is_async else litellm.image_generation(**params)
    assert response.data[0].b64_json == "aW1hZ2U="
    assert proxy_transport.call_count == int(use_proxy)
    if use_proxy:
        assert proxy_transport.call_args.kwargs["proxy"] == proxy_url
    assert "_credential_proxy" not in repr(logging.mock_calls)
    assert proxy_url not in repr(logging.mock_calls)
    assert not custom_client.is_closed()
    if isinstance(custom_client, AsyncOpenAI):
        await custom_client.close()
    else:
        custom_client.close()


def test_image_generation_rejects_unsupported_credential_proxy_transport() -> None:
    with patch("litellm.images.main.bfl_image_generation") as provider:
        with pytest.raises(litellm.APIConnectionError, match="Connection proxies are not supported"):
            image_generation(
                model="black_forest_labs/flux-pro",
                prompt="A red circle",
                _credential_proxy_url="http://proxy.invalid:8080",
                _credential_proxy_trusted=CREDENTIAL_PROXY_TRUSTED,
            )
    provider.image_generation.assert_not_called()


@pytest.mark.asyncio
async def test_named_no_proxy_image_generation_keeps_sync_client_off_event_loop() -> None:
    event_loop_thread: Final = threading.get_ident()

    def respond(request: httpx.Request) -> httpx.Response:
        assert threading.get_ident() != event_loop_thread
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"images": [{"image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        handler: Final = HTTPHandler(client=http_client)
        with patch.object(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="direct-images",
                    credential_info={},
                    credential_values={"api_key": "test-key"},
                )
            ],
        ):
            response: Final = await litellm.aimage_generation(
                model="openrouter/test-image-model",
                prompt="A red circle",
                api_base="https://upstream.invalid/v1",
                litellm_credential_name="direct-images",
                client=handler,
            )
        assert response.data[0].b64_json == "aW1hZ2U="
        assert handler.client is http_client
        assert not http_client.is_closed
