import asyncio
import json
import socketserver
import threading
from contextlib import contextmanager

import pytest

import litellm
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens, ManagedChatGPTAccessToken
from litellm.types.utils import CredentialItem


@pytest.fixture(autouse=True)
async def flush_request_logging(isolate_litellm_state):
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    yield
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


class _Proxy(socketserver.StreamRequestHandler):
    def handle(self):
        request_line = self.rfile.readline().decode("ascii")
        while self.rfile.readline() not in (b"\r\n", b""):
            pass
        self.server.seen.append(request_line)
        body = self.server.body
        self.wfile.write(
            f"HTTP/1.1 {self.server.status_code}\r\nContent-Type: ".encode()
            + self.server.content_type
            + b"\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )


@contextmanager
def _proxy(body: bytes, content_type: bytes = b"application/json", status_code: int = 200):
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Proxy)
    server.daemon_threads = True
    server.body = body
    server.content_type = content_type
    server.status_code = status_code
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _credential(name: str, server) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_info={},
        credential_values={"litellm_internal_proxy_url": f"http://127.0.0.1:{server.server_address[1]}"},
    )


@pytest.mark.asyncio
async def test_chatgpt_responses_and_native_messages_use_named_proxy(monkeypatch):
    completed_response = {
        "id": "resp_proxy",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5",
        "output": [{
            "id": "msg_proxy",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "proxied", "annotations": []}],
        }],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    responses_body = (
        "event: response.completed\ndata: "
        + json.dumps({"type": "response.completed", "sequence_number": 1, "response": completed_response})
        + "\n\n"
    ).encode()
    messages_body = json.dumps(
        {
            "id": "msg_proxy",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "proxied"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()
    with _proxy(responses_body, b"text/event-stream") as responses_proxy, _proxy(messages_body) as messages_proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="chatgpt-account",
                    credential_info={"provider": "chatgpt", "auth_type": "oauth"},
                    credential_values={
                        **_credential("chatgpt-account", responses_proxy).credential_values,
                        "litellm_internal_chatgpt_auth_token": ChatGPTTokens(
                            "access-token", "refresh", "id", 9999999999, "account-id"
                        ).to_json(),
                    },
                ),
                _credential("claude-account", messages_proxy),
            ],
        )
        response_stream = await litellm.aresponses(
            model="chatgpt/gpt-5",
            input="hello",
            stream=True,
            api_key=ManagedChatGPTAccessToken("access-token"),
            chatgpt_auth_account_id="account-id",
            api_base="http://responses-upstream.invalid/backend-api/codex",
            litellm_credential_name="chatgpt-account",
        )
        events = [event async for event in response_stream]
        bridged = await litellm.acompletion(
            model="chatgpt/gpt-5.5",
            messages=[{"role": "user", "content": "hello"}],
            api_key=ManagedChatGPTAccessToken("access-token"),
            chatgpt_auth_account_id="account-id",
            api_base="http://responses-upstream.invalid/backend-api/codex",
            litellm_credential_name="chatgpt-account",
        )
        assert bridged.choices[0].message.content == "proxied"
        message = await litellm.anthropic.messages.acreate(
            model="anthropic/claude-test",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            api_key="sk-ant-test",
            api_base="http://messages-upstream.invalid/v1/messages",
            litellm_credential_name="claude-account",
        )

    assert sum(event.type == "response.completed" for event in events) == 1
    assert message["content"][0]["text"] == "proxied"
    assert responses_proxy.seen == [
        "POST http://responses-upstream.invalid/backend-api/codex/responses HTTP/1.1\r\n"
    ] * 2
    assert messages_proxy.seen == ["POST http://messages-upstream.invalid/v1/messages HTTP/1.1\r\n"]


class _CredentialHook:
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        return {**kwargs, "api_key": "sk-ant-test"}


@pytest.mark.asyncio
async def test_count_tokens_keeps_two_named_accounts_on_their_own_proxies(monkeypatch):
    with _proxy(b'{"input_tokens":11}') as first, _proxy(b'{"input_tokens":22}') as second:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [_credential("first", first), _credential("second", second)],
        )
        counter = AnthropicTokenCounter(oauth_credential_hook=_CredentialHook())
        one = await counter.count_tokens(
            "claude-test",
            [{"role": "user", "content": "one"}],
            None,
            deployment={
                "litellm_params": {
                    "litellm_credential_name": "first",
                    "api_base": "http://count-upstream.invalid/v1/messages/count_tokens",
                }
            },
        )
        two = await counter.count_tokens(
            "claude-test",
            [{"role": "user", "content": "two"}],
            None,
            deployment={
                "litellm_params": {
                    "litellm_credential_name": "second",
                    "api_base": "http://count-upstream.invalid/v1/messages/count_tokens",
                }
            },
        )

    assert one is not None and one.total_tokens == 11
    assert two is not None and two.total_tokens == 22
    assert len(first.seen) == len(second.seen) == 1


@pytest.mark.asyncio
async def test_count_tokens_resolves_cold_credential_before_selecting_proxy(monkeypatch):
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

    with _proxy(b'{"input_tokens":23}') as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(litellm, "credential_list", [])

        class LoadingHook:
            async def async_pre_call_deployment_hook(self, kwargs, call_type):
                CredentialAccessor.upsert_credentials([_credential("cold-account", proxy)])
                return {**kwargs, "api_key": "sk-ant-test"}

        result = await AnthropicTokenCounter(oauth_credential_hook=LoadingHook()).count_tokens(
            "claude-test",
            [{"role": "user", "content": "hello"}],
            None,
            deployment={
                "litellm_params": {
                    "litellm_credential_name": "cold-account",
                    "api_base": "http://count-upstream.invalid/v1/messages/count_tokens",
                }
            },
        )

    assert result is not None and not result.error, result
    assert result.total_tokens == 23
    assert proxy.seen == ["POST http://count-upstream.invalid/v1/messages/count_tokens HTTP/1.1\r\n"]


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_count_tokens_rechecks_proxy_after_token_recovery(monkeypatch, removed):
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.llms.anthropic.oauth_client import ManagedAnthropicOAuthToken

    with _proxy(b'{"error":{"type":"authentication_error"}}', status_code=401) as first, _proxy(
        b'{"input_tokens":29}'
    ) as second:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(litellm, "credential_list", [_credential("subscription", first)])

        class RecoveringHook:
            async def async_pre_call_deployment_hook(self, kwargs, call_type):
                return {**kwargs, "api_key": ManagedAnthropicOAuthToken("sk-ant-oat-rejected")}

            async def recover_rejected_token(self, credential_name, rejected_access_token):
                assert credential_name == "subscription"
                assert rejected_access_token == "sk-ant-oat-rejected"
                CredentialAccessor.upsert_credentials([
                    CredentialItem(
                        credential_name="subscription",
                        credential_info={"proxy_configured": True},
                        credential_values={} if removed else _credential("subscription", second).credential_values,
                    )
                ])
                return "sk-ant-oat-refreshed"

        hook = RecoveringHook()
        monkeypatch.setattr(litellm, "callbacks", [hook])
        result = await AnthropicTokenCounter(oauth_credential_hook=hook).count_tokens(
            "claude-test",
            [{"role": "user", "content": "hello"}],
            None,
            deployment={
                "litellm_params": {
                    "litellm_credential_name": "subscription",
                    "api_base": "http://count-upstream.invalid/v1/messages/count_tokens",
                }
            },
        )

    expected_request = "POST http://count-upstream.invalid/v1/messages/count_tokens HTTP/1.1\r\n"
    assert first.seen == [expected_request]
    assert second.seen == ([] if removed else [expected_request])
    assert result is not None and result.error is removed, result
    assert result.total_tokens == (0 if removed else 29)


@pytest.mark.asyncio
async def test_sync_responses_does_not_trust_unproxied_async_client(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    body = json.dumps({
        "id": "resp_proxy",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "test",
        "output": [],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
    }).encode()
    with _proxy(body) as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(litellm, "credential_list", [_credential("responses-account", proxy)])
        unproxied_client = AsyncHTTPHandler()
        try:
            response = await asyncio.to_thread(
                litellm.responses,
                model="openai/test",
                input="hello",
                api_key="test-key",
                api_base="http://responses-upstream.invalid/v1",
                litellm_credential_name="responses-account",
                client=unproxied_client,
                max_retries=0,
            )
        finally:
            await unproxied_client.close()

    assert response.status == "completed"
    assert response.output == []
    assert proxy.seen == ["POST http://responses-upstream.invalid/v1/responses HTTP/1.1\r\n"]


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_anthropic_api_surfaces_replace_direct_client_with_credential_proxy(monkeypatch, surface, stream):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    message = {
        "id": "msg_proxy",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "proxied"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = (
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "proxied"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    body = (
        "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        if stream else json.dumps(message)
    ).encode()
    with _proxy(body, b"text/event-stream" if stream else b"application/json") as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setattr(litellm, "credential_list", [_credential("subscription", proxy)])
        direct_client = AsyncHTTPHandler()
        params = {
            "model": "anthropic/claude-test",
            "api_key": "sk-ant-test",
            "api_base": "http://anthropic-upstream.invalid/v1/messages",
            "litellm_credential_name": "subscription",
            "client": direct_client,
            "stream": stream,
        }
        try:
            match surface:
                case "chat":
                    response = await litellm.acompletion(
                        **params, messages=[{"role": "user", "content": "hello"}], max_tokens=10,
                    )
                case "responses":
                    response = await litellm.aresponses(**params, input="hello", max_output_tokens=10)
                case "messages":
                    response = await litellm.anthropic.messages.acreate(
                        **params, messages=[{"role": "user", "content": "hello"}], max_tokens=10,
                    )
            if stream:
                assert any("proxied" in str(chunk) for chunk in [part async for part in response])
            else:
                match surface:
                    case "chat":
                        assert response.choices[0].message.content == "proxied"
                    case "responses":
                        assert response.output[0].content[0].text == "proxied"
                    case "messages":
                        assert response["content"][0]["text"] == "proxied"
        finally:
            await direct_client.close()

    assert proxy.seen == ["POST http://anthropic-upstream.invalid/v1/messages HTTP/1.1\r\n"]


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recovery", ["unchanged", "changed", "missing", "cancelled", "error", "rejected"])
async def test_public_anthropic_oauth_replay_validates_transport(monkeypatch, surface, stream, recovery):
    import httpx

    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.llms.anthropic.common_utils import ANTHROPIC_SUBSCRIPTION_BETA_HEADER
    from litellm.llms.anthropic.oauth_client import AnthropicOAuthTokens, ManagedAnthropicOAuthToken
    from litellm.llms.custom_httpx import http_handler
    from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_BETA_HEADER

    captured = []
    wires = []
    closed = []
    recovered = []
    proxy_url = "http://proxy.invalid:8080"
    identity = json.dumps({"user_id": "a" * 64, "account_id": "account", "session_id": "session"})
    message = {
        "id": "msg_retry", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "replayed"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = (
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "replayed"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )

    class TrackedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error":{"type":"authentication_error","message":"rejected"}}'

        async def aclose(self):
            closed.append(True)

    def upstream(request):
        wires.append(request.content)
        captured.append((dict(request.headers), json.loads(request.content)))
        if len(captured) == 1 or recovery == "rejected":
            return httpx.Response(401, stream=TrackedBody())
        body = (
            "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
            if stream else json.dumps(message)
        )
        return httpx.Response(200, content=body, headers={
            "content-type": "text/event-stream" if stream else "application/json",
        })

    class RecoveringHook:
        async def recover_rejected_token(self, credential_name, rejected_access_token):
            recovered.append((credential_name, rejected_access_token))
            if recovery == "cancelled":
                raise asyncio.CancelledError()
            if recovery == "error":
                raise RuntimeError("secret-refresh-detail")
            if recovery in ("changed", "missing"):
                CredentialAccessor.upsert_credentials([CredentialItem(
                    credential_name="subscription", credential_info={"proxy_configured": True},
                    credential_values={} if recovery == "missing" else {
                        "litellm_internal_proxy_url": "http://secret:password@changed.invalid:8080",
                    },
                )])
            return "sk-ant-oat-refreshed"

    monkeypatch.setattr(http_handler, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(upstream))
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="subscription", credential_info={"provider": "anthropic", "auth_type": "oauth"},
        credential_values={
            "litellm_internal_proxy_url": proxy_url,
            "litellm_internal_anthropic_auth_token": AnthropicOAuthTokens(
                "sk-ant-oat-rejected", "refresh", 9999999999, "account", "a" * 64,
            ).to_json(),
        },
    )])
    monkeypatch.setattr(litellm, "callbacks", [RecoveringHook()])

    async def invoke():
        params = {
            "model": "anthropic/claude-sonnet-5", "api_key": ManagedAnthropicOAuthToken("sk-ant-oat-rejected"),
            "litellm_credential_name": "subscription", "stream": stream, "max_retries": 0,
            "metadata": {"user_id": identity},
        }
        if surface == "chat":
            result = await litellm.acompletion(**params, messages=[{"role": "user", "content": "hello"}], max_tokens=10)
        elif surface == "responses":
            result = await litellm.aresponses(**params, input="hello", max_output_tokens=10)
        else:
            result = await litellm.anthropic_messages(
                **params, messages=[{"role": "user", "content": "hello"}], max_tokens=10,
            )
        if stream:
            return [part async for part in result]
        return result

    if recovery == "unchanged":
        result = await invoke()
        assert "replayed" in str(result)
    elif recovery == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await invoke()
    else:
        with pytest.raises(Exception) as caught:
            await invoke()
        assert caught.value.status_code == (401 if recovery == "rejected" else 503)
        assert "secret-refresh-detail" not in str(caught.value)
        assert "secret:password" not in str(caught.value)
    assert recovered == [("subscription", "sk-ant-oat-rejected")]
    assert len(captured) == (2 if recovery in ("unchanged", "rejected") else 1)
    assert closed == [True] * (2 if recovery == "rejected" else 1)
    assert [sorted(headers["anthropic-beta"].split(",")) for headers, _ in captured] == [
        sorted((ANTHROPIC_SUBSCRIPTION_BETA_HEADER, ANTHROPIC_OAUTH_BETA_HEADER))
    ] * len(captured)
    from litellm.llms.anthropic.subscription_billing import serialize_anthropic_subscription_request

    for wire in wires:
        assert b"cch=" in wire
        assert serialize_anthropic_subscription_request(json.loads(wire)).encode() == wire
    outgoing_identity = json.loads(captured[0][1]["metadata"]["user_id"])
    assert outgoing_identity["device_id"] == "a" * 64
    assert outgoing_identity["account_uuid"] == "account"
    assert outgoing_identity["session_id"] == captured[0][0]["x-claude-code-session-id"]
    if len(captured) == 2:
        assert captured[1][1] == captured[0][1]
        assert captured[0][0]["authorization"] == "Bearer sk-ant-oat-rejected"
        assert captured[1][0]["authorization"] == "Bearer sk-ant-oat-refreshed"
