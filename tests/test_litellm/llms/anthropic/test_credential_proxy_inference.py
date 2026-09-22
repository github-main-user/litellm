import json
import socketserver
import threading
from contextlib import contextmanager

import pytest

import litellm
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.types.utils import CredentialItem


class _Proxy(socketserver.StreamRequestHandler):
    def handle(self):
        request_line = self.rfile.readline().decode("ascii")
        while self.rfile.readline() not in (b"\r\n", b""):
            pass
        self.server.seen.append(request_line)
        body = self.server.body
        self.wfile.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: "
            + self.server.content_type
            + b"\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )


@contextmanager
def _proxy(body: bytes, content_type: bytes = b"application/json"):
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Proxy)
    server.daemon_threads = True
    server.body = body
    server.content_type = content_type
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
            [_credential("chatgpt-account", responses_proxy), _credential("claude-account", messages_proxy)],
        )
        response_stream = await litellm.aresponses(
            model="chatgpt/gpt-5",
            input="hello",
            api_key="access-token",
            chatgpt_auth_account_id="account-id",
            api_base="http://responses-upstream.invalid/backend-api/codex",
            litellm_credential_name="chatgpt-account",
        )
        events = [event async for event in response_stream]
        bridged = await litellm.acompletion(
            model="chatgpt/gpt-5.5",
            messages=[{"role": "user", "content": "hello"}],
            api_key="access-token",
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
