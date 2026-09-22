import asyncio
import ipaddress
import json
import socketserver
import ssl
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

import litellm
from litellm.litellm_core_utils.credential_proxy import (
    get_credential_proxy_url,
    validate_proxy_url,
)
from litellm.llms.custom_httpx.http_handler import (
    CREDENTIAL_PROXY_TRUSTED,
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.types.utils import CredentialItem


class _Proxy(socketserver.StreamRequestHandler):
    def handle(self):
        request_line = self.rfile.readline().decode("ascii")
        headers = {}
        while True:
            line = self.rfile.readline().decode("latin1")
            if line in ("\r\n", ""):
                break
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
        self.server.seen.append((request_line, headers))
        content_length = int(headers.get("content-length", "0"))
        self.server.bodies.append(self.rfile.read(content_length))
        body = self.server.body
        self.wfile.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nContent-Type: "
            + self.server.content_type
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )


class _SocksProxy(_Proxy):
    def handle(self):
        version, count = self.rfile.read(2)
        assert version == 5
        assert 2 in self.rfile.read(count)
        self.wfile.write(b"\x05\x02")
        auth_version, username_length = self.rfile.read(2)
        assert auth_version == 1
        username = self.rfile.read(username_length)
        password_length = self.rfile.read(1)[0]
        password = self.rfile.read(password_length)
        self.server.auth.append((username, password))
        self.wfile.write(b"\x01\x00")
        version, command, reserved, address_type = self.rfile.read(4)
        assert (version, command, reserved, address_type) == (5, 1, 0, 3)
        hostname_length = self.rfile.read(1)[0]
        hostname = self.rfile.read(hostname_length)
        port = int.from_bytes(self.rfile.read(2), "big")
        self.server.targets.append((hostname, port))
        self.wfile.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
        super().handle()


@contextmanager
def _proxy(body=b"proxied", content_type=b"application/json", handler=_Proxy, tls_context=None):
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    if tls_context is not None:
        server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    server.daemon_threads = True
    server.seen = []
    server.bodies = []
    server.auth = []
    server.targets = []
    server.body = body
    server.content_type = content_type
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://proxy:21",
        "http://",
        "http://proxy/path",
        "http://proxy?q=secret",
        "http://proxy#fragment",
        " http://proxy",
        "http://proxy:99999",
        "http://user%0apass@proxy",
        "http://proxy/%zz",
    ],
)
def test_validate_proxy_url_rejects_without_echoing_secret(url):
    with pytest.raises(ValueError) as exc:
        validate_proxy_url(url)
    assert url not in str(exc.value)


def test_validate_proxy_url_supports_authenticated_socks_and_http():
    assert validate_proxy_url("HTTP://u:p@proxy.example:8080") == "http://u:p@proxy.example:8080"
    assert validate_proxy_url("socks5h://u:p@[::1]:1080") == "socks5h://u:p@[::1]:1080"


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["socks5", "socks5h"])
async def test_socks_authentication_and_dns_use_only_configured_proxy(monkeypatch, scheme):
    with _proxy(handler=_SocksProxy) as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        handler = AsyncHTTPHandler(
            proxy_url=f"{scheme}://proxy-user:proxy-password@127.0.0.1:{proxy.server_address[1]}"
        )
        try:
            response = await handler.get("http://upstream.invalid/test")
            assert response.content == b"proxied"
        finally:
            await handler.close()
        assert proxy.auth == [(b"proxy-user", b"proxy-password")]
        assert proxy.targets == [(b"upstream.invalid", 80)]
        assert proxy.seen[0][0].startswith("GET /test ")
        assert "proxy-authorization" not in proxy.seen[0][1]


def test_configured_but_missing_proxy_fails_closed(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [CredentialItem(credential_name="broken", credential_info={"proxy_configured": True}, credential_values={})],
    )
    with pytest.raises(ValueError, match="proxy configuration is unavailable"):
        get_credential_proxy_url("broken")


def test_https_proxy_uses_verified_tls(monkeypatch, tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Local test proxy")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "proxy.pem"
    key_path = tmp_path / "proxy.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    original_factory = ssl.create_default_context

    def trusted_context(*args, **kwargs):
        context = original_factory(*args, **kwargs)
        context.load_verify_locations(cafile=cert_path)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        return context

    monkeypatch.setattr(ssl, "create_default_context", trusted_context)
    monkeypatch.setenv("NO_PROXY", "*")
    with _proxy(tls_context=server_context) as proxy:
        handler = HTTPHandler(proxy_url=f"https://user:password@127.0.0.1:{proxy.server_address[1]}")
        try:
            assert handler.get("http://upstream.invalid/test").content == b"proxied"
        finally:
            handler.close()
        assert len(proxy.seen) == 1
        assert proxy.seen[0][1]["proxy-authorization"].startswith("Basic ")


def test_unverified_provider_proxy_fails_closed(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="unsupported",
        credential_info={"provider": "vertex_ai", "proxy_configured": True},
        credential_values={"litellm_internal_proxy_url": "http://proxy.invalid:8080"},
    )])
    with pytest.raises(ValueError, match="not supported"):
        get_credential_proxy_url("unsupported")


def test_named_credentials_are_isolated(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="one",
                credential_info={},
                credential_values={"litellm_internal_proxy_url": "http://one.invalid:8001"},
            ),
            CredentialItem(
                credential_name="two",
                credential_info={},
                credential_values={"litellm_internal_proxy_url": "socks5h://two.invalid:8002"},
            ),
            CredentialItem(credential_name="none", credential_info={}, credential_values={}),
        ],
    )
    assert get_credential_proxy_url("one") == "http://one.invalid:8001"
    assert get_credential_proxy_url("two") == "socks5h://two.invalid:8002"
    assert get_credential_proxy_url("none") is None
    assert get_credential_proxy_url("missing") is None


def test_router_rejects_missing_named_credential(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [])
    router = litellm.Router(
        model_list=[
            {
                "model_name": "named",
                "litellm_params": {
                    "model": "openai/test",
                    "litellm_credential_name": "missing",
                },
            }
        ]
    )
    deployment = router.get_deployment_by_model_group_name("named")
    with pytest.raises(ValueError, match="was not found"):
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs={"metadata": {}, "_credential_proxy_url": "http://attacker.invalid:1"},
        )


def test_sync_proxy_is_only_route_even_with_no_proxy(monkeypatch):
    with _proxy() as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        url = f"http://user:password@127.0.0.1:{proxy.server_address[1]}"
        handler = HTTPHandler(proxy_url=url)
        try:
            # This host cannot be reached directly; success proves proxy routing.
            response = handler.get("http://upstream.invalid/test")
        finally:
            handler.close()
        assert response.content == b"proxied"
        request_line, headers = proxy.seen[0]
        assert request_line.startswith("GET http://upstream.invalid/test ")
        assert headers["proxy-authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_async_proxy_is_only_route_even_with_no_proxy(monkeypatch):
    with _proxy(b"async-proxied") as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        handler = AsyncHTTPHandler(proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}")
        try:
            response = await handler.get("http://upstream.invalid/test")
        finally:
            await handler.close()
        assert response.content == b"async-proxied"


def test_openai_compatible_inference_uses_credential_transport(monkeypatch):
    payload = json.dumps(
        {
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "via proxy"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
    with _proxy(payload) as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        response = litellm.completion(
            model="openai/test",
            messages=[{"role": "user", "content": "hello"}],
            api_key="not-secret",
            api_base="http://upstream.invalid/v1",
            max_retries=0,
            _credential_proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}",
            _credential_proxy_trusted=CREDENTIAL_PROXY_TRUSTED,
        )
    assert response.choices[0].message.content == "via proxy"
    assert proxy.seen[0][0].startswith("POST http://upstream.invalid/v1/chat/completions ")


def test_router_named_credential_uses_proxy_without_leaking_proxy_secret(monkeypatch):
    payload = json.dumps(
        {
            "id": "chatcmpl-router-proxy",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "router"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
    with _proxy(payload) as proxy:
        proxy_url = f"http://proxy-user:proxy-password@127.0.0.1:{proxy.server_address[1]}"
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="named",
                    credential_info={},
                    credential_values={
                        "api_key": "provider-key",
                        "litellm_internal_proxy_url": proxy_url,
                    },
                )
            ],
        )
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "group",
                    "litellm_params": {
                        "model": "openai/test",
                        "api_base": "http://upstream.invalid/v1",
                        "litellm_credential_name": "named",
                        "max_retries": 0,
                    },
                }
            ]
        )
        response = router.completion(model="group", messages=[{"role": "user", "content": "hello"}])
    assert response.choices[0].message.content == "router"
    assert proxy.seen[0][0].startswith("POST http://upstream.invalid/v1/chat/completions ")
    assert b"proxy-password" not in proxy.bodies[0]
    assert b"litellm_internal_proxy_url" not in proxy.bodies[0]


def test_azure_inference_uses_credential_transport(monkeypatch):
    payload = json.dumps(
        {
            "id": "chatcmpl-azure-proxy",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "azure"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
    with _proxy(payload) as proxy:
        monkeypatch.setenv("NO_PROXY", "*")
        response = litellm.completion(
            model="azure/deployment",
            messages=[{"role": "user", "content": "hello"}],
            api_key="not-secret",
            api_base="http://upstream.invalid",
            api_version="2024-01-01",
            max_retries=0,
            _credential_proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}",
            _credential_proxy_trusted=CREDENTIAL_PROXY_TRUSTED,
        )
    assert response.choices[0].message.content == "azure"
    assert proxy.seen[0][0].startswith(
        "POST http://upstream.invalid/openai/deployments/deployment/chat/completions?"
    )


@pytest.mark.asyncio
async def test_two_named_transports_stream_concurrently_and_configuration_changes(monkeypatch):
    def stream_payload(value):
        chunk = {
            "id": f"chatcmpl-{value}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
        }
        done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode()

    async def call(proxy, value):
        stream = await litellm.acompletion(
            model="openai/test",
            messages=[{"role": "user", "content": "hello"}],
            api_key=f"key-{value}",
            api_base="http://upstream.invalid/v1",
            stream=True,
            max_retries=0,
            _credential_proxy_url=f"http://user-{value}:pass@127.0.0.1:{proxy.server_address[1]}",
            _credential_proxy_trusted=CREDENTIAL_PROXY_TRUSTED,
        )
        parts = []
        async for item in stream:
            content = item.choices[0].delta.content
            if content:
                parts.append(content)
        return "".join(parts)

    monkeypatch.setenv("NO_PROXY", "*")
    with _proxy(stream_payload("one"), b"text/event-stream") as first, _proxy(
        stream_payload("two"), b"text/event-stream"
    ) as second:
        assert await asyncio.gather(call(first, "one"), call(second, "two")) == ["one", "two"]
        assert first.seen[0][1]["proxy-authorization"].startswith("Basic ")
        assert first.seen[0][1]["proxy-authorization"] != second.seen[0][1]["proxy-authorization"]
        assert second.seen[0][0].startswith("POST http://upstream.invalid/v1/chat/completions ")

        # A changed credential route is used immediately; no SDK/client cache can retain the first proxy.
        assert await call(second, "two") == "two"
        assert len(first.seen) == 1
        assert len(second.seen) == 2


def test_client_cannot_select_internal_proxy(monkeypatch):
    captured = {}

    def fake_completion(*args, **kwargs):
        captured.update(kwargs)
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm.main.openai_chat_completions, "completion", fake_completion)
    litellm.completion(
        model="openai/test",
        messages=[{"role": "user", "content": "hello"}],
        api_key="key",
        _credential_proxy_url="http://attacker.invalid:8080",
        litellm_internal_proxy_url="http://attacker.invalid:8081",
    )
    assert not isinstance(captured.get("client"), (HTTPHandler, AsyncHTTPHandler))
    assert "litellm_internal_proxy_url" not in captured.get("litellm_params", {})


def test_proxy_failure_does_not_fall_back_directly():
    # Reserve and release a local port. The resulting refused connection must be
    # reported rather than triggering a direct request to the target.
    import httpx

    sock = __import__("socket").socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    handler = HTTPHandler(proxy_url=f"http://127.0.0.1:{port}", timeout=0.2)
    try:
        with pytest.raises(httpx.TransportError):
            handler.get("http://127.0.0.1:1/direct-must-not-run")
    finally:
        handler.close()
