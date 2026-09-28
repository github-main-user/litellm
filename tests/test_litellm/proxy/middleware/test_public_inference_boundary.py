import asyncio
import json

import pytest

from litellm.proxy.middleware.public_inference_boundary import PublicInferenceBoundary


def invoke(messages, path="/v1/chat/completions", kind="http", error=None, method="POST"):
    sent = []

    async def app(scope, receive, send):
        for message in messages:
            await send(message)
        if error is not None:
            raise error

    async def send(message):
        sent.append(message)

    asyncio.run(
        PublicInferenceBoundary(app, "VoidAPI")({"type": kind, "path": path, "method": method}, lambda: None, send)
    )
    return sent


def start(status=200, content=b"application/json", extra=()):
    return {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", content),
            (b"x-litellm-model-id", b"secret-route"),
            (b"x-upstream-debug", b"account-123"),
            (b"retry-after", b"4"),
            *extra,
        ],
    }


def body(payload, more=False):
    return {"type": "http.response.body", "body": payload, "more_body": more}


def payload(messages):
    return b"".join(item.get("body", b"") for item in messages[1:])


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/new-inference-route", "/openai/v1/future"])
def test_all_public_routes_sanitize_errors(path):
    result = invoke([start(503), body(b'{"error":{"message":"provider key secret"}}')], path)
    assert result[0]["status"] == 503
    assert json.loads(payload(result))["error"]["message"] == "VoidAPI - Service temporarily unavailable."
    assert (b"retry-after", b"4") in result[0]["headers"]
    assert all(b"litellm" not in k and b"debug" not in k for k, _ in result[0]["headers"])
    assert b"secret" not in payload(result)


@pytest.mark.parametrize(
    "error,status,kind,text",
    [
        (
            {"error": {"type": "budget_exceeded", "message": "private budget"}},
            429,
            "budget_exceeded",
            "Budget exceeded.",
        ),
        (
            {"error": {"type": "rate_limit_error", "message": "private rate"}},
            429,
            "rate_limit_error",
            "Rate limit exceeded. Please try again later.",
        ),
        (
            {"error": {"type": "auth_error", "message": "invalid virtual key"}},
            401,
            "authentication_error",
            "Authentication failed.",
        ),
        (
            {"error": {"type": "litellm.AuthenticationError", "message": "upstream key"}},
            503,
            "service_unavailable_error",
            "Service temporarily unavailable.",
        ),
        (
            {"detail": {"error": {"type": "budget_exceeded", "message": "nested"}}},
            429,
            "budget_exceeded",
            "Budget exceeded.",
        ),
    ],
)
def test_bounded_error_classification(error, status, kind, text):
    upstream = 401 if status == 503 else status
    result = invoke([start(upstream), body(json.dumps(error).encode())])
    assert result[0]["status"] == status
    assert json.loads(payload(result))["error"] == {
        "message": f"VoidAPI - {text}",
        "type": kind,
        "param": None,
        "code": kind,
    }


def test_anthropic_and_bounded_malformed_error():
    failure = invoke([start(400, b"text/plain"), body(b"password secret")], "/v1/messages")
    assert json.loads(payload(failure)) == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "VoidAPI - Invalid request.",
        },
    }
    large = invoke([start(502), body(b"provider-secret" * 10000)], "/v1/chat/completions")
    assert b"provider-secret" not in payload(large)


def test_binary_success_keeps_encoding_and_length_without_size_limit():
    image = b"\xff" * (9 * 1024 * 1024)
    result = invoke(
        [
            start(200, b"image/png", ((b"content-length", str(len(image)).encode()), (b"content-encoding", b"gzip"))),
            body(image),
        ],
        "/v1/images/generations",
    )
    assert payload(result) == image
    assert (b"content-length", str(len(image)).encode()) in result[0]["headers"]
    assert (b"content-encoding", b"gzip") in result[0]["headers"]
    assert all(k != b"x-upstream-debug" for k, _ in result[0]["headers"])


def test_compressed_error_body_never_passes_through():
    result = invoke(
        [
            start(401, extra=((b"content-encoding", b"gzip"), (b"content-length", b"999"))),
            body(b"compressed private credentials"),
        ]
    )
    assert result[0]["status"] == 502
    assert b"credentials" not in payload(result)
    assert all(k not in (b"content-encoding", b"content-length") for k, _ in result[0]["headers"])


def test_prefixed_public_path_is_covered(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATHS", "/gateway")
    result = invoke([start(500), body(b"private")], "/gateway/v1/new-route")
    assert b"private" not in payload(result)


def test_success_json_is_untouched_and_success_error_is_replaced():
    success = b'{"choices":[{"message":{"content":"LiteLLM is a library"}}]}'
    result = invoke([start(extra=((b"content-length", str(len(success)).encode()),)), body(success)])
    assert payload(result) == success
    assert (b"content-length", str(len(success)).encode()) in result[0]["headers"]
    failure = invoke([start(extra=((b"content-length", b"100"),)), body(b'{"error":{"message":"private provider"}}')])
    assert b"private provider" not in payload(failure)
    assert all(k != b"content-length" for k, _ in failure[0]["headers"])


def test_sse_split_crlf_failed_response_keeps_safe_id():
    stream = (
        b'event: response.failed\r\ndata: {"type":"response.failed","response":'
        b'{"id":"resp_0123456789abcdef","status":"failed","model":"internal-route",'
        b'"error":{"message":"provider route abc"}}}\r\n\r\n'
    )
    result = invoke(
        [
            start(200, b"text/event-stream"),
            body(b'event: response.output_text.delta\r\ndata: {"delta":"hi"}\r\n\r\n' + stream[:27], True),
            body(stream[27:43], True),
            body(stream[43:]),
        ],
        "/openai/v1/responses",
    )
    out = payload(result)
    assert b'"delta":"hi"' in out
    assert b"provider route" not in out and b"internal-route" not in out
    assert b'"id":"resp_0123456789abcdef"' in out
    assert b'"status":"failed"' in out
    assert b'"object":"response"' in out
    assert result[-1]["more_body"] is False


@pytest.mark.parametrize(
    "tail",
    [
        b"event: error\ndata: not-json",
        b'event: response.failed\ndata: {"response":{"id":"secret-account"}}',
        b"provider credentials",
    ],
)
def test_unterminated_stream_tail_fails_closed(tail):
    result = invoke([start(200, b"text/event-stream"), body(tail)], "/v1/responses")
    assert b"credentials" not in payload(result) and b"secret-account" not in payload(result)
    assert b"VoidAPI" in payload(result)


def test_oversized_complete_frame_rejected_before_output(monkeypatch):
    monkeypatch.setattr("litellm.proxy.middleware.public_inference_boundary._MAX_FRAME", 128)
    huge = b'event: response.output_text.delta\ndata: {"delta":"' + b"secret" * 50 + b'"}\n\n'
    result = invoke([start(200, b"text/event-stream"), body(huge)], "/v1/responses")
    assert b"secret" not in payload(result)
    assert result[-1]["more_body"] is False


def test_failure_frame_terminates_stream_without_leaking_following_frames():
    result = invoke(
        [
            start(200, b"text/event-stream"),
            body(b'event: error\ndata: private provider\n\ndata: {"delta":"private after"}\n\n'),
        ],
        "/v1/chat/completions",
    )
    assert b"private" not in payload(result)
    assert sum(message.get("more_body") is False for message in result[1:]) == 1


def test_invalid_stream_payload_cannot_escape():
    result = invoke([start(200, b"text/event-stream"), body(b"data: provider-secret\n\n")], "/v1/chat/completions")
    assert b"provider-secret" not in payload(result)
    assert b"VoidAPI" in payload(result)


def test_websocket_accept_denial_and_failure():
    result = invoke(
        [
            {"type": "websocket.accept", "headers": [(b"x-debug", b"secret"), (b"retry-after", b"4")]},
            {"type": "websocket.send", "text": '{"type":"error","message":"account secret"}'},
            {"type": "websocket.close", "code": 1011, "reason": "provider route"},
        ],
        "/v1/realtime",
        "websocket",
    )
    assert result[0]["headers"] == [(b"retry-after", b"4")]
    assert "secret" not in result[1]["text"]
    assert "route" not in result[2]["reason"]
    denial = invoke(
        [
            {"type": "websocket.http.response.start", "status": 401, "headers": [(b"x-debug", b"secret")]},
            {
                "type": "websocket.http.response.body",
                "body": b'{"error":{"type":"litellm.AuthenticationError","message":"private"}}',
            },
        ],
        "/v1/realtime",
        "websocket",
    )
    assert denial[0]["status"] == 503
    assert b"private" not in denial[1]["body"]
    limited = invoke(
        [
            {"type": "websocket.http.response.start", "status": 429, "headers": []},
            {"type": "websocket.http.response.body", "body": b'{"error":{"type":"budget_exceeded"}}'},
        ],
        "/v1/realtime",
        "websocket",
    )
    assert limited[0]["status"] == 429
    assert json.loads(limited[1]["body"])["error"]["message"] == "VoidAPI - Budget exceeded."
    malformed = invoke(
        [{"type": "websocket.accept"}, {"type": "websocket.send", "text": "provider-secret"}],
        "/v1/realtime",
        "websocket",
    )
    assert "provider-secret" not in malformed[1]["text"]


def test_exception_before_start_and_late_stream_logged(caplog):
    early = invoke([], error=RuntimeError("sensitive exception"))
    assert early[0]["status"] == 500
    assert b"sensitive" not in payload(early)
    late = invoke(
        [start(200, b"text/event-stream"), body(b'data: {"delta":"ok"}\n\n', True)],
        "/v1/responses",
        error=RuntimeError("late secret"),
    )
    assert late[0]["status"] == 200
    assert late[-1]["more_body"] is False
    assert b"late secret" not in payload(late)
    assert "sensitive exception" in caplog.text
    assert "late secret" in caplog.text


def test_exception_after_terminal_send_does_not_duplicate(caplog):
    result = invoke([start(), body(b'{"choices":[]}')], error=RuntimeError("private exception"))
    assert len(result) == 2
    assert "private exception" in caplog.text


def test_non_public_unaffected():
    result = invoke([start(500), body(b"internal")], "/internal/admin")
    assert payload(result) == b"internal"


def test_sse_heartbeat_preserves_following_success():
    result = invoke(
        [
            start(200, b"text/event-stream"),
            body(b": internal heartbeat\n\n", True),
            body(b'data: {"delta":"still connected"}\n\n'),
        ]
    )
    assert b"still connected" in payload(result)
    assert b"internal heartbeat" not in payload(result)
    assert b"VoidAPI" not in payload(result)


def test_preflight_preserves_cors_and_plain_body():
    result = invoke(
        [
            start(
                200,
                b"text/plain",
                (
                    (b"access-control-allow-methods", b"POST"),
                    (b"access-control-allow-headers", b"Authorization, Content-Type"),
                ),
            ),
            body(b"OK"),
        ],
        method="OPTIONS",
    )
    assert result[0]["status"] == 200
    assert (b"access-control-allow-methods", b"POST") in result[0]["headers"]
    assert (b"access-control-allow-headers", b"Authorization, Content-Type") in result[0]["headers"]
    assert payload(result) == b"OK"


@pytest.mark.parametrize("raw", [b"litellm.AuthenticationError: secret", b'{"message":"secret credential"}'])
def test_mislabeled_successful_error_is_not_exposed(raw):
    result = invoke([start(200, b"text/plain"), body(raw)])
    assert result[0]["status"] == 500
    assert b"secret" not in payload(result)


def test_anthropic_budget_uses_supported_error_type():
    result = invoke([start(429), body(b'{"error":{"type":"budget_exceeded"}}')], "/v1/messages")
    assert json.loads(payload(result))["error"] == {
        "type": "rate_limit_error",
        "message": "VoidAPI - Budget exceeded.",
    }


def test_redirect_only_exposes_public_relative_path():
    result = invoke(
        [
            start(307, extra=((b"location", b"http://internal-host/v1/responses"),)),
            body(b""),
        ]
    )
    assert result[0]["status"] == 307
    assert (b"location", b"/v1/responses") in result[0]["headers"]
    assert b"internal-host" not in repr(result).encode()


def test_failed_response_id_cannot_expose_account_name():
    result = invoke(
        [
            start(200, b"text/event-stream"),
            body(
                b'event: response.failed\ndata: {"type":"response.failed","response":'
                b'{"id":"resp_private-credential","status":"failed","error":{"message":"secret"}}}\n\n'
            ),
        ],
        "/v1/responses",
    )
    assert b"private-credential" not in payload(result)
    assert b'"object":"response"' in payload(result)


def test_generated_responses_failure_advances_sequence():
    result = invoke(
        [
            start(200, b"text/event-stream"),
            body(
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta",'
                b'"sequence_number":17,"delta":"ok"}\n\n',
                True,
            ),
        ],
        "/v1/responses",
        error=RuntimeError("internal failure"),
    )
    frames = payload(result).split(b"\n\n")
    terminal = json.loads(next(frame.split(b"data: ", 1)[1] for frame in frames if b"response.failed" in frame))
    assert terminal["sequence_number"] == 18
    assert terminal["response"]["status"] == "failed"
    assert "internal failure" not in repr(terminal)
