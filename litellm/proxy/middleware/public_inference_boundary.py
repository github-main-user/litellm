import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.parse import urlsplit

from starlette.datastructures import QueryParams
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware.public_inference_ids import (
    STATE_KEY,
    InferenceIdStore,
    PublicInferenceIds,
    current_public_ids,
)

logger = logging.getLogger(__name__)
_HEADERS = frozenset(
    (
        b"content-type",
        b"retry-after",
        b"access-control-allow-origin",
        b"access-control-allow-credentials",
        b"access-control-allow-methods",
        b"access-control-allow-headers",
        b"access-control-max-age",
        b"vary",
        b"cache-control",
        b"strict-transport-security",
        b"x-content-type-options",
        b"x-frame-options",
        b"referrer-policy",
        b"content-security-policy",
    )
)
_MAX_ERROR = 65536
_MAX_FRAME = 50 * 1024 * 1024
_SAFE_ID = re.compile(r"resp_[a-fA-F0-9]{16,128}\Z")


def _route(scope: Scope) -> str | None:
    path = scope.get("path", "")
    for root in os.getenv("SERVER_ROOT_PATHS", "").split(","):
        root = root.strip().rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
            break
    for prefix in ("/v1/", "/openai/v1/"):
        if path == prefix[:-1] or path.startswith(prefix):
            return path[len(prefix) :]
    bare = path.removeprefix("/openai").lstrip("/")
    if bare.split("/", 1)[0] in (
        "responses",
        "containers",
        "files",
        "batches",
        "videos",
        "fine_tuning",
        "models",
        "chat",
        "completions",
        "messages",
        "embeddings",
        "images",
        "audio",
    ):
        return bare
    return None


def _headers(headers: list[tuple[bytes, bytes]], *, replacement: bool = False) -> list[tuple[bytes, bytes]]:
    allowed = (
        _HEADERS
        if replacement
        else _HEADERS
        | {
            b"content-length",
            b"content-encoding",
            b"content-disposition",
            b"content-range",
            b"accept-ranges",
            b"etag",
            b"last-modified",
        }
    )
    selected = [
        (k, v)
        for k, v in headers
        if k.lower() in allowed
        and (k.lower() != b"retry-after" or (v.isdigit() and len(v) <= 8))
        and (not replacement or k.lower() not in (b"content-type", b"cache-control"))
    ]
    if not replacement:
        for key, value in headers:
            if key.lower() != b"location":
                continue
            try:
                target = urlsplit(value)
            except (ValueError, UnicodeError):
                continue
            if target.path.startswith((b"/v1/", b"/openai/v1/")):
                selected.append((b"location", target.path + (b"?" + target.query if target.query else b"")))
    return selected


def _replacement_headers(headers: list[tuple[bytes, bytes]], sse: bool = False) -> list[tuple[bytes, bytes]]:
    return _headers(headers, replacement=True) + [
        (b"content-type", b"text/event-stream" if sse else b"application/json"),
        (b"cache-control", b"no-store"),
    ]


def _parse(raw: bytes) -> object | None:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        return None


def _details(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    error = value.get("error")
    if isinstance(error, dict):
        return error
    detail = value.get("detail")
    if isinstance(detail, dict):
        return _details(detail) or detail
    response = value.get("response")
    if isinstance(response, dict):
        return _details(response)
    return value


def _failure(value: object, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(value, list):
        return any(_failure(item, depth + 1) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("type") in ("error", "response.failed") or value.get("status") == "failed":
        return True
    if value.get("error") not in (None, {}, ""):
        return True
    if set(value).issubset({"message", "detail", "code", "type", "status"}) and any(
        isinstance(value.get(key), str) for key in ("message", "detail")
    ):
        return True
    return any(_failure(value.get(key), depth + 1) for key in ("response", "detail", "data", "output"))


def _category(status: int, value: object) -> tuple[int, str, str]:
    detail = _details(value)
    identifiers = " ".join(str(detail.get(k, "")) for k in ("type", "code", "param")).lower()
    outer = value if isinstance(value, dict) else {}
    identifiers += " " + " ".join(str(outer.get(k, "")) for k in ("type", "code")).lower()
    if "budget_exceeded" in identifiers or "budgetexceeded" in identifiers:
        return 429, "budget_exceeded", "Budget exceeded."
    message = detail.get("message", "")
    provider_auth = isinstance(message, str) and any(
        marker in message for marker in ("litellm.AuthenticationError", "litellm.PermissionDeniedError")
    )
    if provider_auth or any(marker in identifiers for marker in ("authenticationerror", "permissiondeniederror")):
        return 503, "service_unavailable_error", "Service temporarily unavailable."
    if status == 500:
        embedded_status = detail.get("status_code", detail.get("code"))
        if isinstance(embedded_status, (str, int)) and str(embedded_status).isdigit():
            candidate = int(embedded_status)
            if 400 <= candidate <= 599:
                status = candidate
    if "rate_limit" in identifiers:
        return 429, "rate_limit_error", "Rate limit exceeded. Please try again later."
    if status == 401:
        return status, "authentication_error", "Authentication failed."
    if status == 403:
        return status, "permission_error", "Permission denied."
    if status == 404:
        return status, "not_found_error", "Requested resource not found."
    if status == 429:
        return status, "rate_limit_error", "Rate limit exceeded. Please try again later."
    if status in (408, 504):
        return status, "timeout_error", "Request timed out. Please try again."
    if status in (400, 405, 413, 415, 422):
        return status, "invalid_request_error", "Invalid request."
    if status in (502, 503):
        return status, "service_unavailable_error", "Service temporarily unavailable."
    return status, "server_error", "Unable to process request."


def _error(status: int, value: object, anthropic: bool, brand: str) -> tuple[int, dict[str, object]]:
    status, kind, message = _category(status, value)
    public_message = f"{brand} - {message}"
    error: dict[str, object] = {"type": kind, "message": public_message}
    if anthropic:
        error["type"] = {
            "budget_exceeded": "rate_limit_error",
            "service_unavailable_error": "api_error",
            "timeout_error": "api_error",
            "server_error": "api_error",
        }.get(kind, kind)
        return status, {"type": "error", "error": error}
    return status, {"error": {**error, "param": None, "code": kind}}


def _encoded(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _event(value: object, event: bytes, anthropic: bool, brand: str, sequence_number: int = 0) -> bytes:
    _, payload = _error(500, value, anthropic, brand)
    if event == b"response.failed":
        response = value.get("response") if isinstance(value, dict) else None
        response_id = response.get("id") if isinstance(response, dict) else None
        safe = response_id if isinstance(response_id, str) and _SAFE_ID.fullmatch(response_id) else None
        _, code, message = _category(500, value)
        created = response.get("created_at") if isinstance(response, dict) else None
        failed: dict[str, object] = {
            "id": safe or f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": created if type(created) is int and 0 <= created <= 4102444800 else int(time.time()),
            "status": "failed",
            "output": [],
            "error": {"code": code, "message": f"{brand} - {message}"},
        }
        payload = {"type": "response.failed", "response": failed, "sequence_number": sequence_number}
    elif not anthropic:
        payload = {"type": "error", **payload}
    return b"event: " + event + b"\ndata: " + _encoded(payload) + b"\n\n"


def _frame(frame: bytes, anthropic: bool, brand: str, sequence_number: int, responses: bool) -> tuple[bytes, bool, int]:
    event = b""
    data: list[bytes] = []
    for line in frame.splitlines():
        if line.startswith(b"event:"):
            event = line[6:].strip()
        elif line.startswith(b"data:"):
            data.append(line[5:].strip())
    if (
        not data
        and event not in (b"error", b"response.failed")
        and all(not line or line.startswith((b":", b"id:", b"retry:", b"event:")) for line in frame.splitlines())
    ):
        return b": keep-alive\n\n", False, sequence_number
    raw = b"\n".join(data)
    value = _parse(raw) if raw and raw != b"[DONE]" else None
    observed = value.get("sequence_number") if isinstance(value, dict) else None
    latest = (
        max(sequence_number, observed) if type(observed) is int and 0 <= observed <= 1000000000 else sequence_number
    )
    failed = (
        event in (b"error", b"response.failed")
        or (bool(data) and raw != b"[DONE]" and (not isinstance(value, dict) or _failure(value)))
        or (not data and bool(frame.strip()))
    )
    if failed:
        is_response = (
            responses
            or event == b"response.failed"
            or (isinstance(value, dict) and value.get("type") == "response.failed")
        )
        name = b"response.failed" if is_response else b"error"
        next_sequence = max(latest, sequence_number + 1)
        return _event(value, name, anthropic, brand, next_sequence), True, next_sequence
    return frame, False, latest


def _frame_end(buf: bytearray) -> int:
    ends = [
        (i, width)
        for separator, width in ((b"\r\n\r\n", 4), (b"\n\n", 2), (b"\r\r", 2))
        if (i := buf.find(separator)) >= 0
    ]
    return min((i + width for i, width in ends), default=0)


async def _public_frame(frame: bytes, ids: PublicInferenceIds | None) -> bytes:
    if ids is None:
        return frame
    lines = frame.splitlines()
    raw = b"\n".join(line[5:].strip() for line in lines if line.startswith(b"data:"))
    value = _parse(raw)
    if not isinstance(value, dict):
        return frame
    payload = await ids.payload(value, incoming=False, resource=ids.resource)
    events = [line for line in lines if line.startswith(b"event:")]
    return b"\n".join([*events, b"data: " + _encoded(payload)]) + b"\n\n"


class PublicInferenceBoundary:
    def __init__(
        self, app: ASGIApp, brand: str = "VoidAPI", id_store_factory: Callable[[], InferenceIdStore] | None = None
    ) -> None:
        self.app = app
        self.brand = brand
        self.id_store_factory = id_store_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route = _route(scope)
        if scope["type"] not in ("http", "websocket") or route is None:
            await self.app(scope, receive, send)
            return
        ids = PublicInferenceIds(self.id_store_factory, self.brand) if self.id_store_factory is not None else None
        if ids is not None:
            scope.setdefault("state", {})[STATE_KEY] = ids
        token = current_public_ids.set(ids)
        anthropic = route.split("/", 1)[0] == "messages"
        try:
            if scope["type"] == "websocket":
                await self._websocket(scope, receive, send, anthropic)
            else:
                await self._http(scope, receive, send, route, anthropic)
        finally:
            current_public_ids.reset(token)

    async def _websocket(self, scope: Scope, receive: Receive, send: Send, anthropic: bool) -> None:
        closed = False
        denial: Message | None = None
        denial_body = bytearray()
        ids = scope.get("state", {}).get(STATE_KEY)
        session_model = QueryParams(scope.get("query_string", b"")).get("model")

        async def safe_receive() -> Message:
            nonlocal session_model
            message = await receive()
            if not isinstance(ids, PublicInferenceIds) or message["type"] != "websocket.receive":
                return message
            raw = message.get("text") if message.get("text") is not None else message.get("bytes")
            value = _parse(raw.encode() if isinstance(raw, str) else raw) if raw is not None else None
            if not isinstance(value, dict):
                return message
            nested = value.get("response", value)
            for frame in (value, nested):
                if not isinstance(frame, dict) or not isinstance(frame.get("model"), str):
                    continue
                if session_model is not None and frame["model"] != session_model:
                    raise ValueError("WebSocket model cannot change during a session")
                session_model = frame["model"]
            if session_model is not None:
                ids.model = session_model
            payload = _encoded(await ids.payload(value, incoming=True))
            return (
                {**message, "text": payload.decode(), "bytes": None}
                if isinstance(raw, str)
                else {**message, "bytes": payload, "text": None}
            )

        async def safe_send(message: Message) -> None:
            nonlocal closed, denial
            kind = message["type"]
            if closed:
                return
            if kind == "websocket.accept":
                await send({**message, "headers": _headers(message.get("headers", []), replacement=True)})
            elif kind == "websocket.close":
                closed = True
                await send({**message, "reason": f"{self.brand}: Connection closed."})
            elif kind == "websocket.http.response.start":
                denial = message
            elif kind == "websocket.http.response.body":
                if denial is None:
                    return
                chunk = message.get("body", b"")
                if len(denial_body) + len(chunk) <= _MAX_ERROR:
                    denial_body.extend(chunk)
                if not message.get("more_body", False):
                    status, payload = _error(denial["status"], _parse(denial_body), anthropic, self.brand)
                    closed = True
                    await send({**denial, "status": status, "headers": _replacement_headers(denial.get("headers", []))})
                    await send({"type": "websocket.http.response.body", "body": _encoded(payload)})
            elif kind == "websocket.send":
                raw = message.get("text") if message.get("text") is not None else message.get("bytes")
                if raw is None:
                    return
                value = _parse(raw.encode() if isinstance(raw, str) else raw)
                if isinstance(value, dict) and isinstance(ids, PublicInferenceIds):
                    value = await ids.payload(value, incoming=False, resource=ids.resource)
                    encoded = _encoded(value)
                    message = (
                        {**message, "text": encoded.decode(), "bytes": None}
                        if isinstance(raw, str)
                        else {**message, "bytes": encoded, "text": None}
                    )
                failed = not isinstance(value, dict) or _failure(value)
                if failed:
                    payload = _event(value, b"error", anthropic, self.brand).split(b"data: ", 1)[1].strip()
                    message = (
                        {**message, "text": payload.decode(), "bytes": None}
                        if isinstance(raw, str)
                        else {**message, "bytes": payload, "text": None}
                    )
                if failed:
                    closed = True
                await send(message)
                if failed:
                    await send({"type": "websocket.close", "code": 1011, "reason": f"{self.brand}: Connection closed."})
            else:
                await send(message)

        try:
            await self.app(scope, safe_receive, safe_send)
            if denial is not None and not closed:
                status, payload = _error(denial["status"], _parse(denial_body), anthropic, self.brand)
                await send({**denial, "status": status, "headers": _replacement_headers(denial.get("headers", []))})
                closed = True
                await send({"type": "websocket.http.response.body", "body": _encoded(payload)})
        except Exception:
            logger.exception("Public inference websocket failed")
            if not closed:
                if denial is not None:
                    await send(
                        {"type": "websocket.http.response.start", "status": 500, "headers": _replacement_headers([])}
                    )
                    await send(
                        {
                            "type": "websocket.http.response.body",
                            "body": _encoded(_error(500, None, anthropic, self.brand)[1]),
                        }
                    )
                else:
                    await send({"type": "websocket.close", "code": 1011, "reason": f"{self.brand}: Connection closed."})

    async def _http(self, scope: Scope, receive: Receive, send: Send, route: str, anthropic: bool) -> None:
        pending: Message | None = None
        started = False
        closed = False
        sse = False
        inspect = False
        status = 500
        buf = bytearray()
        stored: BinaryIO | None = None
        error_overflow = False
        sequence_number = -1
        context = scope.get("state", {}).get(STATE_KEY)
        ids = context if isinstance(context, PublicInferenceIds) else None

        async def send_start(new_status: int | None = None, replacement: bool = False) -> None:
            nonlocal started
            if pending is None or started:
                return
            headers = pending.get("headers", [])
            sanitized = _replacement_headers(headers, sse=sse and replacement) if replacement else _headers(headers)
            if sse or (inspect and ids is not None):
                sanitized = [
                    (k, v) for k, v in sanitized if k.lower() not in (b"content-length", b"content-encoding", b"etag")
                ]
            started = True
            await send({**pending, "status": new_status if new_status is not None else status, "headers": sanitized})

        async def finish_error(code: int, value: object = None, stream: bool = False) -> None:
            nonlocal closed
            actual, payload = _error(code, value, anthropic, self.brand)
            await send_start(actual, replacement=True)
            body = (
                _event(
                    value,
                    b"response.failed" if route.startswith("responses") else b"error",
                    anthropic,
                    self.brand,
                    sequence_number + 1,
                )
                if stream
                else _encoded(payload)
            )
            closed = True
            await send({"type": "http.response.body", "body": body, "more_body": False})

        async def safe_send(message: Message) -> None:
            nonlocal pending, status, sse, inspect, stored, closed, error_overflow, sequence_number
            kind = message["type"]
            if closed:
                return
            if kind == "http.response.start":
                pending = message
                status = message["status"]
                headers = message.get("headers", [])
                content_type = next((v.lower() for k, v in headers if k.lower() == b"content-type"), b"")
                compressed = any(k.lower() == b"content-encoding" for k, _ in headers)
                sse = b"text/event-stream" in content_type
                if (
                    300 <= status < 400
                    and status != 304
                    and not any(key == b"location" for key, _ in _headers(headers))
                ):
                    status = 502
                inspect = (
                    200 <= status < 300
                    and status != 204
                    and not sse
                    and scope.get("method") not in ("HEAD", "OPTIONS")
                    and (
                        route.startswith("chat/completions")
                        or route.split("/", 1)[0] in ("responses", "completions", "messages")
                        or (
                            ids is not None
                            and content_type.split(b";", 1)[0] == b"application/json"
                            and not route.endswith("/content")
                        )
                    )
                )
                if compressed and (status >= 400 or sse or inspect):
                    status = 502
                if status >= 400:
                    sse = False
                if inspect and status < 400:
                    stored = SpooledTemporaryFile(max_size=1024 * 1024)
                return
            if kind != "http.response.body":
                await send(message)
                return
            chunk = message.get("body", b"")
            more = message.get("more_body", False)
            if status >= 400:
                if not error_overflow and len(buf) + len(chunk) <= _MAX_ERROR:
                    buf.extend(chunk)
                else:
                    error_overflow = True
                    buf.clear()
                if not more:
                    await finish_error(status, _parse(buf) if not error_overflow else None)
                return
            if inspect and stored is not None:
                stored.write(chunk)
                if not more:
                    stored.seek(0)
                    data = stored.read()
                    stored.close()
                    stored = None
                    value = _parse(data)
                    if not isinstance(value, dict) or _failure(value):
                        await finish_error(500, value)
                    else:
                        body = (
                            _encoded(await ids.payload(value, incoming=False, resource=ids.resource))
                            if ids is not None
                            else data
                        )
                        await send_start()
                        closed = True
                        await send({"type": "http.response.body", "body": body, "more_body": False})
                return
            if not sse:
                await send_start()
                closed = not more
                await send(message)
                return
            buf.extend(chunk)
            while end := _frame_end(buf):
                if end > _MAX_FRAME:
                    await finish_error(500, stream=True)
                    return
                frame = bytes(buf[:end])
                del buf[:end]
                checked, failed, sequence_number = _frame(
                    await _public_frame(frame, ids),
                    anthropic,
                    self.brand,
                    sequence_number,
                    route.startswith("responses"),
                )
                await send_start()
                closed = failed
                await send({"type": "http.response.body", "body": checked, "more_body": not failed})
                if failed:
                    return
            if len(buf) > _MAX_FRAME:
                await finish_error(500, stream=True)
            elif not more:
                if buf:
                    checked, failed, sequence_number = _frame(
                        await _public_frame(bytes(buf) + b"\n\n", ids),
                        anthropic,
                        self.brand,
                        sequence_number,
                        route.startswith("responses"),
                    )
                    await send_start()
                    closed = failed
                    await send({"type": "http.response.body", "body": checked, "more_body": not failed})
                    if failed:
                        return
                await send_start()
                closed = True
                await send({"type": "http.response.body", "body": b"", "more_body": False})

        try:
            await self.app(scope, receive, safe_send)
            if pending is not None and not closed:
                if status >= 400 or not started:
                    await finish_error(500)
                elif sse:
                    await finish_error(500, stream=True)
                else:
                    closed = True
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
        except Exception:
            logger.exception("Public inference request failed")
            if not closed:
                if not started:
                    pending = {"type": "http.response.start", "status": 500, "headers": []}
                    status = 500
                    sse = False
                    await finish_error(500)
                elif sse:
                    await finish_error(500, stream=True)
                else:
                    closed = True
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            if stored is not None:
                stored.close()
