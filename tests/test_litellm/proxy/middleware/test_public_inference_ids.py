import base64
import json
import secrets
from collections.abc import Callable

import httpx
import pytest
from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.types import Message, Scope

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
from litellm.proxy.middleware.public_inference_boundary import PublicInferenceBoundary
from litellm.proxy.middleware.public_inference_ids import STATE_KEY, PublicInferenceIds, current_public_ids
from litellm.types.llms.openai import ResponsesAPIResponse


class MemoryIds:
    def __init__(self) -> None:
        self.by_source: dict[tuple[str, str, str], str] = {}
        self.by_public: dict[tuple[str, str, str], str] = {}
        self.unavailable = False

    async def publish(
        self, owner: str, kind: str, value: str, *, identity: str | None = None, replace: bool = False
    ) -> str:
        if self.unavailable:
            raise RuntimeError("database unavailable")
        key = (owner, kind, identity or value)
        if key not in self.by_source:
            public = f"{dict(response='resp_', reasoning='enc_', item='item_', container='cntr_').get(kind, kind + '_')}{secrets.token_hex(16)}"
            self.by_source[key] = public
            self.by_public[(owner, kind, public)] = value
        elif replace:
            self.by_public[(owner, kind, self.by_source[key])] = value
        return self.by_source[key]

    async def resolve(self, owner: str, kind: str, public_id: str) -> str | None:
        if self.unavailable:
            raise RuntimeError("database unavailable")
        return self.by_public.get((owner, kind, public_id))


def wrapped_response() -> str:
    route = "litellm:custom_llm_provider:openai;model_id:private-deployment;response_id:resp_provider-123"
    return "resp_" + base64.b64encode(route.encode()).decode()


def authenticated(user_id: str = "alice", team_id: str = "team", api_key: str = "sk-test") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, team_id=team_id, api_key=api_key)


async def http_exchange(
    factory: Callable[[], MemoryIds],
    user: UserAPIKeyAuth,
    request_body: dict[str, object],
    upstream: dict[str, object] | None = None,
    *,
    path: str = "/v1/responses",
    query: str = "",
    path_params: dict[str, str] | None = None,
) -> tuple[list[Message], dict[str, object]]:
    sent: list[Message] = []
    observed: dict[str, object] = {}

    async def app(scope: Scope, receive, send) -> None:
        context: PublicInferenceIds = scope["state"][STATE_KEY]
        request = Request(scope)
        request._json = request_body.copy()
        data = request_body.copy()
        try:
            await context.authorize_request(request, user, data)
        except HTTPException as exc:
            await send(
                {
                    "type": "http.response.start",
                    "status": exc.status_code,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'"denied"'})
            return
        observed.update(
            body=data, cached=request._json, path=request.path_params, query=request.query_params.multi_items()
        )
        result = (
            upstream
            if upstream is not None
            else {"object": "response", "id": wrapped_response(), "model": "private-model"}
        )
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(result).encode()})

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": query.encode(),
        "path_params": path_params or {},
        "headers": [],
        "state": {},
    }
    await PublicInferenceBoundary(app, id_store_factory=factory)(scope, lambda: None, send)
    return sent, observed


def response(messages: list[Message]) -> dict[str, object]:
    return json.loads(b"".join(item.get("body", b"") for item in messages[1:]))


@pytest.mark.asyncio
async def test_responses_roundtrip_keeps_internal_routing_and_leaves_client_data_alone() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    encrypted = "litellm_enc:bW9kZWxfaWQ6cHJpdmF0ZQ==;provider-secret"
    item_id = "encitem_" + base64.b64encode(b"litellm:model_id:private-deployment;item_id:reason-1").decode()
    original = {
        "object": "response",
        "id": internal,
        "model": "internal-model",
        "output": [{"type": "reasoning", "id": item_id, "encrypted_content": encrypted}],
        "_hidden_params": {"api_key": "secret"},
    }
    messages, _ = await http_exchange(lambda: store, authenticated(), {"model": "public-model"}, original)
    public = response(messages)
    assert public["id"].startswith("resp_") and public["id"] != internal
    assert public["model"] == "public-model"
    assert public["output"][0]["id"].startswith("item_")
    assert public["output"][0]["encrypted_content"].startswith("enc_")
    assert "private-deployment" not in json.dumps(public) and "secret" not in json.dumps(public)

    client_text = f"keep {internal} {encrypted} {item_id} unchanged"
    body = {
        "model": "public-model",
        "previous_response_id": public["id"],
        "input": [
            {
                "type": "reasoning",
                "id": public["output"][0]["id"],
                "encrypted_content": public["output"][0]["encrypted_content"],
            },
            {"role": "user", "content": [{"type": "input_text", "text": client_text}]},
            {"type": "function_call_output", "call_id": item_id, "output": client_text},
        ],
        "instructions": client_text,
        "metadata": {"id": internal, "encrypted_content": encrypted},
    }
    _, seen = await http_exchange(lambda: store, authenticated(), body)
    restored = seen["body"]
    assert restored["previous_response_id"] == internal
    assert restored["input"][0] == original["output"][0]
    assert restored["input"][1] == body["input"][1]
    assert restored["input"][2] == body["input"][2]
    assert restored["instructions"] == client_text and restored["metadata"] == body["metadata"]
    assert seen["cached"] == restored


@pytest.mark.asyncio
async def test_itemless_reasoning_content_roundtrips_after_context_restart() -> None:
    store = MemoryIds()
    internal = "litellm_enc:cm91dGU=;ciphertext"
    original = {
        "object": "response",
        "id": wrapped_response(),
        "output": [{"type": "reasoning", "encrypted_content": internal}],
    }
    first, _ = await http_exchange(lambda: store, authenticated(), {}, original)
    public = response(first)["output"][0]["encrypted_content"]
    assert public.startswith("enc_") and public != internal
    _, observed = await http_exchange(
        lambda: store, authenticated(), {"input": [{"type": "reasoning", "encrypted_content": public}]}
    )
    assert observed["body"]["input"] == original["output"]


@pytest.mark.asyncio
async def test_unknown_expired_cross_owner_tampered_and_legacy_ids_are_denied() -> None:
    store = MemoryIds()
    messages, _ = await http_exchange(lambda: store, authenticated(), {"model": "public"})
    public = response(messages)["id"]
    assert isinstance(public, str)
    other = authenticated("bob")
    for user, candidate in (
        (other, public),
        (authenticated(), "resp_" + secrets.token_hex(16)),
        (authenticated(), public[:-1] + ("0" if public[-1] != "0" else "1")),
        (authenticated(), wrapped_response()),
        (authenticated(), "encitem_" + base64.b64encode(b"litellm:model_id:private;item_id:reason").decode()),
    ):
        denied, seen = await http_exchange(lambda: store, user, {"previous_response_id": candidate})
        assert denied[0]["status"] in (400, 404) and not seen
        assert wrapped_response().encode() not in repr(denied).encode()
    store.by_public.clear()
    expired, _ = await http_exchange(lambda: store, authenticated(), {"previous_response_id": public})
    assert expired[0]["status"] == 404


@pytest.mark.asyncio
async def test_path_and_query_use_restored_ids_without_touching_unrelated_query() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    item = "item_provider-123"
    issued = response(
        (
            await http_exchange(
                lambda: store,
                authenticated(),
                {},
                {
                    "object": "response",
                    "id": internal,
                    "output": [{"type": "reasoning", "id": item}],
                },
            )
        )[0]
    )
    public = issued["id"]
    cursor = issued["output"][0]["id"]
    sent, seen = await http_exchange(
        lambda: store,
        authenticated(),
        {},
        path=f"/v1/responses/{public}/input_items",
        path_params={"response_id": public},
        query=f"after={cursor}&after={cursor}&label=resp_example",
    )
    assert sent[0]["status"] == 200
    assert seen["path"] == {"response_id": internal}
    assert seen["query"] == [("after", item), ("after", item), ("label", "resp_example")]


@pytest.mark.asyncio
async def test_streamed_events_share_opaque_ids_across_arbitrary_chunks() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    raw_item = "reason-9"
    item = "encitem_" + base64.b64encode(b"litellm:model_id:private-deployment;item_id:reason-9").decode()
    encrypted = "litellm_enc:cm91dGU=;ciphertext"
    events = [
        {"type": "response.created", "response": {"object": "response", "id": internal, "model": "secret-model"}},
        {"type": "response.output_item.added", "response_id": internal, "item": {"type": "reasoning", "id": raw_item}},
        {
            "type": "response.reasoning_summary_text.delta",
            "response_id": internal,
            "item_id": raw_item,
            "delta": "client-visible text",
        },
        {
            "type": "response.output_item.done",
            "response_id": internal,
            "item": {"type": "reasoning", "id": item, "encrypted_content": encrypted},
            "item_id": item,
        },
        {
            "type": "response.completed",
            "response": {
                "object": "response",
                "id": internal,
                "output": [{"type": "reasoning", "id": item, "encrypted_content": encrypted}],
            },
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "response_id": internal,
            "item_id": raw_item,
            "delta": "still visible",
        },
    ]
    stream = b"".join(b"data: " + json.dumps(event).encode() + b"\r\n\r\n" for event in events)
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        context: PublicInferenceIds = scope["state"][STATE_KEY]
        await context.authorize_request(Request(scope), authenticated(), {"model": "public-model"})
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        for offset in range(0, len(stream), 7):
            await send({"type": "http.response.body", "body": stream[offset : offset + 7], "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def send(message: Message) -> None:
        sent.append(message)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {"type": "http", "path": "/v1/responses", "method": "POST", "query_string": b"", "headers": [], "state": {}},
        lambda: None,
        send,
    )
    output = b"".join(part.get("body", b"") for part in sent[1:])
    decoded = [json.loads(frame.split(b"data: ", 1)[1]) for frame in output.split(b"\n\n") if b"data: " in frame]
    public_id = decoded[0]["response"]["id"]
    assert all(event.get("response_id", event.get("response", {}).get("id")) == public_id for event in decoded)
    public_item = decoded[1]["item"]["id"]
    assert public_item.startswith("item_")
    assert (
        decoded[2]["item_id"]
        == decoded[3]["item_id"]
        == decoded[4]["response"]["output"][0]["id"]
        == decoded[5]["item_id"]
        == public_item
    )
    assert decoded[3]["item"]["id"] == public_item
    assert decoded[3]["item"]["encrypted_content"].startswith("enc_")
    assert decoded[2]["delta"] == "client-visible text"
    assert internal.encode() not in output and item.encode() not in output and encrypted.encode() not in output
    # A later raw publication must not overwrite the final routable wrapper in storage.
    owner = next(owner for owner, kind, public in store.by_public if kind == "item" and public == public_item)
    assert (
        await store.publish(owner, "item", raw_item, identity=json.dumps(["private-deployment", raw_item]))
        == public_item
    )
    assert await store.resolve(owner, "item", public_item) == item
    _, seen = await http_exchange(
        lambda: store,
        authenticated(),
        {
            "previous_response_id": public_id,
            "input": [
                {"type": "reasoning", "id": public_item, "encrypted_content": decoded[3]["item"]["encrypted_content"]}
            ],
        },
    )
    assert seen["body"]["previous_response_id"] == internal
    assert seen["body"]["input"] == [{"type": "reasoning", "id": item, "encrypted_content": encrypted}]


@pytest.mark.asyncio
async def test_websocket_flat_and_nested_response_create_share_context() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    public = response(
        (await http_exchange(lambda: store, authenticated(), {}, {"object": "response", "id": internal}))[0]
    )["id"]
    incoming = [
        {"type": "response.create", "previous_response_id": public, "model": "public"},
        {"type": "response.create", "response": {"previous_response_id": public, "model": "public"}},
    ]
    observed: list[dict[str, object]] = []
    sent: list[Message] = []
    queue = iter([{"type": "websocket.receive", "text": json.dumps(value)} for value in incoming])

    async def app(scope: Scope, receive, send) -> None:
        context: PublicInferenceIds = scope["state"][STATE_KEY]
        context.bind(authenticated())
        await send({"type": "websocket.accept"})
        for _ in incoming:
            received = json.loads((await receive())["text"])
            observed.append(received)
            await send(
                {
                    "type": "websocket.send",
                    "text": json.dumps(
                        {
                            "type": "response.created",
                            "response": {"object": "response", "id": internal, "model": "private"},
                        }
                    ),
                }
            )

    async def receive() -> Message:
        return next(queue)

    async def send(message: Message) -> None:
        sent.append(message)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {"type": "websocket", "path": "/v1/responses", "query_string": b"", "headers": [], "state": {}},
        receive,
        send,
    )
    assert observed[0]["previous_response_id"] == observed[1]["response"]["previous_response_id"] == internal
    outgoing = [json.loads(message["text"]) for message in sent if message["type"] == "websocket.send"]
    assert len(outgoing) == 2
    assert all(event["response"]["id"] == public and event["response"]["model"] == "public" for event in outgoing)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [b"", b"model=allowed"])
@pytest.mark.parametrize("nested", [False, True])
async def test_websocket_rejects_model_switch_before_forwarding(query: bytes, nested: bool) -> None:
    store = MemoryIds()
    disallowed = {"model": "disallowed"}
    frames = ([] if query else [{"type": "response.create", "model": "allowed"}]) + [
        {"type": "response.create", "response": disallowed} if nested else {"type": "response.create", **disallowed}
    ]
    queue = iter(frames)
    forwarded: list[object] = []
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        await send({"type": "websocket.accept"})
        for _ in frames:
            forwarded.append(json.loads((await receive())["text"]))

    async def receive() -> Message:
        return {"type": "websocket.receive", "text": json.dumps(next(queue))}

    async def send(message: Message) -> None:
        sent.append(message)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {"type": "websocket", "path": "/v1/responses", "query_string": query, "headers": [], "state": {}},
        receive,
        send,
    )
    assert forwarded == ([] if query else [frames[0]])
    assert sent[-1]["type"] == "websocket.close"
    assert "disallowed" not in json.dumps(sent)


@pytest.mark.asyncio
async def test_missing_store_or_publish_failure_never_emits_internal_ids() -> None:
    store = MemoryIds()
    store.unavailable = True

    def missing_store() -> MemoryIds:
        raise RuntimeError("database unavailable")

    internal = wrapped_response()
    for factory in (missing_store, lambda: store):
        sent, _ = await http_exchange(factory, authenticated(), {}, {"object": "response", "id": internal})
        assert sent[0]["status"] >= 500
        assert internal.encode() not in repr(sent).encode()
        assert b"private-deployment" not in repr(sent).encode()


@pytest.mark.asyncio
async def test_same_user_new_key_allowed_but_other_team_denied() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    public = response(
        (await http_exchange(lambda: store, authenticated(), {}, {"object": "response", "id": internal}))[0]
    )["id"]
    allowed, observed = await http_exchange(
        lambda: store, authenticated(api_key="sk-other"), {"previous_response_id": public}
    )
    assert allowed[0]["status"] == 200 and observed["body"]["previous_response_id"] == internal
    denied, observed = await http_exchange(
        lambda: store, authenticated(team_id="other", api_key="sk-other"), {"previous_response_id": public}
    )
    assert denied[0]["status"] == 404 and not observed


@pytest.mark.asyncio
async def test_nested_annotation_part_and_container_ids_roundtrip() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    container = "cntr_" + base64.b64encode(b"litellm:private-container").decode()
    file = "file-" + base64.b64encode(b"litellm:private-file").decode()
    original = {
        "object": "response",
        "id": internal,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [
                            {"type": "container_file_citation", "container_id": container, "file_id": file},
                        ],
                    },
                    {"part": {"container_id": container, "file_id": file}},
                ],
            }
        ],
    }
    sent, _ = await http_exchange(lambda: store, authenticated(), {}, original)
    public = response(sent)
    annotation = public["output"][0]["content"][0]["annotations"][0]
    part = public["output"][0]["content"][1]["part"]
    assert annotation["container_id"].startswith("cntr_") and annotation["container_id"] != container
    assert annotation["file_id"].startswith("file_") and annotation["file_id"] != file
    assert part == {"container_id": annotation["container_id"], "file_id": annotation["file_id"]}
    _, seen = await http_exchange(lambda: store, authenticated(), {"input": public["output"]})
    assert seen["body"]["input"] == original["output"]


@pytest.mark.asyncio
async def test_responses_security_hook_trusts_only_resolved_ids_inside_boundary() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    public = response(
        (await http_exchange(lambda: store, authenticated(), {}, {"object": "response", "id": internal}))[0]
    )["id"]
    hook = ResponsesIDSecurity(general_settings_reader=lambda: {}, signing_key_reader=lambda: "configured")
    observations: dict[str, object] = {}

    async def app(scope: Scope, receive, send) -> None:
        context: PublicInferenceIds = scope["state"][STATE_KEY]
        user = authenticated()
        assert current_public_ids.get() is context
        data = {"previous_response_id": public}
        await context.authorize_request(Request(scope), user, data)
        result = await hook.async_pre_call_hook(user, None, data, "aresponses")
        observations["pre"] = result
        raw = ResponsesAPIResponse(id=internal, created_at=1234567890, output=[], status="completed")
        observations["post"] = (await hook.async_post_call_success_hook(data, user, raw)).id
        unowned = "resp_" + base64.b64encode(b"litellm:model_id:other;response_id:resp_unowned").decode()
        try:
            await hook.async_pre_call_hook(user, None, {"previous_response_id": unowned}, "aresponses")
        except HTTPException as exc:
            observations["denied"] = exc.status_code
        else:
            observations["denied"] = None
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps({"object": "response", "id": raw.id}).encode()})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {"type": "http", "path": "/v1/responses", "method": "POST", "query_string": b"", "headers": [], "state": {}},
        lambda: None,
        send,
    )
    assert sent[0]["status"] == 200
    assert observations["pre"]["previous_response_id"] == internal
    assert observations["post"] == internal
    assert observations["denied"] == 403
    assert response(sent)["id"] == public
    assert current_public_ids.get() is None


@pytest.mark.asyncio
async def test_fastapi_dependency_resolves_parsed_body_path_and_query() -> None:
    store = MemoryIds()
    internal = wrapped_response()
    item = "item_provider-123"
    issued = response(
        (
            await http_exchange(
                lambda: store,
                authenticated(),
                {},
                {
                    "object": "response",
                    "id": internal,
                    "output": [{"type": "reasoning", "id": item}],
                },
            )
        )[0]
    )

    class InputBody(BaseModel):
        previous_response_id: str = Field(alias="previous_response_id")

    api = FastAPI()
    observed: dict[str, str] = {}

    async def authorize(request: Request) -> None:
        context: PublicInferenceIds = request.scope["state"][STATE_KEY]
        await context.authorize_request(request, authenticated(), await request.json())

    @api.post("/v1/responses/{response_id}/input_items", dependencies=[Depends(authorize)])
    async def handler(
        payload: InputBody = Body(),
        response_id: str = Path(),
        after: str = Query(),
    ) -> dict[str, str]:
        observed.update(previous_response_id=payload.previous_response_id, response_id=response_id, after=after)
        return observed

    transport = httpx.ASGITransport(app=PublicInferenceBoundary(api, id_store_factory=lambda: store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post(
            f"/v1/responses/{issued['id']}/input_items",
            params={"after": issued["output"][0]["id"]},
            json={"previous_response_id": issued["id"]},
        )
    assert result.status_code == 200, result.text
    assert observed == {"previous_response_id": internal, "response_id": internal, "after": item}
    assert result.json() == {
        "previous_response_id": issued["id"],
        "response_id": issued["id"],
        "after": issued["output"][0]["id"],
    }


@pytest.mark.asyncio
async def test_identical_item_ids_from_different_deployments_do_not_change_affinity() -> None:
    store = MemoryIds()
    first = PublicInferenceIds(lambda: store, "VoidAPI")
    second = PublicInferenceIds(lambda: store, "VoidAPI")
    first.bind(authenticated())
    second.bind(authenticated())
    one = "encitem_" + base64.b64encode(b"litellm:model_id:account-one;item_id:same-id").decode()
    two = "encitem_" + base64.b64encode(b"litellm:model_id:account-two;item_id:same-id").decode()
    public_one = await first.identifier("item", one, incoming=False)
    public_two = await second.identifier("item", two, incoming=False)
    assert public_one != public_two
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    assert await continued.identifier("item", public_one, incoming=True) == one
    assert await continued.identifier("item", public_two, incoming=True) == two


@pytest.mark.asyncio
async def test_anthropic_reasoning_bridge_is_opaque_but_native_signatures_and_text_stay_unchanged() -> None:
    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    signature = "litellm_encrypted_reasoning:litellm_enc:cm91dGU=;provider-ciphertext"
    thinking = {"type": "thinking", "thinking": "text mentioning litellm_enc: unchanged", "signature": signature}
    redacted = {"type": "redacted_thinking", "data": signature}
    native = {"type": "thinking", "thinking": "native reasoning", "signature": "native-provider-signature"}
    result = await context.payload({"content": [thinking, redacted, native]}, incoming=False)
    public = result["content"][0]["signature"]
    assert public.startswith("enc_") and "litellm" not in public
    assert result["content"][0]["thinking"] == thinking["thinking"]
    assert result["content"][1]["data"] == public
    assert result["content"][2] == native
    delta = await context.payload({"delta": {"type": "signature_delta", "signature": signature}}, incoming=False)
    assert delta["delta"]["signature"] == public
    request = {"messages": [{"role": "assistant", "content": result["content"]}]}
    restored = await context.payload(request, incoming=True)
    assert restored["messages"][0]["content"] == [thinking, redacted, native]


@pytest.mark.asyncio
async def test_container_file_list_cursors_match_ids_and_restore_for_pagination() -> None:
    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    scope: Scope = {
        "type": "http",
        "path": "/v1/containers/container-native/files",
        "method": "GET",
        "headers": [],
        "query_string": b"",
        "path_params": {},
    }
    await context.authorize_request(Request(scope), authenticated(), {})
    original = {
        "object": "list",
        "data": [{"object": "container.file", "id": "cfile-native", "path": "litellm.txt"}],
        "first_id": "cfile-native",
        "last_id": "cfile-native",
        "has_more": False,
    }
    public = await context.payload(original, incoming=False, resource=context.resource)
    cursor = public["data"][0]["id"]
    assert cursor.startswith("file_")
    assert public["first_id"] == public["last_id"] == cursor
    assert public["data"][0]["path"] == "litellm.txt"
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    request = Request({**scope, "query_string": f"after={cursor}".encode()})
    await continued.authorize_request(request, authenticated(), {})
    assert request.query_params["after"] == "cfile-native"
