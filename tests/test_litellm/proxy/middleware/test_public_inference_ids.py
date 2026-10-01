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
            public = f"{dict(response='resp_', reasoning='enc_', item='item_', tool='call_', container='cntr_').get(kind, kind + '_')}{secrets.token_hex(16)}"
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
    method: str = "POST",
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
        "method": method,
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
@pytest.mark.parametrize("native_id", [False, True])
async def test_response_retrieval_recovers_public_model_without_request_model(native_id: bool) -> None:
    store = MemoryIds()
    created, _ = await http_exchange(lambda: store, authenticated(), {"model": "public-alias"})
    public = response(created)["id"]
    upstream = {
        "object": "response",
        "id": "resp_provider-123" if native_id else wrapped_response(),
        "model": "private-deployment-name",
        "metadata": {"model": "user-value"},
    }
    sent, seen = await http_exchange(
        lambda: store,
        authenticated(),
        {},
        upstream,
        path=f"/v1/responses/{public}",
        path_params={"response_id": public},
        method="GET",
    )
    assert sent[0]["status"] == 200
    assert response(sent) == {**upstream, "id": public, "model": "public-alias"}
    assert seen["path"]["response_id"] == wrapped_response()
    assert seen["body"] == {}
    denied, _ = await http_exchange(
        lambda: store,
        authenticated(user_id="bob"),
        {},
        upstream,
        path=f"/v1/responses/{public}",
        path_params={"response_id": public},
        method="GET",
    )
    assert denied[0]["status"] == 404
    assert "private-deployment-name" not in json.dumps(response(sent))


@pytest.mark.asyncio
async def test_previous_response_alias_does_not_override_requested_model() -> None:
    store = MemoryIds()
    created, _ = await http_exchange(lambda: store, authenticated(), {"model": "old-alias"})
    public = response(created)["id"]
    sent, seen = await http_exchange(
        lambda: store, authenticated(), {"model": "new-alias", "previous_response_id": public}
    )
    assert sent[0]["status"] == 200
    assert response(sent)["model"] == "new-alias"
    assert seen["body"] == {"model": "new-alias", "previous_response_id": wrapped_response()}
    fetched, _ = await http_exchange(
        lambda: store,
        authenticated(),
        {},
        path=f"/v1/responses/{public}",
        path_params={"response_id": public},
        method="GET",
    )
    assert response(fetched)["model"] == "old-alias"


@pytest.mark.asyncio
async def test_response_without_saved_alias_fails_closed_instead_of_exposing_upstream_model() -> None:
    store = MemoryIds()
    issuer = PublicInferenceIds(lambda: store, "VoidAPI")
    issuer.bind(authenticated())
    public = await issuer.identifier("response", wrapped_response(), incoming=False)
    sent, _ = await http_exchange(
        lambda: store,
        authenticated(),
        {},
        path=f"/v1/responses/{public}",
        path_params={"response_id": public},
        method="GET",
    )
    assert sent[0]["status"] >= 400
    assert b"private-model" not in b"".join(item.get("body", b"") for item in sent)


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
            {"type": "function_call_output", "call_id": "call_native-example", "output": client_text},
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
                {"model": "public-model"},
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
        (
            await http_exchange(
                lambda: store, authenticated(), {"model": "public-model"}, {"object": "response", "id": internal}
            )
        )[0]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_public_usage_removes_cache_extensions_without_changing_standard_fields(transport: str, api: str) -> None:
    store = MemoryIds()
    details_key = "prompt_tokens_details" if api == "chat" else "input_tokens_details"
    usage = {
        ("prompt_tokens" if api == "chat" else "input_tokens"): 25,
        ("completion_tokens" if api == "chat" else "output_tokens"): 10,
        "total_tokens": 35,
        details_key: {
            "cached_tokens": 7,
            "audio_tokens": 0,
            "text_tokens": 20,
            "cache_write_tokens": 5,
            "cache_creation_tokens": 5,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 5},
        },
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 7,
    }
    client_metadata = {"cache_creation_input_tokens": "client-owned"}
    payload = {"usage": usage, "metadata": client_metadata}
    if api == "responses":
        payload.update(object="response", id=wrapped_response(), output=[{"type": "message", "content": []}])
    else:
        payload.update(choices=[{"message": {"content": "cache_creation_token_details stays in text"}}])
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        context = scope["state"][STATE_KEY]
        context.bind(authenticated())
        context.resource = "response" if api == "responses" else None
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(message: Message) -> None:
        sent.append(message)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/responses" if api == "responses" else "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(message["text"] for message in sent if message["type"] == "websocket.send"))
    else:
        body = b"".join(message.get("body", b"") for message in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    expected = {
        key: value
        for key, value in usage.items()
        if key not in ("cache_creation_input_tokens", "cache_read_input_tokens")
    }
    expected[details_key] = {"cached_tokens": 7, "audio_tokens": 0, "text_tokens": 20}
    assert result["usage"] == expected
    assert result["metadata"] == client_metadata
    if api == "chat":
        assert result["choices"] == payload["choices"]
    else:
        assert result["output"] == payload["output"]
        assert result["id"] != payload["id"]


@pytest.mark.asyncio
async def test_cache_cleanup_leaves_incoming_usage_and_native_anthropic_usage_unchanged() -> None:
    context = PublicInferenceIds(lambda: MemoryIds(), "VoidAPI")
    context.bind(authenticated())
    native = {"type": "message", "usage": {"input_tokens": 5, "output_tokens": 2, "cache_creation_input_tokens": 3}}
    assert await context.payload(native, incoming=False) == native
    incoming = {"usage": {"prompt_tokens": 5, "cache_creation_input_tokens": 3}}
    assert await context.payload(incoming, incoming=True) == incoming


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
@pytest.mark.parametrize("unique", [False, True])
async def test_provider_extensions_remove_only_empty_and_duplicate_fields(transport: str, unique: bool) -> None:
    store = MemoryIds()
    thinking = [{"type": "thinking", "thinking": "Reasoning", "signature": "native-signature"}]
    fields = {"citations": None, "thinking_blocks": thinking, "empty_results": [], "native_finish_reason": "end_turn"}
    extra = {
        "web_search_results": [{"url": "https://example.com", "title": "Source"}],
        "signature": "native-signature",
        "enabled": False,
        "count": 0,
    }
    if unique:
        fields.update(extra)
    message = {"content": "Answer", "thinking_blocks": thinking, "provider_specific_fields": fields}
    payload = {"choices": [{"message": message}], "metadata": {"provider_specific_fields": fields}}
    before = json.dumps(payload)
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    expected = {"content": "Answer", "thinking_blocks": thinking}
    if unique:
        expected["provider_specific_fields"] = {"signature": "native-signature"}
    assert result["choices"][0]["message"] == expected
    assert result["metadata"] == payload["metadata"]
    assert json.dumps(payload) == before
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    assert await context.payload(payload, incoming=True) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("fields", [None, {}, {"citations": None, "thinking_blocks": []}])
async def test_empty_provider_wrapper_is_removed_without_touching_tool_output(fields: object) -> None:
    context = PublicInferenceIds(lambda: MemoryIds(), "VoidAPI")
    payload = {"choices": [{"delta": {"content": "text", "provider_specific_fields": fields}}]}
    assert await context.payload(payload, incoming=False) == {"choices": [{"delta": {"content": "text"}}]}
    tool_result = {"messages": [{"role": "tool", "content": {"provider_specific_fields": fields}}]}
    assert await context.payload(tool_result, incoming=False) == tool_result


@pytest.mark.asyncio
@pytest.mark.parametrize("native_reason", ["end_turn", "tool_use", "max_tokens", "STOP"])
async def test_native_finish_reason_is_removed_from_actual_choices_without_changing_normalized_reason(
    native_reason: str,
) -> None:
    from litellm.types.utils import Choices

    choice = Choices(finish_reason=native_reason, message={"role": "assistant", "content": "Answer"})
    payload = {"choices": [choice.model_dump(exclude_none=True)]}
    assert payload["choices"][0]["provider_specific_fields"]["native_finish_reason"] == native_reason
    snapshot = json.dumps(payload)
    context = PublicInferenceIds(lambda: MemoryIds(), "VoidAPI")
    result = await context.payload(payload, incoming=False)
    assert result["choices"][0]["finish_reason"] == choice.finish_reason
    assert result["choices"][0]["message"] == payload["choices"][0]["message"]
    assert "provider_specific_fields" not in result["choices"][0]
    assert json.dumps(payload) == snapshot
    assert await context.payload(payload, incoming=True) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
async def test_vertex_metadata_is_removed_from_actual_response_without_altering_client_content(transport: str) -> None:
    from litellm import ModelResponse

    metadata = {
        "vertex_ai_grounding_metadata": {"groundingChunks": [{"web": {"uri": "https://example.com"}}]},
        "vertex_ai_url_context_metadata": {"urlMetadata": [{"retrievedUrl": "https://example.com"}]},
        "vertex_ai_safety_results": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}],
        "vertex_ai_citation_metadata": {"citations": [{"uri": "https://example.com"}]},
    }
    response_model = ModelResponse(
        usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "vertex_ai_grounding_metadata is client text",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com",
                                "title": "Source",
                                "start_index": 0,
                                "end_index": 6,
                            },
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call_example",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps(metadata),
                            },
                        }
                    ],
                },
            }
        ],
    )
    for name, value in metadata.items():
        setattr(response_model, name, value)
    payload = response_model.model_dump(exclude_none=True)
    payload["metadata"] = metadata
    assert all(name in payload for name in metadata)
    snapshot = json.dumps(payload)
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=MemoryIds)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    assert all(name not in result for name in metadata)
    public_tool = result["choices"][0]["message"]["tool_calls"][0]
    original_tool = payload["choices"][0]["message"]["tool_calls"][0]
    assert public_tool["id"] != original_tool["id"]
    assert public_tool == {**original_tool, "id": public_tool["id"]}
    assert result["choices"][0]["message"]["content"] == payload["choices"][0]["message"]["content"]
    assert result["choices"][0]["message"]["annotations"] == payload["choices"][0]["message"]["annotations"]
    assert result["usage"] == payload["usage"]
    assert result["metadata"] == metadata
    assert json.dumps(payload) == snapshot
    context = PublicInferenceIds(MemoryIds, "VoidAPI")
    assert await context.payload(payload, incoming=True) == payload
    tool_result = {"messages": [{"role": "tool", "content": metadata}]}
    assert await context.payload(tool_result, incoming=False) == tool_result


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_nested_code_interpreter_container_ids_are_opaque_and_owner_scoped(transport: str, wrapped: bool) -> None:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.responses.utils import ResponsesAPIRequestUtils

    native_container = "cntr_native-example"
    container = (
        ResponsesAPIRequestUtils._build_container_id("anthropic", "private-deployment", native_container)
        if wrapped
        else native_container
    )
    results = AnthropicConfig()._build_code_interpreter_results(
        tool_results=[
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "toolu_example",
                "content": {"stdout": "container_id is user-visible output", "stderr": ""},
            }
        ],
        code_by_id={"toolu_example": "print('container_id is user-visible output')"},
        container_id=container,
    )
    payload = {
        "object": "response",
        "output": [item.model_dump(exclude_none=True) for item in results],
        "container": {"id": container},
    }
    snapshot = json.dumps(payload)
    store = MemoryIds()
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    public = result["output"][0]["container_id"]
    assert public.startswith("cntr_") and public != container
    assert result["container"]["id"] == public
    original_result = payload["output"][0]
    assert result["output"][0] == {**original_result, "id": result["output"][0]["id"], "container_id": public}
    assert container not in json.dumps(result)
    assert json.dumps(payload) == snapshot
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    assert await continued.payload({"tools": [{"type": "code_interpreter", "container": public}]}, incoming=True) == {
        "tools": [{"type": "code_interpreter", "container": container}]
    }
    other = PublicInferenceIds(lambda: store, "VoidAPI")
    other.bind(authenticated(user_id="bob"))
    with pytest.raises(HTTPException) as denied:
        await other.identifier("container", public, incoming=True)
    assert denied.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
@pytest.mark.parametrize("node", ["message", "delta"])
@pytest.mark.parametrize("signed", [False, True])
async def test_tool_calls_filter_internal_metadata_and_roundtrip_ids_arguments_and_signatures(
    transport: str,
    node: str,
    signed: bool,
) -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _encode_tool_call_id_with_signature,
        _get_thought_signature_from_tool,
    )

    internal = {
        "api_base": "https://private.example",
        "api_key": "internal-placeholder",
        "model_id": "private-account",
        "litellm_trace_id": "private-trace",
    }
    signature_fields = {"thought_signature": "native-signature"} if signed else {}
    call_id = _encode_tool_call_id_with_signature("call_example", "native-signature" if signed else None)
    arguments = json.dumps(
        {"api_key": "client-input", "model_id": "client-input", "provider_specific_fields": internal}
    )
    call = {
        "id": call_id,
        "type": "function",
        "index": 0,
        "provider_specific_fields": {**internal, **signature_fields},
        "function": {
            "name": "lookup",
            "arguments": arguments,
            "provider_specific_fields": {**internal, **signature_fields},
        },
    }
    payload = {"choices": [{node: {"role": "assistant", "tool_calls": [call]}}], "metadata": internal}
    expected_call = {key: value for key, value in call.items() if key != "provider_specific_fields"}
    expected_call["function"] = {"name": "lookup", "arguments": arguments}
    snapshot = json.dumps(payload)
    store = MemoryIds()
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    actual_call = result["choices"][0][node]["tool_calls"][0]
    assert actual_call["id"].startswith("call_") and "__thought__" not in actual_call["id"]
    assert actual_call == {**expected_call, "id": actual_call["id"]}
    assert _get_thought_signature_from_tool(actual_call) is None
    assert result["metadata"] == internal
    assert json.dumps(payload) == snapshot
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    restored = await context.payload(result, incoming=True)
    restored_call = restored["choices"][0][node]["tool_calls"][0]
    assert restored_call == expected_call
    assert _get_thought_signature_from_tool(restored_call) == _get_thought_signature_from_tool(call)
    tool_output = {"messages": [{"role": "tool", "content": {"tool_calls": [call]}}]}
    assert await context.payload(tool_output, incoming=False) == tool_output


@pytest.mark.asyncio
async def test_streamed_tool_id_upgrade_preserves_signature_for_chat_and_responses_continuation() -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _encode_tool_call_id_with_signature,
        _get_thought_signature_from_tool,
    )

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    context.model = "public-model"
    raw = "call_native-example"
    signed = _encode_tool_call_id_with_signature(raw, "native-signature")
    added = await context.payload({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": raw}]}}]}, incoming=False)
    public = added["choices"][0]["delta"]["tool_calls"][0]["id"]
    assert public.startswith("call_") and public != raw
    assert await context.identifier("tool", public, incoming=True) == raw
    done = await context.payload(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": signed, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        },
        incoming=False,
    )
    assert done["choices"][0]["message"]["tool_calls"][0]["id"] == public
    assert await context.identifier("tool", public, incoming=True) == signed
    later = await context.payload({"call_id": raw}, incoming=False)
    assert later["call_id"] == public
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    messages = {
        "messages": [
            {"role": "assistant", "tool_calls": done["choices"][0]["message"]["tool_calls"]},
            {"role": "tool", "tool_call_id": public, "content": "Result"},
        ]
    }
    restored = await continued.payload(messages, incoming=True)
    assistant_call = restored["messages"][0]["tool_calls"][0]
    assert assistant_call["id"] == signed
    assert _get_thought_signature_from_tool(assistant_call) == "native-signature"
    assert restored["messages"][1] == {"role": "tool", "tool_call_id": signed, "content": "Result"}
    response_input = {
        "input": [
            {"type": "function_call", "call_id": public, "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": public, "output": {"call_id": "client-owned"}},
        ]
    }
    response_restored = await continued.payload(response_input, incoming=True)
    assert response_restored["input"][0]["call_id"] == signed
    assert response_restored["input"][1] == {
        "type": "function_call_output",
        "call_id": signed,
        "output": {"call_id": "client-owned"},
    }
    other = PublicInferenceIds(lambda: store, "VoidAPI")
    other.bind(authenticated(user_id="bob"))
    with pytest.raises(HTTPException) as denied:
        await other.payload(messages, incoming=True)
    assert denied.value.status_code == 404
    with pytest.raises(HTTPException) as legacy:
        await continued.identifier("tool", signed, incoming=True)
    assert legacy.value.status_code == 400


@pytest.mark.asyncio
async def test_reused_native_tool_ids_in_different_responses_do_not_overwrite_signatures() -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import _encode_tool_call_id_with_signature

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    ids: list[str] = []
    originals: list[str] = []
    for response_id, signature in (("chatcmpl_one", "first-signature"), ("chatcmpl_two", "second-signature")):
        original = _encode_tool_call_id_with_signature("call_reused", signature)
        originals.append(original)
        result = await context.payload(
            {"id": response_id, "choices": [{"message": {"tool_calls": [{"id": original}]}}]}, incoming=False
        )
        ids.append(result["choices"][0]["message"]["tool_calls"][0]["id"])
    assert ids[0] != ids[1]
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    for public, original in zip(ids, originals, strict=True):
        assert await continued.identifier("tool", public, incoming=True) == original
        assert await continued.identifier("tool", original, incoming=False) == public


@pytest.mark.asyncio
@pytest.mark.parametrize("embedded", [None, "same-signature", "different-signature"])
@pytest.mark.parametrize("location", ["tool", "function"])
async def test_only_signatures_recoverable_from_the_stored_id_are_removed(embedded: str | None, location: str) -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _encode_tool_call_id_with_signature,
        _get_thought_signature_from_tool,
    )

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    original_id = _encode_tool_call_id_with_signature("call_example", embedded)
    signature_fields = {"thought_signature": "same-signature", "enabled": False}
    call = {"id": original_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    if location == "tool":
        call["provider_specific_fields"] = signature_fields
    else:
        call["function"]["provider_specific_fields"] = signature_fields
    payload = {
        "choices": [{"message": {"tool_calls": [call]}}],
        "metadata": {"provider_specific_fields": signature_fields},
    }
    snapshot = json.dumps(payload)
    public = await context.payload(payload, incoming=False)
    public_call = public["choices"][0]["message"]["tool_calls"][0]
    assert "provider_specific_fields" not in public_call
    assert "provider_specific_fields" not in public_call["function"]
    assert public["metadata"] == payload["metadata"]
    assert json.dumps(payload) == snapshot
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    restored = await continued.payload(
        {"messages": [{"role": "assistant", "tool_calls": [public_call]}]}, incoming=True
    )
    restored_call = restored["messages"][0]["tool_calls"][0]
    assert restored_call["id"] == _encode_tool_call_id_with_signature("call_example", "same-signature")
    assert _get_thought_signature_from_tool(restored_call) == _get_thought_signature_from_tool(call)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
@pytest.mark.parametrize("keep_results", [False, True])
async def test_nonstandard_search_call_summary_is_removed_but_search_results_are_preserved(
    transport: str, keep_results: bool
) -> None:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    search_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_example",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "Source",
                    "encrypted_content": "native-ciphertext",
                }
            ],
        }
    ]
    calls = AnthropicConfig()._build_web_search_calls(
        search_results,
        {
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_example",
                    "name": "web_search",
                    "input": {"query": "example"},
                }
            ]
        },
    )
    summaries = [call.model_dump(exclude_none=True) for call in calls]
    assert summaries
    fields = {"web_search_calls": summaries}
    if keep_results:
        fields.update(web_search_results=search_results, citations=[{"url": "https://example.com"}])
    payload = {
        "choices": [
            {"message": {"content": "web_search_calls is user-visible text", "provider_specific_fields": fields}}
        ],
        "metadata": {"web_search_calls": summaries},
    }
    snapshot = json.dumps(payload)
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=MemoryIds)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    expected = {"content": "web_search_calls is user-visible text"}
    assert result["choices"][0]["message"] == expected
    assert result["metadata"] == payload["metadata"]
    assert json.dumps(payload) == snapshot
    context = PublicInferenceIds(MemoryIds, "VoidAPI")
    assert await context.payload(payload, incoming=True) == payload
    tool_output = {"messages": [{"role": "tool", "content": {"provider_specific_fields": fields}}]}
    assert await context.payload(tool_output, incoming=False) == tool_output


@pytest.mark.asyncio
async def test_standard_responses_web_search_call_is_preserved() -> None:
    from litellm.types.responses.main import build_web_search_call

    call = build_web_search_call(
        "example", {"query": "example"}, {"content": [{"type": "web_search_result", "url": "https://example.com"}]}
    ).model_dump(exclude_none=True)
    context = PublicInferenceIds(MemoryIds, "VoidAPI")
    context.bind(authenticated())
    payload = {"object": "response", "output": [call]}
    result = await context.payload(payload, incoming=False)
    translated = result["output"][0]
    assert translated == {**call, "id": translated["id"]}
    assert translated["id"].startswith("item_")


@pytest.mark.asyncio
@pytest.mark.parametrize("signature_first", [False, True])
async def test_streamed_signature_only_delta_is_saved_without_emitting_provider_fields(signature_first: bool) -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import _get_thought_signature_from_tool

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    id_delta = {"index": 0, "id": "call_example", "type": "function", "function": {"name": "lookup", "arguments": ""}}
    signature_delta = {"index": 0, "function": {"provider_specific_fields": {"thought_signature": "native-signature"}}}
    results = [
        await context.payload({"choices": [{"delta": {"tool_calls": [delta]}}]}, incoming=False)
        for delta in ([signature_delta, id_delta] if signature_first else [id_delta, signature_delta])
    ]
    assert all("provider_specific_fields" not in json.dumps(result) for result in results)
    public_call = results[1 if signature_first else 0]["choices"][0]["delta"]["tool_calls"][0]
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    restored = await continued.payload(
        {"messages": [{"role": "assistant", "tool_calls": [public_call]}]}, incoming=True
    )
    call = restored["messages"][0]["tool_calls"][0]
    assert _get_thought_signature_from_tool(call) == "native-signature"
    assert call["id"].split("__thought__", 1)[0] == "call_example"
    content = {"role": "tool", "tool_call_id": public_call["id"], "content": "Result"}
    reply = await continued.payload({"messages": [content]}, incoming=True)
    assert reply["messages"][0]["tool_call_id"] == call["id"]


@pytest.mark.asyncio
async def test_reasoning_in_provider_fields_is_promoted_before_wrapper_removal() -> None:
    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    thinking = [{"type": "thinking", "thinking": "Reasoning", "signature": "native-signature"}]
    payload = {
        "choices": [{"message": {"content": "Answer", "provider_specific_fields": {"thinking_blocks": thinking}}}]
    }
    result = await context.payload(payload, incoming=False)
    assert result == {"choices": [{"message": {"content": "Answer", "thinking_blocks": thinking}}]}
    existing = {
        "choices": [
            {"message": {"thinking_blocks": thinking, "provider_specific_fields": {"thinking_blocks": ["duplicate"]}}}
        ]
    }
    assert (await context.payload(existing, incoming=False))["choices"][0]["message"]["thinking_blocks"] == thinking


@pytest.mark.asyncio
async def test_native_messages_keep_signature_extension_needed_by_the_adapter() -> None:
    store = MemoryIds()
    payload = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_example",
                "name": "lookup",
                "input": {},
                "provider_specific_fields": {"signature": "native-signature"},
            }
        ],
    }
    messages, _ = await http_exchange(lambda: store, authenticated(), {}, payload, path="/v1/messages")
    assert response(messages) == payload


@pytest.mark.asyncio
async def test_response_function_call_signature_is_stored_using_call_id_not_output_item_id() -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import _get_thought_signature_from_tool

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    payload = {
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_item",
                "call_id": "call_native",
                "name": "lookup",
                "arguments": "{}",
                "provider_specific_fields": {"thought_signature": "native-signature"},
            }
        ],
    }
    result = await context.payload(payload, incoming=False)
    item = result["output"][0]
    assert "provider_specific_fields" not in item
    assert item["id"].startswith("item_") and item["call_id"].startswith("call_")
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    restored = await continued.payload({"input": [item]}, incoming=True)
    call = restored["input"][0]
    assert call["id"] == "fc_item"
    assert _get_thought_signature_from_tool({"id": call["call_id"]}) == "native-signature"
    assert call["name"] == "lookup" and call["arguments"] == "{}"


@pytest.mark.asyncio
async def test_interleaved_choice_tool_signatures_do_not_overwrite_each_other() -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import _get_thought_signature_from_tool

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    first = await context.payload(
        {
            "id": "chatcmpl_example",
            "choices": [
                {"index": index, "delta": {"tool_calls": [{"index": 0, "id": "call_reused"}]}} for index in (0, 1)
            ],
        },
        incoming=False,
    )
    ids = [choice["delta"]["tool_calls"][0]["id"] for choice in first["choices"]]
    assert ids[0] != ids[1]
    for index in (1, 0):
        frame = await context.payload(
            {
                "id": "chatcmpl_example",
                "choices": [
                    {
                        "index": index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "provider_specific_fields": {"thought_signature": f"signature-{index}"}
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            incoming=False,
        )
        assert "provider_specific_fields" not in json.dumps(frame)
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    for index, public in enumerate(ids):
        internal = await continued.identifier("tool", public, incoming=True)
        assert _get_thought_signature_from_tool({"id": internal}) == f"signature-{index}"


@pytest.mark.asyncio
async def test_unmigrated_message_and_legacy_function_signatures_are_not_lost() -> None:
    context = PublicInferenceIds(MemoryIds, "VoidAPI")
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Answer",
                    "provider_specific_fields": {"thought_signatures": ["native-signature"], "unused": "drop"},
                    "function_call": {
                        "name": "lookup",
                        "arguments": "{}",
                        "provider_specific_fields": {"thought_signature": "legacy-signature"},
                    },
                }
            }
        ]
    }
    result = await context.payload(payload, incoming=False)
    message = result["choices"][0]["message"]
    assert message["provider_specific_fields"] == {"thought_signatures": ["native-signature"]}
    assert message["function_call"] == payload["choices"][0]["message"]["function_call"]


@pytest.mark.asyncio
@pytest.mark.parametrize("signature_first", [False, True])
async def test_reused_tool_ids_at_different_indices_keep_separate_streamed_signatures(signature_first: bool) -> None:
    from litellm.litellm_core_utils.prompt_templates.factory import _get_thought_signature_from_tool

    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())

    async def chunk(calls):
        return await context.payload(
            {"id": "chatcmpl_same", "choices": [{"index": 0, "delta": {"tool_calls": calls}}]}, incoming=False
        )

    async def signatures():
        for index in (1, 0):
            await chunk(
                [
                    {
                        "index": index,
                        "function": {"provider_specific_fields": {"thought_signature": f"signature-{index}"}},
                    }
                ]
            )

    if signature_first:
        await signatures()
    initial = await chunk(
        [
            {"index": index, "id": "call_reused", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            for index in (0, 1)
        ]
    )
    calls = initial["choices"][0]["delta"]["tool_calls"]
    ids = [call["id"] for call in calls]
    assert ids[0] != ids[1]
    if not signature_first:
        await signatures()
    for index in (1, 0):
        repeated = await chunk([{"index": index, "id": "call_reused"}])
        assert repeated["choices"][0]["delta"]["tool_calls"][0]["id"] == ids[index]
    final = await context.payload(
        {
            "id": "chatcmpl_same",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_reused",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                                "provider_specific_fields": {"thought_signature": f"signature-{index}"},
                            }
                            for index in (0, 1)
                        ]
                    },
                }
            ],
        },
        incoming=False,
    )
    final_calls = final["choices"][0]["message"]["tool_calls"]
    assert [call["id"] for call in final_calls] == ids
    continued = PublicInferenceIds(lambda: store, "VoidAPI")
    continued.bind(authenticated())
    restored = await continued.payload(
        {
            "messages": [
                {"role": "assistant", "tool_calls": final_calls},
                *(
                    {"role": "tool", "tool_call_id": public, "content": f"result-{index}"}
                    for index, public in enumerate(ids)
                ),
            ]
        },
        incoming=True,
    )
    for index, call in enumerate(restored["messages"][0]["tool_calls"]):
        assert _get_thought_signature_from_tool(call) == f"signature-{index}"
        assert restored["messages"][index + 1]["tool_call_id"] == call["id"]
        assert restored["messages"][index + 1]["content"] == f"result-{index}"
    assert context.tool_index == 0 and continued.tool_index == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_json", [False, True])
async def test_id_restoration_synchronizes_all_body_readers_without_adding_auth_fields(cached_json: bool) -> None:
    from litellm.proxy.common_utils.http_parsing_utils import (
        _read_request_body,
        _safe_get_request_parsed_body,
        read_raw_json_body,
    )

    store = MemoryIds()
    issuer = PublicInferenceIds(lambda: store, "VoidAPI")
    issuer.bind(authenticated())
    internal = wrapped_response()
    public = await issuer.identifier("response", internal, incoming=False)
    body = {"previous_response_id": public, "input": "Continue 🐈", "metadata": {"previous_response_id": public}}
    raw = json.dumps(body).encode()

    async def receive() -> Message:
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "state": {},
        },
        receive,
    )
    await _read_request_body(request)
    original_json = await request.json() if cached_json else None
    data = await _read_request_body(request)
    data["litellm_metadata"] = {"auth_only": True}
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    await context.authorize_request(request, authenticated(), data)
    expected = {**body, "previous_response_id": internal}
    assert data == {**expected, "litellm_metadata": {"auth_only": True}}
    assert _safe_get_request_parsed_body(request) == expected
    assert await _read_request_body(request) == expected
    assert await request.json() == expected
    assert json.loads(await request.body()) == expected
    assert json.loads(await read_raw_json_body(request)) == expected
    if original_json is not None:
        assert original_json == expected


@pytest.mark.asyncio
async def test_id_restoration_preserves_multipart_upload_body_and_file_objects() -> None:
    from io import BytesIO

    from starlette.datastructures import UploadFile

    from litellm.proxy.common_utils.http_parsing_utils import (
        _safe_get_request_parsed_body,
        _safe_set_request_parsed_body,
    )

    store = MemoryIds()
    issuer = PublicInferenceIds(lambda: store, "VoidAPI")
    issuer.bind(authenticated())
    public = await issuer.identifier("file", "file-native", incoming=False)
    upload = UploadFile(BytesIO(b"file content"), filename="example.txt")
    raw = b"--boundary\r\nOpaque multipart bytes\r\n--boundary--\r\n"
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/files",
            "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")],
            "query_string": b"",
            "state": {},
        }
    )
    request._body = raw
    _safe_set_request_parsed_body(request, {"file_id": public, "file": upload})
    data = _safe_get_request_parsed_body(request)
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    await context.authorize_request(request, authenticated(), data)
    assert data["file_id"] == "file-native"
    assert _safe_get_request_parsed_body(request)["file_id"] == "file-native"
    assert _safe_get_request_parsed_body(request)["file"] is upload
    assert await request.body() == raw
    assert not hasattr(request, "_json")
    await upload.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("block_type", ["image", "document"])
@pytest.mark.parametrize("placement", ["message", "tool_result", "document_content"])
async def test_anthropic_file_sources_restore_with_owner_checks(block_type: str, placement: str) -> None:
    store = MemoryIds()
    issuer = PublicInferenceIds(lambda: store, "VoidAPI")
    issuer.bind(authenticated())
    native = "file-native-example"
    public = await issuer.identifier("file", native, incoming=False)
    block = {"type": block_type, "source": {"type": "file", "file_id": public}}
    expected_block = {"type": block_type, "source": {"type": "file", "file_id": native}}

    def placed(item):
        if placement == "tool_result":
            return {"type": "tool_result", "tool_use_id": "tool_native", "content": [item]}
        if placement == "document_content":
            return {"type": "document", "source": {"type": "content", "content": [item]}}
        return item

    body = {"messages": [{"role": "user", "content": [placed(block)]}], "metadata": {"source": block["source"]}}
    sent, seen = await http_exchange(
        lambda: store, authenticated(), body, {"type": "message", "content": []}, path="/v1/messages"
    )
    assert sent[0]["status"] == 200
    expected = {"messages": [{"role": "user", "content": [placed(expected_block)]}], "metadata": body["metadata"]}
    assert seen["body"] == expected
    assert seen["cached"] == expected
    denied, _ = await http_exchange(lambda: store, authenticated(user_id="bob"), body, path="/v1/messages")
    assert denied[0]["status"] == 404
    unknown = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": block_type, "source": {"type": "file", "file_id": "file_" + "f" * 32}}],
            }
        ]
    }
    missing, _ = await http_exchange(lambda: store, authenticated(), unknown, path="/v1/messages")
    assert missing[0]["status"] == 404


@pytest.mark.asyncio
async def test_anthropic_source_restoration_does_not_rewrite_user_data() -> None:
    store = MemoryIds()
    context = PublicInferenceIds(lambda: store, "VoidAPI")
    context.bind(authenticated())
    public = await context.identifier("file", "file-native", incoming=False)
    file_source = {"type": "image", "source": {"type": "file", "file_id": public}}
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": f"https://example.com/{public}"}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": public}},
                    {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": public}},
                    {"type": "text", "text": json.dumps(file_source)},
                    {"type": "tool_result", "tool_use_id": "tool_native", "content": json.dumps(file_source)},
                    {"type": "tool_result", "tool_use_id": "tool_native", "content": {"data": file_source}},
                    {"type": "tool_use", "id": "tool_native", "name": "lookup", "input": file_source},
                ],
            }
        ],
        "metadata": file_source,
    }
    assert await context.payload(payload, incoming=True) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "sse", "websocket"])
async def test_gemini_server_tool_history_is_not_exposed_by_public_boundary(transport: str) -> None:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    parts = [
        {
            "toolCall": {"toolType": "GOOGLE_SEARCH", "id": "search-1", "args": {"query": "example"}},
            "thoughtSignature": "call-signature",
        },
        {
            "toolResponse": {
                "toolType": "GOOGLE_SEARCH",
                "id": "search-1",
                "response": {"result": "Found example", "model_id": "user-owned-result"},
            },
            "thoughtSignature": "result-signature",
        },
    ]
    invocations = VertexGeminiConfig._extract_server_side_tool_invocations(parts)
    fields = {"server_side_tool_invocations": invocations}
    message = {
        "role": "assistant",
        "content": "Answer",
        "provider_specific_fields": {**fields, "model_id": "private-route", "unneeded_diagnostic": "discard"},
    }
    payload = {"choices": [{("delta" if transport == "sse" else "message"): message}]}
    snapshot = json.dumps(payload)
    store = MemoryIds()
    sent: list[Message] = []

    async def app(scope: Scope, receive, send) -> None:
        scope["state"][STATE_KEY].bind(authenticated())
        if transport == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps(payload)})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream" if transport == "sse" else b"application/json")],
            }
        )
        body = json.dumps(payload).encode()
        await send({"type": "http.response.body", "body": b"data: " + body + b"\n\n" if transport == "sse" else body})

    async def send(value: Message) -> None:
        sent.append(value)

    await PublicInferenceBoundary(app, id_store_factory=lambda: store)(
        {
            "type": "websocket" if transport == "websocket" else "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "state": {},
        },
        lambda: None,
        send,
    )
    if transport == "websocket":
        result = json.loads(next(value["text"] for value in sent if value["type"] == "websocket.send"))
    else:
        body = b"".join(value.get("body", b"") for value in sent)
        result = json.loads(body.removeprefix(b"data: ").strip())
    public_message = result["choices"][0]["delta" if transport == "sse" else "message"]
    assert public_message == {"role": "assistant", "content": "Answer"}
    assert json.dumps(payload) == snapshot
    for private_value in (
        "server_side_tool_invocations",
        "GOOGLE_SEARCH",
        "search-1",
        "call-signature",
        "result-signature",
        "private-route",
        "unneeded_diagnostic",
    ):
        assert private_value not in json.dumps(result)
