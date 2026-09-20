import json
from collections.abc import AsyncIterator
from copy import deepcopy

import pytest
from openai._streaming import SSEDecoder

from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.subscription_tools import (
    prepare_subscription_tools,
    restore_subscription_tool_stream,
    restore_subscription_tools,
)


def test_subscription_tool_round_trip_preserves_payloads_and_custom_names():
    body = {
        "tools": [
            {"name": "read", "input_schema": {"type": "object"}},
            {"name": "TodoWRITE", "input_schema": {"type": "object"}},
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"name": "bash", "type": "bash_20250124"},
        ],
        "tool_choice": {"type": "tool", "name": "read", "disable_parallel_tool_use": True},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_read", "name": "read", "input": {"name": "read"}},
                    {"type": "tool_use", "id": "toolu_bash", "name": "bash", "input": {"command": "pwd"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_read", "content": "read stays in text"},
                    {"type": "image", "source": {"type": "base64", "data": "read"}},
                ],
            },
        ],
    }
    original = deepcopy(body)
    wire, reverse = prepare_subscription_tools(body)

    assert [tool["name"] for tool in wire["tools"]] == ["Read", "TodoWrite", "get_weather", "bash"]
    assert wire["tool_choice"] == {**body["tool_choice"], "name": "Read"}
    assert reverse == {"Read": "read", "TodoWrite": "TodoWRITE"}
    assert wire["messages"][0]["content"][0] == {**body["messages"][0]["content"][0], "name": "Read"}
    assert wire["messages"][0]["content"][1] == body["messages"][0]["content"][1]
    assert wire["messages"][1] == body["messages"][1]
    response = {"content": wire["messages"][0]["content"]}
    assert restore_subscription_tools(response, reverse) == {"content": body["messages"][0]["content"]}
    assert body == original
    assert response["content"][0]["name"] == "Read"


@pytest.mark.parametrize("names", [("read", "Read"), ("Read", "read"), ("READ", "read")])
def test_subscription_rejects_ambiguous_tool_names(names):
    with pytest.raises(AnthropicError, match="collide after normalization") as error:
        prepare_subscription_tools({"tools": [{"name": name} for name in names], "messages": []})
    assert error.value.status_code == 400


def test_subscription_rejects_collisions_with_history_and_reserved_tools():
    with pytest.raises(AnthropicError, match="collide after normalization"):
        prepare_subscription_tools(
            {
                "tools": [{"name": "read"}],
                "messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": "Read"}]}],
            }
        )
    with pytest.raises(AnthropicError, match="collide after normalization"):
        prepare_subscription_tools(
            {
                "tools": [{"name": "read"}, {"name": "Read", "type": "vendor_tool"}],
                "messages": [],
            }
        )


def test_subscription_rewrites_history_only_tools_and_tool_references():
    body = {
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "name": "read", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "tool_reference", "tool_name": "read"},
                        ],
                    }
                ],
            },
            {
                "role": "system",
                "content": [{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "read"}}],
            },
        ],
    }
    wire, reverse = prepare_subscription_tools(body)
    assert wire["messages"][0]["content"][0]["name"] == "Read"
    assert wire["messages"][1]["content"][0]["content"][0]["tool_name"] == "Read"
    assert wire["messages"][2]["content"][0]["tool"]["name"] == "Read"
    assert reverse == {"Read": "read"}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
async def test_subscription_stream_restores_names_across_byte_boundaries(chunk_size):
    events = [
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"name":"Read","text":"café"}'},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_custom", "name": "get_weather", "input": {}},
        },
        {"type": "error", "error": {"type": "overloaded_error", "message": "Try again"}},
        {"type": "message_stop"},
    ]
    raw = b"".join(
        f"event: {event['type']}\r\ndata: {json.dumps(event, ensure_ascii=False)}\r\n\r\n".encode() for event in events
    )

    async def chunks() -> AsyncIterator[bytes]:
        for offset in range(0, len(raw), chunk_size):
            yield raw[offset : offset + chunk_size]

    output = [chunk async for chunk in restore_subscription_tool_stream(chunks(), {"Read": "read"})]
    parsed = [event.json() for event in SSEDecoder().iter_bytes(iter(output))]
    expected = deepcopy(events)
    expected[1]["content_block"]["name"] = "read"
    assert parsed == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 10000])
async def test_subscription_stream_preserves_native_sse_bytes(chunk_size):
    unchanged = (
        b": heartbeat\r\n\r\n"
        b"event: ping\nid: keep\nx-extra: yes\n"
        b"data: {\ndata: \"type\": \"ping\",\ndata: \"text\": \"caf\xc3\xa9\"\ndata: }\n\n"
    )
    changed_payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {}},
        "text": "café",
    }
    changed = (
        b": event comment\r\n"
        b"event: content_block_start\r\n"
        b"id: 7\r\n"
        b"retry: 12\r\n"
        b"x-extra: untouched\r\n"
        + b"data: "
        + json.dumps(changed_payload, ensure_ascii=False).encode()
        + b"\r\n\r\n"
    )
    unterminated_tail = b"event: pending\ndata: not-finished"
    raw = unchanged + changed + unterminated_tail

    async def chunks() -> AsyncIterator[bytes]:
        for offset in range(0, len(raw), chunk_size):
            yield raw[offset : offset + chunk_size]

    restored_payload = deepcopy(changed_payload)
    restored_payload["content_block"]["name"] = "read"
    expected_changed = (
        b": event comment\r\n"
        b"event: content_block_start\r\n"
        b"id: 7\r\n"
        b"retry: 12\r\n"
        b"x-extra: untouched\r\n"
        + b"data: "
        + json.dumps(restored_payload).encode()
        + b"\r\n\r\n"
    )
    output = b"".join([chunk async for chunk in restore_subscription_tool_stream(chunks(), {"Read": "read"})])

    assert output == unchanged + expected_changed + unterminated_tail


@pytest.mark.asyncio
async def test_subscription_stream_without_name_mapping_preserves_transport_chunks():
    chunks = (b": hea", b"rtbeat\n", b"\n")

    async def upstream() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    assert [chunk async for chunk in restore_subscription_tool_stream(upstream(), {})] == list(chunks)


@pytest.mark.asyncio
async def test_subscription_stream_closes_upstream_when_consumer_stops():
    closed = []

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        finally:
            closed.append(True)

    stream = restore_subscription_tool_stream(chunks(), {"Read": "read"})
    await anext(stream)
    await stream.aclose()
    assert closed == [True]


@pytest.mark.asyncio
async def test_subscription_stream_closes_upstream_after_error():
    closed = []

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield b': heartbeat\n\n'
            raise RuntimeError("upstream failed")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="upstream failed"):
        _ = [chunk async for chunk in restore_subscription_tool_stream(chunks(), {"Read": "read"})]
    assert closed == [True]
