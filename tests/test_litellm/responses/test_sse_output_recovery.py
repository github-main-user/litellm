"""Tests for litellm.responses.sse_output_recovery helpers."""

from litellm.responses.sse_output_recovery import (
    _MAX_CONTENT_INDEX,
    record_output_text_chunk,
)


def test_text_chunk_with_oversized_content_index_is_dropped():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": _MAX_CONTENT_INDEX + 1,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    item = text_only_items[0]
    assert item["content"] == []


def test_text_chunk_with_negative_content_index_is_dropped():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": -1,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    assert text_only_items[0]["content"] == []


def test_text_chunk_at_max_content_index_is_recorded():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": _MAX_CONTENT_INDEX,
            "text": "kept",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    content = text_only_items[0]["content"]
    assert len(content) == _MAX_CONTENT_INDEX + 1
    assert content[_MAX_CONTENT_INDEX]["text"] == "kept"


def test_delta_recovery_merges_indices_and_prefers_done_events():
    from litellm.responses.sse_output_recovery import (
        merge_recovered_output_items,
        record_recovery_text_chunk,
    )

    chunks = {}
    for output_index, content_index, text in [(1, 1, "Hel"), (0, 0, "ignored"), (1, 0, "First"), (1, 1, "lo")]:
        record_recovery_text_chunk(
            {
                "type": "response.output_text.delta",
                "output_index": output_index,
                "content_index": content_index,
                "delta": text,
            },
            chunks,
        )
    record_recovery_text_chunk(
        {"type": "response.output_text.done", "output_index": 1, "content_index": 0, "text": "Authoritative"},
        chunks,
    )
    record_recovery_text_chunk(
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "ignored"},
        chunks,
    )
    item = {"type": "message", "content": [{"type": "output_text", "text": "Full item"}]}
    recovered = merge_recovered_output_items({0: item}, chunks, completed=True)
    assert recovered == {
        0: item,
        1: {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Authoritative", "annotations": []},
                {"type": "output_text", "text": "Hello", "annotations": []},
            ],
        },
    }
    assert merge_recovered_output_items({}, chunks, completed=False) == {
        1: {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Authoritative", "annotations": []},
            ],
        },
    }
