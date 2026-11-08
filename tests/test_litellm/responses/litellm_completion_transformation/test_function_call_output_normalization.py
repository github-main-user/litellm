"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


@pytest.mark.parametrize("output_type", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (None, ""),
        ("", ""),
        ([], ""),
        ([{"type": "input_text", "text": ""}], ""),
        ([{"type": "output_text", "text": ""}], ""),
        ([{"type": "text", "text": ""}], ""),
        ([{"type": kind, "text": ""} for kind in ("input_text", "output_text", "text")], ""),
        ("[]", "[]"),
        ([{"type": "input_text", "text": " \n"}], " \n"),
        ([{"type": "input_text", "text": 17}], '[{"type": "input_text", "text": 17}]'),
        ([{"type": "input_text"}], '[{"type": "input_text"}]'),
        ([{"type": "input_text", "text": None}], '[{"type": "input_text", "text": null}]'),
        (
            [{"type": "input_text", "text": ""}, {"type": "unknown", "value": 17}],
            '[{"type": "input_text", "text": ""}, {"type": "unknown", "value": 17}]',
        ),
        ([{"type": "input_text", "text": ""}, 17], '[{"type": "input_text", "text": ""}, 17]'),
    ],
)
def test_empty_tool_outputs_preserve_call_id_without_inventing_json_text(
    output_type: str, output: object, expected: str
) -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {"type": output_type, "call_id": "call_empty", "output": output}
        )
    )
    assert messages == [{"role": "tool", "tool_call_id": "call_empty", "content": expected}]


def test_empty_text_alongside_image_preserves_media_and_tool_correlation() -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": ""},
                    {"type": "input_text", "text": "Screenshot"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        )
    )
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_image",
            "content": [
                {"type": "text", "text": "Screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
