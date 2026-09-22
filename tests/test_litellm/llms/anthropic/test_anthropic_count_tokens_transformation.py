import copy

from litellm.llms.anthropic.common_utils import ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)


def test_transform_basic_request():
    """Test basic request with only model and messages."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result == {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_transform_includes_system():
    """Test that system prompt is included when provided."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
    )

    assert result["system"] == "You are a helpful assistant."
    assert result["model"] == "claude-3-5-sonnet"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


def test_transform_includes_tools():
    """Test that tools are included when provided."""
    config = AnthropicCountTokensConfig()

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
    )

    assert result["tools"] == tools


def test_transform_includes_system_and_tools():
    """Test that both system and tools are included together."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="Be helpful",
        tools=[{"name": "my_tool", "input_schema": {"type": "object"}}],
    )

    assert "system" in result
    assert "tools" in result
    assert "messages" in result
    assert "model" in result


def test_transform_no_system_no_tools():
    """Test that None system/tools are not included."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system=None,
        tools=None,
    )

    assert "system" not in result
    assert "tools" not in result


def test_subscription_transform_canonicalizes_identity_and_tools_without_mutating_request():
    config = AnthropicCountTokensConfig()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "read", "input": {}}],
        }
    ]
    tools = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object"}}]
    original_messages = copy.deepcopy(messages)
    original_tools = copy.deepcopy(tools)

    result = config.transform_request_to_count_tokens(
        model="claude-test",
        messages=messages,
        tools=tools,
        system="Keep the user context",
        subscription_request=True,
    )

    assert result["system"] == [{"type": "text", "text": ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT}]
    assert result["messages"][0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "<system-reminder>"},
            {"type": "text", "text": "Keep the user context"},
            {"type": "text", "text": "</system-reminder>"},
        ],
    }
    assert result["tools"][0]["name"] == "Read"
    assert result["messages"][1]["content"][0]["name"] == "Read"
    assert messages == original_messages
    assert tools == original_tools
