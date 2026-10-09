import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.llms.anthropic.subscription_billing import (
    anthropic_subscription_cch,
    serialize_anthropic_subscription_request,
)

_VECTORS: Final = json.loads(
    (Path(__file__).parent / "fixtures" / "subscription_cch_2_1_283.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", _VECTORS, ids=[case["name"] for case in _VECTORS])
def test_subscription_cch_matches_native_machine_code_vectors(case: dict[str, str]) -> None:
    assert anthropic_subscription_cch(case["body"].encode()) == case["cch"]


@pytest.mark.timeout(3)
def test_subscription_cch_handles_large_nested_payload_without_quadratic_scans() -> None:
    import xxhash

    base: Final = json.loads(_VECTORS[0]["body"])
    body: Final = {
        "model": base["model"], "system": base["system"],
        "tools": [{"input_schema": {"examples": [
            {"max_tokens": 1, "description": "x" * 100} for _ in range(16000)
        ]}}],
        "max_tokens": 16,
    }
    selected: Final = json.dumps({
        "model": "", "system": base["system"],
        "tools": [{"input_schema": {"examples": [{"description": "x" * 100} for _ in range(16000)]}}],
    }, separators=(",", ":")).encode()
    expected: Final = xxhash.xxh64(selected, seed=0x4D659218E32A3268).intdigest() & 0xFFFFF
    assert anthropic_subscription_cch(json.dumps(body, separators=(",", ":")).encode()) == f"{expected:05x}"


@pytest.mark.parametrize("cch_fields", ["", " cch=00000;", " cch=12345;", " cch=stale; cch=abcde;"])
def test_serializer_refreshes_only_billing_cch_and_preserves_input(cch_fields: str) -> None:
    text: Final = (
        "x-anthropic-billing-header: cc_version=2.1.283.79f; cc_entrypoint=cli;"
        + cch_fields + " cc_is_subagent=true; cc_workload=test;"
    )
    original: Final[dict[str, JsonValue]] = {
        "messages": [{"role": "user", "content": 'cch=00000, cch=12345 and "system":[] remain unchanged 🦊'}],
        "metadata": {"nested": {"system": ["cch=00000"]}},
        "system": [
            {"cache_control": {"type": "ephemeral", "ttl": "1h"}, "text": text, "type": "text"},
            {"type": "text", "text": "Unrelated cch=00000 and cch=12345"},
        ],
        "tools": [{"name": "test", "input_schema": {"properties": {"model": {"type": "string"}}}}],
        "model": "claude-sonnet-5",
        "max_tokens": 32,
    }
    snapshot: Final = deepcopy(original)
    wire: Final = serialize_anthropic_subscription_request(original)
    actual: Final = json.loads(wire)
    match: Final = re.search(r" cch=([0-9a-f]{5});", actual["system"][0]["text"])
    assert match is not None
    assert len(re.findall(r"\bcch=", actual["system"][0]["text"])) == 1
    assert actual == {
        **snapshot,
        "system": [
            {
                **snapshot["system"][0],
                "text": (
                    "x-anthropic-billing-header: cc_version=2.1.283.79f; cc_entrypoint=cli;"
                    f" cch={match.group(1)}; cc_is_subagent=true; cc_workload=test;"
                ),
            },
            snapshot["system"][1],
        ],
    }
    assert original == snapshot
    assert wire.startswith('{"system":[{"type":"text","text":"x-anthropic-billing-header:')
    placeholder: Final = wire.replace(f"cch={match.group(1)}", "cch=00000", 1).encode()
    assert anthropic_subscription_cch(placeholder) == match.group(1)
    assert serialize_anthropic_subscription_request(actual) == wire


@pytest.mark.parametrize("system", [None, [], "text", [{"type": "text", "text": "Not attribution"}]])
def test_serializer_rejects_missing_attribution_instead_of_sending_unsigned_request(system: JsonValue) -> None:
    with pytest.raises(ValueError, match="attribution is missing"):
        serialize_anthropic_subscription_request({"system": system})


def test_serializer_rejects_attribution_outside_native_window() -> None:
    with pytest.raises(ValueError, match="signing window"):
        serialize_anthropic_subscription_request({
            "system": [{
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=" + "x" * 300 + "; cc_entrypoint=cli;",
            }],
        })


@pytest.mark.parametrize("field", ["cc_version=cch=00000; cc_entrypoint=cli;", "cc_version=caller; cc_entrypoint=cch=00000;"])
def test_serializer_rejects_ambiguous_placeholder_without_corrupting_attribution(field: str) -> None:
    body: Final = {"system": [{"type": "text", "text": "x-anthropic-billing-header: " + field}]}
    with pytest.raises(ValueError, match="ambiguous"):
        serialize_anthropic_subscription_request(body)
    assert body["system"][0]["text"] == "x-anthropic-billing-header: " + field


@pytest.mark.parametrize("changed", [
    {"stream": True},
    {"metadata": {"user_id": "another-session"}},
    {"messages": [{"role": "user", "content": "different content"}]},
    {"tools": [{"name": "Read", "input_schema": {"type": "object"}}]},
])
def test_serializer_resigns_after_body_changes(changed: dict[str, JsonValue]) -> None:
    original: Final = json.loads(_VECTORS[0]["body"])
    first: Final = json.loads(serialize_anthropic_subscription_request(original))
    second: Final = json.loads(serialize_anthropic_subscription_request({**first, **changed}))
    assert second["system"][0]["text"] != first["system"][0]["text"]
    assert {name: value for name, value in second.items() if name != "system"} == {
        name: value for name, value in {**first, **changed}.items() if name != "system"
    }


@pytest.mark.parametrize("provider", ["anthropic", "azure_ai", "vertex_ai", "bedrock/claude"])
@pytest.mark.parametrize("oauth", [False, True])
def test_messages_signing_remains_subscription_and_provider_scoped(provider: str, oauth: bool) -> None:
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
    from litellm.llms.anthropic.oauth_client import ManagedAnthropicOAuthToken
    from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig
    from litellm.llms.bedrock.claude_platform.messages_transformation import BedrockClaudePlatformMessagesConfig
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config: Final = (
        AnthropicMessagesConfig() if provider == "anthropic"
        else AzureAnthropicMessagesConfig() if provider == "azure_ai"
        else BedrockClaudePlatformMessagesConfig() if provider == "bedrock/claude"
        else ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-sonnet-5", provider=LlmProviders.VERTEX_AI,
        )
    )
    assert config is not None
    headers: Final = {"authorization": "Bearer sk-ant-oat-test"} if oauth else {"x-api-key": "sk-ant-api-test"}
    request: Final = json.loads(_VECTORS[0]["body"])
    original: Final = deepcopy(request)
    signed_headers, wire = config.sign_request(
        headers=headers, optional_params={}, request_data=request, api_base="https://api.anthropic.com/v1/messages",
        api_key=ManagedAnthropicOAuthToken("sk-ant-oat-test") if oauth else "sk-ant-api-test",
    )
    assert signed_headers == headers
    if provider == "anthropic" and oauth:
        assert wire == serialize_anthropic_subscription_request(request).encode()
    else:
        assert wire is None
    assert request == original
