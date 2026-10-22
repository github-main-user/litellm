"""
Test Anthropic structured output with Pydantic models.

This test file verifies that Pydantic models with various constraints
are properly converted to Anthropic-compatible JSON schemas.
"""

import pytest
from pydantic import BaseModel, Field
from typing import List


class TestAnthropicStructuredOutput:
    """Test Anthropic structured output schema transformations."""

    def test_max_length_on_list_field_filtered(self):
        """
        Test that max_length on List fields is filtered out for Anthropic models.

        Anthropic doesn't support 'maxItems' property for array types in their
        output_format.schema, so we need to filter it out.

        Related issue: https://github.com/BerriAI/litellm/issues/19444
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        # Define a Pydantic model with max_length on a List field
        class ResponseModel(BaseModel):
            items: List[str] = Field(max_length=5, description="List of items")
            name: str = Field(description="Name field")

        config = AnthropicConfig()

        # Get the JSON schema from the Pydantic model
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        # Extract the actual schema
        schema = json_schema["json_schema"]["schema"]

        # Verify that maxItems is present in the raw schema (from Pydantic)
        assert "maxItems" in schema["properties"]["items"]

        # Now apply the Anthropic output format transformation
        response_format = {
            "type": "json_schema",
            "json_schema": {**json_schema["json_schema"], "strict": False},
        }

        output_format = config.map_response_format_to_anthropic_output_format(response_format)

        # Verify that maxItems is filtered out for Anthropic
        assert output_format is not None
        assert "schema" in output_format
        transformed_schema = output_format["schema"]

        # maxItems should be removed from the items property
        assert "maxItems" not in transformed_schema["properties"]["items"]

        # But other properties should remain
        assert "type" in transformed_schema["properties"]["items"]
        assert transformed_schema["properties"]["items"]["type"] == "array"
        assert "description" in transformed_schema["properties"]["items"]

    def test_min_length_on_list_field_filtered(self):
        """
        Test that min_length on List fields is filtered out for Anthropic models.

        Anthropic likely doesn't support 'minItems' either.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            items: List[str] = Field(min_length=2, description="List of items")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        response_format = {
            "type": "json_schema",
            "json_schema": {**json_schema["json_schema"], "strict": False},
        }

        output_format = config.map_response_format_to_anthropic_output_format(response_format)

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # minItems should be removed
        assert "minItems" not in transformed_schema["properties"]["items"]

    def test_nested_array_constraints_filtered(self):
        """
        Test that array constraints are filtered at all nesting levels.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class NestedItem(BaseModel):
            tags: List[str] = Field(max_length=3)

        class ResponseModel(BaseModel):
            items: List[NestedItem] = Field(max_length=5)

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        response_format = {
            "type": "json_schema",
            "json_schema": {**json_schema["json_schema"], "strict": False},
        }

        output_format = config.map_response_format_to_anthropic_output_format(response_format)

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # Top-level maxItems should be removed
        assert "maxItems" not in transformed_schema["properties"]["items"]

        # Nested maxItems should also be removed
        if "$defs" in transformed_schema:
            nested_item_schema = transformed_schema["$defs"].get("NestedItem", {})
            if "properties" in nested_item_schema and "tags" in nested_item_schema["properties"]:
                assert "maxItems" not in nested_item_schema["properties"]["tags"]

    def test_other_constraints_preserved(self):
        """
        Test that constraints are properly handled (removed from schema, added to description).

        Per Anthropic API requirements, constraints like minLength/maxLength and
        minimum/maximum must be removed from the schema but documented in descriptions.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            name: str = Field(max_length=100, min_length=1, description="Name")
            age: int = Field(ge=0, le=150, description="Age")
            items: List[str] = Field(description="Items list")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        response_format = {
            "type": "json_schema",
            "json_schema": {**json_schema["json_schema"], "strict": False},
        }

        output_format = config.map_response_format_to_anthropic_output_format(response_format)

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # String constraints should be REMOVED from schema (Anthropic doesn't support them)
        name_schema = transformed_schema["properties"]["name"]
        assert "maxLength" not in name_schema
        assert "minLength" not in name_schema
        # But constraint info should be added to description
        assert "description" in name_schema
        assert "minimum length: 1" in name_schema["description"]
        assert "maximum length: 100" in name_schema["description"]

        # Number constraints should be REMOVED from schema (Anthropic doesn't support them)
        age_schema = transformed_schema["properties"]["age"]
        assert "minimum" not in age_schema
        assert "maximum" not in age_schema
        # But constraint info should be added to description
        assert "description" in age_schema
        assert "minimum value: 0" in age_schema["description"]
        assert "maximum value: 150" in age_schema["description"]


class TestAnthropicOutputFormatSchemaBudget:
    """The $defs inlining in map_response_format_to_anthropic_output_format is byte-bounded."""

    @staticmethod
    def _response_format(schema: dict) -> dict:
        return {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}

    def test_schema_bomb_rejected(self):
        """A compact request whose $defs expand past the byte budget raises instead of materialising."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        big = {"type": "string", "description": "x" * 200_000}
        schema = {
            "type": "object",
            "$defs": {"Big": big},
            "properties": {f"p{i}": {"$ref": "#/$defs/Big"} for i in range(60)},
        }

        with pytest.raises(ValueError, match="budget"):
            AnthropicConfig().map_response_format_to_anthropic_output_format(self._response_format(schema))

    def test_normal_defs_still_resolve(self):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "$defs": {"Item": {"type": "string", "description": "an item"}},
            "properties": {"a": {"$ref": "#/$defs/Item"}, "b": {"$ref": "#/$defs/Item"}},
        }

        output_format = AnthropicConfig().map_response_format_to_anthropic_output_format(self._response_format(schema))
        assert output_format is not None
        resolved = output_format["schema"]["properties"]
        assert resolved["a"]["type"] == "string"
        assert resolved["b"]["type"] == "string"
        assert "$ref" not in str(resolved)


import copy
import json
import httpx
import respx
import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

PRESERVED_SCHEMA = {
    "type": "object",
    "properties": {
        "strict": {"type": "boolean"},
        "additionalProperties": {"type": "string"},
        "nested": {"type": "array", "items": {"$ref": "#/$defs/strict"}},
    },
    "required": ["strict", "additionalProperties", "nested"],
    "additionalProperties": False,
    "$defs": {
        "strict": {
            "type": "object",
            "properties": {"strict": {"type": "string"}},
            "required": ["strict"],
            "additionalProperties": False,
        }
    },
}


def preserved_response_format(strict=True, schema=None):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "description": "An exact answer",
            "strict": strict,
            "schema": copy.deepcopy(schema if schema is not None else PRESERVED_SCHEMA),
        },
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "minimum": 10},
        {"oneOf": [{"type": "number"}, {"type": "integer"}]},
        {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}, "additionalProperties": False},
        {"$ref": "#/$defs/x", "$defs": {"x": {"type": "integer", "maximum": 3}}},
        {"type": "array", "items": {"type": "integer", "minimum": 10}},
        {"type": "object", "additionalProperties": True},
    ],
)
def test_preserved_anthropic_rejects_strict_weakening(schema):
    with pytest.raises(litellm.UnsupportedParamsError, match="strict"):
        AnthropicConfig().map_response_format_to_anthropic_output_format(preserved_response_format(schema=schema))


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_preserved_documented_anthropic_native_capability_preserves_override(model, monkeypatch):
    assert AnthropicConfig._supports_model_capability(model, "supports_native_structured_output", "anthropic")
    monkeypatch.setitem(
        litellm.model_cost,
        f"anthropic/{model}",
        {"litellm_provider": "anthropic", "mode": "chat", "supports_native_structured_output": False},
    )
    assert not AnthropicConfig._supports_model_capability(model, "supports_native_structured_output", "anthropic")


def test_preserved_anthropic_nonstrict_compatibility():
    result = AnthropicConfig().map_response_format_to_anthropic_output_format(
        preserved_response_format(False, {"type": "integer", "minimum": 10})
    )
    assert "minimum" not in result["schema"]
    assert "10" in result["schema"]["description"]


@pytest.mark.parametrize('provider', ['anthropic'])
@pytest.mark.parametrize('strict', [True, False])
def test_preserved_downgrade_guards(provider, strict, monkeypatch):
    value = preserved_response_format(strict)
    monkeypatch.setattr(AnthropicConfig, '_supports_model_capability', lambda *args: False)
    call = lambda: AnthropicConfig().map_openai_params({'response_format': value}, {}, 'claude-3-haiku-20240307', False)
    if strict:
        with pytest.raises(litellm.UnsupportedParamsError, match='strict'):
            call()
    else:
        assert call() is not None or provider == 'gemini'


@pytest.mark.parametrize("provider", ["anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("strict", [True, False])
def test_preserved_provider_schema_reaches_wire_and_rejection_survives(provider, stream, strict):
    value = preserved_response_format(strict)
    if provider == "anthropic":
        model = "anthropic/claude-sonnet-4-5-20250929"
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": "schema rejected"}}
    else:
        model = "gemini/gemini-2.5-flash"
        error = {"error": {"code": 400, "message": "schema rejected", "status": "INVALID_ARGUMENT"}}
    with respx.mock(assert_all_called=True, assert_all_mocked=True) as mock:
        route = mock.post(url__regex=r"https://schema.test/.*").mock(return_value=httpx.Response(400, json=error))

        def invoke_provider():
            result = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Return JSON"}],
                response_format=value,
                stream=stream,
                api_key="synthetic-test-key",
                api_base="https://schema.test",
                num_retries=0,
            )
            if stream:
                list(result)

        with pytest.raises(litellm.BadRequestError, match="schema rejected"):
            invoke_provider()
        assert route.call_count == 1
        body = json.loads(route.calls[0].request.content)
        if provider == "anthropic":
            expected = (
                PRESERVED_SCHEMA
                if strict
                else AnthropicConfig().map_response_format_to_anthropic_output_format(value)["schema"]
            )
            assert body["output_format"]["schema"] == expected
            assert "tools" not in body
        else:
            assert body["generationConfig"]["response_json_schema"] == PRESERVED_SCHEMA


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_exact_claude_reasoning_reaches_public_thinking(model, asynchronous):
    with respx.mock(assert_all_called=True, assert_all_mocked=True) as mock:
        route = mock.post("https://schema.test/v1/messages").respond(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "stop at wire"}}
        )

        async def invoke_provider():
            call = litellm.acompletion if asynchronous else litellm.completion
            result = call(
                model=f"anthropic/{model}",
                api_key="synthetic",
                api_base="https://schema.test",
                messages=[{"role": "user", "content": "Think"}],
                reasoning_effort="high",
                num_retries=0,
            )
            if asynchronous:
                await result

        with pytest.raises(litellm.BadRequestError, match="stop at wire"):
            await invoke_provider()
        thinking = json.loads(route.calls[0].request.content)["thinking"]
        assert thinking["type"] in ("adaptive", "enabled")
        if thinking["type"] == "enabled":
            assert thinking["budget_tokens"] > 0
