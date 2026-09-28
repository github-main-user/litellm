import pytest
from pydantic import ValidationError

from litellm.proxy.management_endpoints.canonical_model_endpoints import (
    CanonicalWrite,
    _deployment_params,
    _mode,
    _model_info,
)


def _payload() -> dict[str, object]:
    return {
        "name": "public-model",
        "group": "group-one",
        "input_price_per_million_tokens": 2.0,
        "output_price_per_million_tokens": 3.0,
        "cache_read_price_per_million_tokens": 0.5,
        "cache_write_price_per_million_tokens": 1.0,
        "connections": [{"id": "binding-1", "provider_connection": "credential", "provider_model": "openai/gpt-4o"}],
    }


def test_canonical_prices_are_applied_to_deployment_per_token() -> None:
    body = CanonicalWrite.model_validate(_payload())

    info = _model_info(body, "deployment-1", "chat")
    assert info["id"] == "deployment-1"
    assert info["mode"] == "chat"
    assert info["group"] == body.group
    assert info["input_cost_per_token"] == 0.000002
    assert info["output_cost_per_token"] == 0.000003
    assert info["cache_read_input_token_cost"] == 0.0000005
    assert info["cache_creation_input_token_cost"] == 0.000001
    assert info["cache_creation_input_token_cost_above_1hr"] == info["cache_creation_input_token_cost"]
    assert info["cache_creation_input_token_cost_above_200k_tokens"] == info["cache_creation_input_token_cost"]
    assert info["cache_read_input_token_cost_above_200k_tokens"] == info["cache_read_input_token_cost"]


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf")])
def test_canonical_rejects_invalid_prices(price: float) -> None:
    with pytest.raises(ValidationError):
        CanonicalWrite.model_validate({**_payload(), "input_price_per_million_tokens": price})


def test_canonical_rejects_duplicate_connection_ids() -> None:
    payload = _payload()
    connections = payload["connections"]
    assert isinstance(connections, list)
    with pytest.raises(ValidationError):
        CanonicalWrite.model_validate({**payload, "connections": connections * 2})


def test_canonical_defaults_cache_prices_and_allows_unbound_credentials() -> None:
    payload = _payload()
    body = CanonicalWrite.model_validate(
        {
            **payload,
            "group": None,
            "cache_read_price_per_million_tokens": None,
            "cache_write_price_per_million_tokens": None,
            "connections": [{"provider_connection": None, "provider_model": "openai/gpt-4o"}],
        }
    )
    params = _deployment_params(body, body.connections[0])
    assert params["cache_read_input_token_cost"] == params["input_cost_per_token"]
    assert params["cache_creation_input_token_cost"] == params["input_cost_per_token"]
    assert "litellm_credential_name" not in params
    assert params["cache_creation_input_token_cost_above_1hr"] == params["cache_creation_input_token_cost"]
    assert params["cache_creation_input_token_cost_above_200k_tokens"] == params["cache_creation_input_token_cost"]
    assert params["cache_read_input_token_cost_above_200k_tokens"] == params["cache_read_input_token_cost"]


def test_chatgpt_models_use_responses() -> None:
    assert _mode("chatgpt/example") == "responses"


def test_canonical_supports_zero_bindings() -> None:
    payload = _payload()
    assert CanonicalWrite.model_validate({**payload, "connections": []}).connections == []
