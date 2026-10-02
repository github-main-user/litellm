from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.canonical_model_endpoints import (
    CanonicalWrite,
    _deployment_params,
    _fields,
    _mode,
    _model_info,
    _parse_body,
    _read,
    create_canonical_model,
    update_canonical_model,
)


def _payload() -> dict[str, object]:
    return {
        "name": "public-model",
        "group": "group-one",
        "base_model": "lab/original",
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


@pytest.mark.parametrize("base_model", ["lab/model-v1.2", "Lab_1/model_name"])
def test_canonical_base_model_is_written_and_returned(base_model: str) -> None:
    body: Final = _parse_body({**_payload(), "base_model": base_model, "connections": []})
    fields: Final = _fields(body, "admin")
    row: Final = SimpleNamespace(**fields, id="canonical-1", connections=[])

    assert fields["base_model"] == base_model
    assert _read(row).model_dump(mode="json")["base_model"] == base_model


@pytest.mark.parametrize("extra", [{}, {"base_model": None}])
def test_canonical_base_model_is_required(extra: dict[str, object]) -> None:
    payload: Final = {key: value for key, value in _payload().items() if key != "base_model"}
    with pytest.raises(HTTPException) as exc:
        _parse_body({**payload, **extra})
    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "invalid_request"


@pytest.mark.parametrize(
    "base_model",
    [
        "",
        "model",
        "/model",
        "lab/",
        "lab/model/extra",
        "lab /model",
        "lab/mo del",
        "lab/*",
        "lab/model\nextra",
        1,
        "lab/" + "m" * 252,
    ],
)
def test_canonical_base_model_rejects_invalid_identifiers(base_model: object) -> None:
    with pytest.raises(HTTPException) as exc:
        _parse_body({**_payload(), "base_model": base_model})

    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "invalid_request"


def test_canonical_base_model_does_not_change_inference_metadata() -> None:
    original: Final = CanonicalWrite.model_validate(_payload())
    tagged: Final = CanonicalWrite.model_validate({**_payload(), "base_model": "lab/model"})

    assert _deployment_params(tagged, tagged.connections[0]) == _deployment_params(original, original.connections[0])
    assert _model_info(tagged, "deployment-1", "chat") == _model_info(original, "deployment-1", "chat")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("extra", [{}, {"base_model": None}, {"base_model": ""}])
async def test_canonical_create_and_put_require_base_model(method: str, extra: dict[str, object]) -> None:
    payload: Final = {key: value for key, value in _payload().items() if key != "base_model"}
    with pytest.raises(HTTPException) as exc:
        if method == "POST":
            await create_canonical_model(user=UserAPIKeyAuth(user_id="admin"), raw={**payload, **extra})
        else:
            await update_canonical_model("canonical-1", user=UserAPIKeyAuth(user_id="admin"), raw={**payload, **extra})
    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "invalid_request"
