import pytest

import litellm
from litellm.proxy.management_endpoints.model_resolution import ResolutionRequest, resolve_identity


@pytest.mark.parametrize(
    "provider,lab",
    [("openai", "openai"), ("anthropic", "anthropic"), ("chatgpt", "openai"), ("deepseek", "deepseek")],
)
def test_native(provider, lab, monkeypatch):
    monkeypatch.setitem(litellm.model_cost, f"{provider}/test", {"litellm_provider": provider, "mode": "responses"})
    assert resolve_identity(f"{lab}/test", "account", provider).provider_model == f"{provider}/test"
    assert resolve_identity("other/test", "account", provider).status == "incompatible"


def test_saved_conflicts():
    assert (
        resolve_identity("openai/test", "a", "openai", ["openai/custom", "openai/custom"]).provider_model
        == "openai/custom"
    )
    assert resolve_identity("openai/test", "a", "openai", ["openai/one", "openai/two"]).status == "unknown"
    assert resolve_identity("openai/test", "a", "openrouter", ["openrouter/test"]).status == "unknown"


def test_deepseek_unknown_model_is_not_resolved():
    result = resolve_identity("deepseek/void-test-nonexistent-model", "account", "deepseek")
    assert result.model_dump() == {
        "provider_connection": "account",
        "status": "unknown",
        "provider_model": None,
    }


def test_deepseek_saved_binding_is_preserved():
    result = resolve_identity("deepseek/test", "account", "deepseek", ("deepseek/custom",))
    assert result.model_dump() == {
        "provider_connection": "account",
        "status": "resolved",
        "provider_model": "deepseek/custom",
    }


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
def test_cometapi_resolves_supported_deepseek_bindings(model):
    assert resolve_identity(f"deepseek/{model}", "comet", "cometapi").model_dump() == {
        "provider_connection": "comet",
        "status": "resolved",
        "provider_model": f"cometapi/{model}",
    }


def test_cometapi_preserves_saved_deepseek_alias():
    assert (
        resolve_identity("deepseek/deepseek-flash", "comet", "cometapi", ("cometapi/saved",)).provider_model
        == "cometapi/saved"
    )


@pytest.mark.parametrize("base_model", ["deepseek/not-a-known-model", "openai/deepseek-flash"])
def test_cometapi_does_not_guess_unknown_or_other_lab_bindings(base_model):
    assert resolve_identity(base_model, "comet", "cometapi").model_dump() == {
        "provider_connection": "comet",
        "status": "unknown",
        "provider_model": None,
    }


def test_cometapi_conflicting_saved_bindings_remain_ambiguous():
    assert (
        resolve_identity("deepseek/deepseek-flash", "comet", "cometapi", ("cometapi/one", "cometapi/two")).status
        == "unknown"
    )


def test_request_validation():
    with pytest.raises(ValueError):
        ResolutionRequest(base_model="bad", provider_connections=[])
    with pytest.raises(ValueError):
        ResolutionRequest(base_model="openai/test", provider_connections=[None] * 201)


@pytest.mark.asyncio
async def test_endpoint_uses_provider_despite_custom_url_and_never_exposes_secrets(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.models.credentials import CredentialItem
    from litellm.repositories.credentials_repository import CredentialsRepository

    from litellm.proxy.management_endpoints import canonical_model_endpoints as endpoint

    table = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                SimpleNamespace(
                    connections=[SimpleNamespace(provider_connection="saved", provider_model="openai/custom")]
                )
            ]
        )
    )
    client = SimpleNamespace(db=SimpleNamespace(litellm_canonicalmodel=table))
    monkeypatch.setattr(endpoint, "_client", lambda: client)
    monkeypatch.setattr(endpoint, "WriterPinnedClient", lambda db: SimpleNamespace(db=db))
    lookup = AsyncMock(
        return_value=CredentialItem(
            credential_name="native",
            credential_info={"provider": "openai"},
            credential_values={"api_base": "https://custom.invalid", "api_key": "SECRET"},
        )
    )
    monkeypatch.setattr(CredentialsRepository, "find_by_name", lookup)
    results = await endpoint.resolve_models(
        ResolutionRequest(base_model="openai/test", provider_connections=["native"]), SimpleNamespace()
    )
    assert results[0].provider_model == "openai/custom"
    assert "SECRET" not in results[0].model_dump_json()
    assert "api_base" not in results[0].model_dump_json()
    assert {call.args[0] for call in lookup.call_args_list} == {"native", "saved"}
