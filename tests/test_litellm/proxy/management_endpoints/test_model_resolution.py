import pytest
import litellm
from litellm.proxy.management_endpoints.model_resolution import ResolutionRequest, resolve_identity


@pytest.mark.parametrize("provider,lab", [("openai", "openai"), ("anthropic", "anthropic"), ("chatgpt", "openai")])
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
    from litellm.proxy.management_endpoints import canonical_model_endpoints as endpoint
    from litellm.repositories.credentials_repository import CredentialsRepository

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
