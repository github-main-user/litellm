from typing import Final

import pytest

import litellm
from litellm import CustomLLM
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    is_registered_custom_provider,
)

CUSTOM_PROVIDER: Final = "test-onprem-llm"


@pytest.fixture
def registered_custom_provider(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": CUSTOM_PROVIDER, "custom_handler": CustomLLM()}])
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    return CUSTOM_PROVIDER


def test_get_llm_provider_resolves_custom_provider_map_prefix_before_first_completion(
    registered_custom_provider: str,
) -> None:
    assert registered_custom_provider not in litellm.provider_list

    model, provider, dynamic_api_key, api_base = get_llm_provider(model=f"{registered_custom_provider}/my-model")

    assert (model, provider, dynamic_api_key, api_base) == ("my-model", registered_custom_provider, None, None)


def test_get_llm_provider_strips_prefix_when_custom_provider_passed_explicitly(
    registered_custom_provider: str,
) -> None:
    model, provider, _, api_base = get_llm_provider(
        model="my-model",
        custom_llm_provider=registered_custom_provider,
        api_base="http://onprem.internal:8080",
    )

    assert (model, provider, api_base) == ("my-model", registered_custom_provider, "http://onprem.internal:8080")


def test_get_llm_provider_still_rejects_unregistered_prefix(registered_custom_provider: str) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        get_llm_provider(model="not-registered-llm/my-model")


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(CUSTOM_PROVIDER, True), ("not-registered-llm", False), (None, False), ("", False)],
)
def test_is_registered_custom_provider(registered_custom_provider: str, candidate: str | None, expected: bool) -> None:
    assert is_registered_custom_provider(candidate) is expected


@pytest.mark.parametrize("explicit_provider", [False, True])
@pytest.mark.parametrize("endpoint", ["completion", "embedding"])
def test_removed_copilot_provider_is_rejected(endpoint: str, explicit_provider: bool) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided|Unmapped LLM provider"):
        if endpoint == "completion":
            litellm.completion(
                model="gpt-5.4" if explicit_provider else "github_copilot/gpt-5.4",
                custom_llm_provider="github_copilot" if explicit_provider else None,
                messages=[{"role": "user", "content": "hello"}],
            )
        else:
            litellm.embedding(
                model="text-embedding-3-small" if explicit_provider else "github_copilot/text-embedding-3-small",
                custom_llm_provider="github_copilot" if explicit_provider else None,
                input=["hello"],
            )


def test_github_models_api_still_resolves() -> None:
    model, provider, api_key, api_base = get_llm_provider(model="github/gpt-5.4", api_key="github-models-key")

    assert (model, provider, api_key) == ("gpt-5.4", "github", "github-models-key")
    assert api_base is not None
    assert api_base.startswith("https://")
