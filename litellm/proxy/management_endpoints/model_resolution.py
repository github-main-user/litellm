from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

import litellm


class ResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    base_model: str = Field(
        min_length=3, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    provider_connections: list[str | None] = Field(max_length=200)


class ModelResolution(BaseModel):
    provider_connection: str | None
    status: Literal["resolved", "unknown", "incompatible"]
    provider_model: str | None = None


def resolve_identity(
    base_model: str, provider_connection: str | None, provider: str | None, known_models: Sequence[str] = ()
) -> ModelResolution:
    lab, _, model = base_model.partition("/")
    native: Final = provider in ("anthropic", "openai", "chatgpt")
    expected_lab: Final = "openai" if provider == "chatgpt" else provider
    if native and lab != expected_lab:
        return ModelResolution(provider_connection=provider_connection, status="incompatible")
    if not native:
        return ModelResolution(provider_connection=provider_connection, status="unknown")
    saved: Final = frozenset(item for item in known_models if item.partition("/")[0] == provider)
    if saved:
        return ModelResolution(
            provider_connection=provider_connection,
            status="resolved" if len(saved) == 1 else "unknown",
            provider_model=next(iter(saved)) if len(saved) == 1 else None,
        )
    candidate: Final = f"{provider}/{model}"
    metadata: Final = litellm.model_cost.get(candidate) or litellm.model_cost.get(model)
    if metadata and metadata.get("litellm_provider") == provider and metadata.get("mode") in ("chat", "responses"):
        return ModelResolution(provider_connection=provider_connection, status="resolved", provider_model=candidate)
    return ModelResolution(provider_connection=provider_connection, status="unknown")
