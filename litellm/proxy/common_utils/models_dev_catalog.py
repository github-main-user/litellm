import asyncio
import json
from collections.abc import Callable
from io import BytesIO
from time import monotonic
from typing import Annotated, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter, ValidationError, model_validator

TokenLimit = Annotated[int, Field(strict=True, gt=0)]
BudgetMinimum = Annotated[float, Field(strict=True, ge=-1, allow_inf_nan=False)]
BudgetMaximum = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ModelLimits(BaseModel):
    model_config = ConfigDict(extra="ignore")
    context: TokenLimit | None = None
    input: TokenLimit | None = None
    output: TokenLimit | None = None


class ModelModalities(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input: list[Literal["text", "image", "audio", "video", "pdf"]] | None = None
    output: list[Literal["text", "image", "audio", "video", "pdf"]] | None = None


class EffortOption(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["effort"]
    values: list[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "default", "null"] | None]


class ToggleOption(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["toggle"]


class BudgetOption(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["budget_tokens"]
    min: BudgetMinimum | None = None
    max: BudgetMaximum | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "BudgetOption":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Minimum budget exceeds maximum")
        return self


ReasoningOption = Annotated[EffortOption | ToggleOption | BudgetOption, Field(discriminator="type")]


class CatalogMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    description: str | None = None
    family: str | None = None
    attachment: StrictBool | None = None
    reasoning: StrictBool | None = None
    reasoning_options: list[ReasoningOption] | None = None
    tool_call: StrictBool | None = None
    structured_output: StrictBool | None = None
    temperature: StrictBool | None = None
    knowledge: str | None = None
    release_date: str | None = None
    last_updated: str | None = None
    modalities: ModelModalities | None = None
    open_weights: StrictBool | None = None
    limit: ModelLimits | None = None


class SourceProvider(BaseModel):
    model_config = ConfigDict(extra="ignore")
    models: dict[str, object] = Field(default_factory=dict)


_SOURCE: Final = TypeAdapter(dict[str, SourceProvider])
_MAX_BYTES: Final = 32 * 1024 * 1024


class ModelsDevCatalog:
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._lock = asyncio.Lock()
        self._providers: dict[str, SourceProvider] = {}
        self._next_refresh = 0.0

    async def _fetch(self) -> dict[str, SourceProvider]:
        async with httpx.AsyncClient(transport=self._transport, timeout=10.0) as client:
            async with client.stream("GET", "https://models.dev/api.json") as response:
                response.raise_for_status()
                with BytesIO() as body:
                    async for chunk in response.aiter_bytes():
                        if body.tell() + len(chunk) > _MAX_BYTES:
                            raise ValueError("models.dev response exceeds size limit")
                        body.write(chunk)
                    parsed: Final = _SOURCE.validate_python(json.loads(body.getvalue()))
                    providers: Final = {
                        key: value for key, value in parsed.items() if key.lower() not in ("void", "void-api")
                    }
                    if not any(provider.models for provider in providers.values()):
                        raise ValueError("models.dev response contains no source models")
                    return providers

    async def load(self) -> "ModelsDevCatalog":
        if self._clock() < self._next_refresh:
            return self
        async with self._lock:
            if self._clock() < self._next_refresh:
                return self
            try:
                providers: Final = await asyncio.wait_for(self._fetch(), timeout=15.0)
            except (httpx.HTTPError, ValueError, asyncio.TimeoutError):
                self._next_refresh = self._clock() + 60.0
                return self
            self._providers = providers
            self._next_refresh = self._clock() + 3600.0
        return self

    def metadata(self, base_model: str) -> CatalogMetadata:
        provider, separator, model = base_model.partition("/")
        source: Final = (
            self._providers.get(provider) if separator and provider.lower() not in ("void", "void-api") else None
        )
        raw: Final = source.models.get(model) if source is not None else None
        try:
            native: Final = CatalogMetadata.model_validate(raw) if raw is not None else CatalogMetadata()
        except ValidationError:
            return CatalogMetadata()
        options: Final = translated_controls(provider, native)
        return native.model_copy(update={"reasoning_options": options})


def translated_controls(provider: str, native: CatalogMetadata) -> list[ReasoningOption] | None:
    if provider not in ("anthropic", "openai") or native.reasoning is not True or native.reasoning_options is None:
        return None
    allowed: Final = frozenset(("minimal", "low", "medium", "high", "xhigh", "max")) | (
        frozenset(("none",)) if provider == "openai" else frozenset()
    )
    options: Final = tuple(
        EffortOption(type="effort", values=[value for value in option.values if value in allowed])
        if isinstance(option, EffortOption)
        else option
        for option in native.reasoning_options
        if isinstance(option, EffortOption) or provider == "anthropic"
    )
    filtered: Final = [option for option in options if not isinstance(option, EffortOption) or option.values]
    return None if not filtered and native.reasoning_options else filtered


models_dev_catalog: Final = ModelsDevCatalog()
