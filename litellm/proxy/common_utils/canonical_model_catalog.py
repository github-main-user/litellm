from collections.abc import Container
from datetime import datetime
from decimal import Decimal
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from litellm.proxy.common_utils.models_dev_catalog import CatalogMetadata, ModelsDevCatalog


class CatalogDeployment(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    blocked: bool


class CatalogConnection(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    deployment: CatalogDeployment | None


class CatalogIdentity(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    base_model: str
    created_at: datetime
    connections: list[CatalogConnection]


class CatalogRecord(CatalogIdentity):
    video_price_per_second: None = None
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    cache_read_price_per_million_tokens: Decimal
    cache_write_price_per_million_tokens: Decimal


class VideoCatalogRecord(CatalogIdentity):
    video_price_per_second: Decimal


class CatalogPricing(BaseModel):
    unit: Literal["USD_per_million_tokens"] = "USD_per_million_tokens"
    input: str
    output: str
    cache_read: str
    cache_write: str


class VideoCatalogPricing(BaseModel):
    unit: Literal["USD_per_second"] = "USD_per_second"
    per_second: str


class CatalogEntry(CatalogMetadata):
    id: str
    base_model: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    pricing: CatalogPricing | VideoCatalogPricing


def catalog_entries(
    records: tuple[CatalogRecord | VideoCatalogRecord, ...],
    owner: str,
    available_names: Container[str] | None,
    hidden_names: Container[str],
    metadata: ModelsDevCatalog | None = None,
) -> tuple[CatalogEntry, ...]:
    visible: Final = (
        row
        for row in records
        if "*" not in row.name
        and row.name not in hidden_names
        and (available_names is None or row.name in available_names)
        and any(
            connection.deployment is not None and not connection.deployment.blocked for connection in row.connections
        )
    )
    return tuple(
        CatalogEntry(
            id=row.name,
            base_model=row.base_model,
            created=int(row.created_at.timestamp()),
            owned_by=owner,
            **(metadata.metadata(row.base_model).model_dump(exclude_none=True) if metadata is not None else {}),
            pricing=VideoCatalogPricing(per_second=format(row.video_price_per_second, "f"))
            if isinstance(row, VideoCatalogRecord)
            else CatalogPricing(
                input=format(row.input_price_per_million_tokens, "f"),
                output=format(row.output_price_per_million_tokens, "f"),
                cache_read=format(row.cache_read_price_per_million_tokens, "f"),
                cache_write=format(row.cache_write_price_per_million_tokens, "f"),
            ),
        )
        for row in visible
    )
