from decimal import Decimal
from typing import Annotated, Final, Literal

from pydantic import AliasChoices, BaseModel, Field, HttpUrl, StrictStr


class ArgumentSpec(BaseModel):
    name: str
    required: bool = False
    allowedValues: list[str | int | float | bool] | None = None


class VideoModel(BaseModel):
    model: str
    arguments: list[ArgumentSpec]
    inputs: list[ArgumentSpec] = Field(default_factory=list)


class ModelList(BaseModel):
    models: list[VideoModel]


class Capabilities(BaseModel):
    availableModels: dict[str, list[VideoModel] | ModelList]

    def text_to_video(self) -> tuple[VideoModel, ...]:
        entries: Final = self.availableModels.get("text_to_video", [])
        return tuple(entries.models if isinstance(entries, ModelList) else entries)


class Submission(BaseModel):
    generationId: StrictStr = Field(min_length=1, validation_alias=AliasChoices("generationId", "generation_id"))
    status: str
    creditsConsumed: int | None = Field(default=None, ge=0)


class VideoWork(BaseModel):
    contentType: str
    status: str | None = None
    url: HttpUrl | None = None


class TaskResult(BaseModel):
    generationId: StrictStr = Field(min_length=1, validation_alias=AliasChoices("generationId", "generation_id"))
    status: str
    works: list[VideoWork] = Field(default_factory=list)

    def billing_outcome(self) -> Literal["pending", "billable", "failed", "unconfirmed"]:
        status: Final = self.status.casefold()
        if status in ("failed", "cancelled"):
            return "failed"
        if status in ("queuing", "running", "submitted", "processing"):
            return "pending"
        if status in ("completed", "partial_completed", "succeed") and any(
            work.contentType == "video"
            and work.url is not None
            and (work.status or status).casefold() in ("completed", "succeed")
            for work in self.works
        ):
            return "billable"
        return "unconfirmed"


class VideoQuote(BaseModel, frozen=True):
    requested_seconds: Annotated[int, Field(strict=True, gt=0)]
    usd_per_second: Annotated[Decimal, Field(gt=0, max_digits=24, decimal_places=12)]

    def final_cost(self, task: TaskResult) -> Decimal | None:
        match task.billing_outcome():
            case "billable":
                return self.usd_per_second * self.requested_seconds
            case "failed":
                return Decimal(0)
            case _:
                return None
