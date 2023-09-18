from pydantic import BaseModel

from main.schemas.base import BaseSchema


class GetFilteredImagesSchema(BaseSchema):
    approved: bool | None
    image_ids: list | None
    assigned_label: str | None
    magnification: int | None
    reaction_time: int | None
    temperature: int | None
    linker: str | None
    year: str | None
    month: str | None
    day: str | None


class UpdateApproval(BaseModel):
    approved: bool | None
    label: str | None
