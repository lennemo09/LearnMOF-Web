from pydantic import BaseModel

from main.schemas.base import BaseSchema


class GetFilteredImagesSchema(BaseSchema):
    approved: bool | None
    image_ids: list | None
    assigned_label: str | None


class UpdateApproval(BaseModel):
    approved: bool | None
    label: str | None
