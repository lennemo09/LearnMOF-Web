from pydantic import BaseModel

from main.schemas.base import BaseSchema


class GetFilteredImagesSchema(BaseSchema):
    approved: bool | None


class UpdateApproval(BaseModel):
    approved: bool | None
    label: str | None
