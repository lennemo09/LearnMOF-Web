from main.schemas.base import BaseSchema


class UpdateProcessImage(BaseSchema):
    image_paths: list | None
    process_id: str | None
