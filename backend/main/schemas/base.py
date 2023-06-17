from pydantic import BaseModel, Extra


class BaseSchema(BaseModel):
    class Config:
        extra = Extra.ignore
        orm_mode = True
        anystr_strip_whitespace = True
