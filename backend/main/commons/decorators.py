from functools import wraps
from typing import Type

import pydantic
import werkzeug
import werkzeug.exceptions
from flask import request

from main.commons import exceptions


def get_request_args():
    if request.method == "GET":
        return request.args.to_dict()

    return request.get_json() or {}


def parse_args_with(schema: Type[pydantic.BaseModel]):
    def decorator(f):
        @wraps(f)
        def wrapper(**kwargs):
            try:
                request_args = get_request_args()
                parsed_args = schema.parse_obj(request_args)
                return f(**kwargs, args=parsed_args)

            except werkzeug.exceptions.BadRequest as e:
                raise exceptions.BadRequest(error_message=e.description)

            except pydantic.ValidationError as e:
                raise exceptions.ValidationError(error_data=e.errors())

        return wrapper

    return decorator
