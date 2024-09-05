from contextlib import AbstractContextManager
from typing import Any, Callable, TypeVar


T = TypeVar("T", bound=AbstractContextManager)
U = TypeVar("U")
def with_context(manager: T, callback: Callable[[T], U]) -> U:
    with manager:
        return callback(manager)


Primitive = str | int | float | bool | None
def unwrap_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Primitive]:
    prefix = f"{prefix}." if len(prefix) > 0 else ""
    unwrapped = {}
    for key, value in data.items():
        if isinstance(value, Primitive):
            unwrapped[f"{prefix}{key}"] = value
        elif isinstance(value, dict):
            unwrapped.update(unwrap_dict(value, prefix=f"{prefix}{key}"))
        elif isinstance(value, list):
            unwrapped.update(unwrap_dict(dict(enumerate(value)), prefix=f"{prefix}{key}"))
        else:
            raise ValueError(f"Unsupported data type in dictionary: {key}={value}")
    return unwrapped
