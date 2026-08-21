"""Documented JSON interchange for BQN numeric atoms and dense arrays."""

from __future__ import annotations

import json
from numbers import Real
from typing import Any

from .errors import DomainError
from .host_value import HostValue


def loads_host_value(text: str) -> HostValue:
    return decode_host_value(json.loads(text))


def decode_host_value(value: Any) -> HostValue:
    if isinstance(value, bool):
        raise DomainError("JSON booleans are not BQN numeric values")
    if isinstance(value, Real):
        return HostValue.from_atom(value)
    if isinstance(value, list):
        shape, data = _rectangular(value)
        return HostValue.from_array(data, shape)
    if isinstance(value, dict) and set(value) == {"shape", "data"}:
        if not isinstance(value["shape"], list) or not isinstance(value["data"], list):
            raise DomainError("explicit array JSON requires list-valued shape and data")
        return HostValue.from_array(value["data"], value["shape"])
    raise DomainError(
        "BQN JSON input must be a number, rectangular nested numeric list, "
        "or an object with shape and data"
    )


def encode_host_value(value: HostValue) -> float | dict[str, list[int] | list[float]]:
    if value.atom:
        return value.data[0]
    return {"shape": list(value.shape), "data": list(value.data)}


def dumps_host_value(value: HostValue) -> str:
    return json.dumps(encode_host_value(value), ensure_ascii=False, allow_nan=False)


def _rectangular(value: list[Any]) -> tuple[tuple[int, ...], list[float]]:
    if not value:
        return (0,), []
    children = [_rectangular_child(item) for item in value]
    child_shape = children[0][0]
    if any(shape != child_shape for shape, _ in children[1:]):
        raise DomainError("nested JSON arrays must be rectangular")
    data = [number for _, child_data in children for number in child_data]
    return (len(value),) + child_shape, data


def _rectangular_child(value: Any) -> tuple[tuple[int, ...], list[float]]:
    if isinstance(value, bool):
        raise DomainError("JSON booleans are not BQN numeric values")
    if isinstance(value, Real):
        return (), [float(value)]
    if isinstance(value, list):
        return _rectangular(value)
    raise DomainError("nested JSON arrays may contain only numbers or arrays")
