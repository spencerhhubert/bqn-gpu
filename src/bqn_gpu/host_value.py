"""Backend-independent values used at integration and test boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Sequence

from .errors import DomainError


Shape = tuple[int, ...]


def checked_shape(shape: Sequence[int]) -> Shape:
    checked: list[int] = []
    for length in shape:
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise DomainError(f"shape lengths must be integers, got {length!r}")
        value = int(length)
        if value < 0:
            raise DomainError(f"shape lengths must be non-negative, got {value}")
        checked.append(value)
    return tuple(checked)


def checked_numbers(values: Iterable[Real]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise DomainError(f"only real numeric values are supported, got {value!r}")
        result.append(float(value))
    return tuple(result)


@dataclass(frozen=True)
class HostValue:
    """A dense real BQN atom or array.

    ``atom`` distinguishes a numeric atom from a rank-0 array. Both contain
    one number and both have an empty shape, but they are different BQN values.
    """

    atom: bool
    shape: Shape
    data: tuple[float, ...]

    def __post_init__(self) -> None:
        shape = checked_shape(self.shape)
        data = checked_numbers(self.data)
        if self.atom and shape:
            raise DomainError("an atom cannot have array axes")
        expected = 1 if self.atom else math.prod(shape)
        if len(data) != expected:
            kind = "atom" if self.atom else f"array with shape {shape}"
            raise DomainError(f"{kind} requires {expected} values, got {len(data)}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", data)

    @classmethod
    def from_atom(cls, value: Real) -> HostValue:
        return cls(atom=True, shape=(), data=checked_numbers((value,)))

    @classmethod
    def from_array(cls, values: Iterable[Real], shape: Sequence[int]) -> HostValue:
        return cls(atom=False, shape=checked_shape(shape), data=checked_numbers(values))
