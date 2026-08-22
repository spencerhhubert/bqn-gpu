"""Framework-independent structural types for execution backends."""

from __future__ import annotations

from numbers import Real
from typing import Callable, Iterable, Protocol, Sequence, TypeVar

from .host_value import HostValue, Shape


class BackendValue(Protocol):
    atom: bool

    @property
    def shape(self) -> Shape: ...

    def to_host(self) -> HostValue: ...


ValueT = TypeVar("ValueT", bound=BackendValue)


class ExecutionBackend(Protocol[ValueT]):
    def atom(self, value: Real) -> ValueT: ...

    def array(self, values: Iterable[Real], shape: Sequence[int]) -> ValueT: ...

    def from_host(self, value: HostValue) -> ValueT: ...

    def call(self, glyph: str, *arguments: ValueT) -> ValueT: ...

    def call_scalar(
        self,
        glyph: str,
        scalar: Real,
        scalar_left: bool,
        argument: ValueT,
    ) -> ValueT: ...

    def call_static(
        self,
        glyph: str,
        left_values: Sequence[int],
        left_atom: bool,
        argument: ValueT,
    ) -> ValueT: ...

    def reduce(self, glyph: str, argument: ValueT) -> ValueT: ...

    def insert(self, glyph: str, argument: ValueT) -> ValueT: ...

    def scan(self, glyph: str, argument: ValueT) -> ValueT: ...

    def map_function(
        self,
        modifier: str,
        rank_specification: Sequence[float],
        operand: object,
        arguments: Sequence[ValueT],
        function: Callable[[Sequence[ValueT]], ValueT],
    ) -> ValueT: ...
