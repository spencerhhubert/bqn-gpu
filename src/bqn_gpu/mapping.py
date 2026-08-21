"""Shape planning for dense Cells, Rank, Each, and Table calls."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterator, Sequence

from .errors import DomainError, ShapeError
from .host_value import Shape


@dataclass(frozen=True)
class MappingPlan:
    frame_shape: Shape
    argument_frames: tuple[Shape, ...]
    cell_ranks: tuple[int, ...]
    table: bool

    def indices(self) -> Iterator[tuple[tuple[int, ...], ...]]:
        ranges = tuple(range(length) for length in self.frame_shape)
        for index in product(*ranges):
            if self.table and len(self.argument_frames) == 2:
                split = len(self.argument_frames[0])
                yield (index[:split], index[split:])
            else:
                yield tuple(index[: len(frame)] for frame in self.argument_frames)


def plan_mapping(
    modifier: str,
    shapes: Sequence[Shape],
    rank_specification: Sequence[float] = (),
) -> MappingPlan:
    if len(shapes) not in {1, 2}:
        raise DomainError("mapping modifiers require one or two arguments")
    if modifier == "⌜":
        return MappingPlan(
            frame_shape=tuple(axis for shape in shapes for axis in shape),
            argument_frames=tuple(shapes),
            cell_ranks=(0,) * len(shapes),
            table=True,
        )
    if modifier == "¨":
        cell_ranks = (0,) * len(shapes)
    elif modifier == "˘":
        cell_ranks = tuple(max(0, len(shape) - 1) for shape in shapes)
    elif modifier == "⎉":
        selected = _select_rank_specification(rank_specification, len(shapes))
        cell_ranks = tuple(
            _effective_rank(value, len(shape))
            for value, shape in zip(selected, shapes, strict=True)
        )
    else:
        raise DomainError(f"mapping modifier {modifier!r} is not implemented")

    frames = tuple(
        shape[: len(shape) - cell_rank] if cell_rank else shape
        for shape, cell_rank in zip(shapes, cell_ranks, strict=True)
    )
    frame_shape = _agree_frames(frames)
    return MappingPlan(frame_shape, frames, cell_ranks, table=False)


def _select_rank_specification(
    values: Sequence[float],
    arity: int,
) -> tuple[float, ...]:
    if len(values) == 1:
        return tuple(values[0] for _ in range(arity))
    if arity == 2 and len(values) == 2:
        return tuple(values)
    if len(values) == 3:
        return (values[0],) if arity == 1 else (values[1], values[2])
    expected = "one or three" if arity == 1 else "one, two, or three"
    raise DomainError(f"Rank requires {expected} rank numbers for this valence")


def _effective_rank(value: float, argument_rank: int) -> int:
    if math.isinf(value):
        return argument_rank if value > 0 else 0
    integer = int(value)
    if integer != value:
        raise DomainError("Rank requires whole-number ranks or positive infinity")
    if integer >= 0:
        return min(integer, argument_rank)
    return max(0, argument_rank + integer)


def _agree_frames(frames: Sequence[Shape]) -> Shape:
    if len(frames) == 1:
        return frames[0]
    left, right = frames
    lower, higher = (left, right) if len(left) <= len(right) else (right, left)
    if higher[: len(lower)] != lower:
        raise ShapeError(
            "BQN mapping frame agreement requires the lower-rank frame to "
            f"prefix the higher-rank frame, got {left} and {right}"
        )
    return higher
