from __future__ import annotations

import math

import pytest

from bqn_gpu.errors import DomainError, ShapeError
from bqn_gpu.mapping import plan_mapping


def test_cells_and_each_plan_major_cells_and_atoms() -> None:
    cells = plan_mapping("˘", [(2, 3), (2, 3)])
    assert cells.frame_shape == (2,)
    assert cells.argument_frames == ((2,), (2,))
    assert cells.cell_ranks == (1, 1)
    assert list(cells.indices()) == [((0,), (0,)), ((1,), (1,))]

    each = plan_mapping("¨", [(2, 3)])
    assert each.frame_shape == (2, 3)
    assert each.cell_ranks == (0,)


def test_table_concatenates_frames_and_splits_indices() -> None:
    table = plan_mapping("⌜", [(2,), (3, 4)])
    assert table.frame_shape == (2, 3, 4)
    assert table.cell_ranks == (0, 0)
    assert next(table.indices()) == ((0,), (0, 0))


@pytest.mark.parametrize(
    ("specification", "shapes", "expected"),
    [
        ([1], [(2, 3), (3,)], (1, 1)),
        ([1, 2], [(2, 3), (2, 2, 3)], (1, 2)),
        ([0, 1, 2], [(2, 3)], (0,)),
        ([0, 1, 2], [(2, 3), (2, 2, 3)], (1, 2)),
        ([-1], [(2, 3, 4)], (2,)),
        ([99], [(2, 3)], (2,)),
        ([math.inf], [(2, 3)], (2,)),
        ([-math.inf], [(2, 3)], (0,)),
    ],
)
def test_rank_selects_and_clamps_cell_ranks(
    specification: list[float],
    shapes: list[tuple[int, ...]],
    expected: tuple[int, ...],
) -> None:
    assert plan_mapping("⎉", shapes, specification).cell_ranks == expected


def test_dyadic_frames_use_leading_axis_agreement() -> None:
    plan = plan_mapping("⎉", [(2, 3, 5), (2, 5)], [1])
    assert plan.frame_shape == (2, 3)
    assert plan.argument_frames == ((2, 3), (2,))
    assert list(plan.indices())[-1] == ((1, 2), (1,))


def test_rank_rejects_invalid_specifications_and_frames() -> None:
    with pytest.raises(DomainError, match="one or three"):
        plan_mapping("⎉", [(2, 3)], [0, 1])
    with pytest.raises(DomainError, match="whole-number"):
        plan_mapping("⎉", [(2, 3)], [0.5])
    with pytest.raises(ShapeError, match="prefix"):
        plan_mapping("⎉", [(2, 3, 5), (4, 5)], [1])
