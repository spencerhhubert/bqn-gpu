from __future__ import annotations

import math
import os
import random

import pytest

from bqn_gpu import HostValue, ShapeError, TinygradBackend, TinygradValue
from cbqn_oracle import CBQNOracle


def assert_matches(actual: TinygradValue, expected: HostValue) -> None:
    host = actual.to_host()
    assert host.atom == expected.atom
    assert host.shape == expected.shape
    assert len(host.data) == len(expected.data)
    for actual_number, expected_number in zip(host.data, expected.data, strict=True):
        if math.isnan(expected_number):
            assert math.isnan(actual_number)
        else:
            assert actual_number == expected_number


def oracle_call(
    backend: TinygradBackend, cbqn: CBQNOracle, *arguments: HostValue
) -> tuple[TinygradValue, HostValue]:
    device_values = tuple(backend.from_host(argument) for argument in arguments)
    actual = backend.call("+", *device_values)
    expected = cbqn.call("+", *arguments)
    return actual, expected


@pytest.mark.parametrize(
    "arguments",
    [
        (HostValue.from_atom(2),),
        (HostValue.from_array([1, -2, math.inf], (3,)),),
        (HostValue.from_atom(2), HostValue.from_atom(3)),
        (HostValue.from_atom(10), HostValue.from_array([1, 2, 3], (3,))),
        (HostValue.from_array([1], ()), HostValue.from_array([2, 3], (2,))),
        (
            HostValue.from_array([10, 20, 30], (3,)),
            HostValue.from_array([1, 2, 3, 4, 5, 6], (3, 2)),
        ),
        (
            HostValue.from_array([1, 2, 3, 4, 5, 6], (3, 2)),
            HostValue.from_array([10, 20, 30], (3,)),
        ),
        (HostValue.from_array([], (0,)), HostValue.from_array([], (0, 3))),
        (HostValue.from_array([1, 2], (2,)), HostValue.from_array([], (2, 0))),
        (
            HostValue.from_array([math.inf, -math.inf, math.nan, -0.0], (4,)),
            HostValue.from_array([1, 1, 1, 0], (4,)),
        ),
    ],
)
def test_deterministic_cases_match_cbqn(
    backend: TinygradBackend,
    cbqn: CBQNOracle,
    arguments: tuple[HostValue, ...],
) -> None:
    actual, expected = oracle_call(backend, cbqn, *arguments)
    assert_matches(actual, expected)


def test_incompatible_shapes_are_rejected(backend: TinygradBackend) -> None:
    pairs = [((2,), (3,)), ((2, 2), (2, 3)), ((2,), (3, 2))]
    for w_shape, x_shape in pairs:
        w = backend.array(range(math.prod(w_shape)), w_shape)
        x = backend.array(range(math.prod(x_shape)), x_shape)
        with pytest.raises(ShapeError):
            backend.add(w, x)


def test_seeded_random_cases_match_cbqn(
    backend: TinygradBackend, cbqn: CBQNOracle
) -> None:
    seed = int(os.environ.get("BQN_GPU_FUZZ_SEED", "20260821"))
    case_count = int(os.environ.get("BQN_GPU_FUZZ_CASES", "64"))
    randomizer = random.Random(seed)

    for case_index in range(case_count):
        higher_rank = randomizer.randint(0, 4)
        higher_shape = tuple(randomizer.randint(0, 5) for _ in range(higher_rank))
        lower_rank = randomizer.randint(0, higher_rank)
        lower_shape = higher_shape[:lower_rank]
        if randomizer.choice((True, False)):
            shapes: tuple[tuple[int, ...] | None, tuple[int, ...] | None] = (
                None,
                higher_shape,
            )
        elif randomizer.choice((True, False)):
            shapes = (lower_shape, higher_shape)
        else:
            shapes = (higher_shape, lower_shape)

        arguments: list[HostValue] = []
        for shape in shapes:
            if shape is None:
                arguments.append(HostValue.from_atom(randomizer.uniform(-1e6, 1e6)))
                continue
            values = [randomizer.uniform(-1e6, 1e6) for _ in range(math.prod(shape))]
            arguments.append(HostValue.from_array(values, shape))

        actual, expected = oracle_call(backend, cbqn, *arguments)
        try:
            assert_matches(actual, expected)
        except AssertionError as error:
            raise AssertionError(
                f"seed={seed} case={case_index} shapes={shapes}"
            ) from error
