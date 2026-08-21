from __future__ import annotations

import pytest

from bqn_gpu import DomainError, HostValue, ShapeError, TinygradBackend, UnsupportedPrimitive


def test_atom_and_rank_zero_array_remain_distinct(backend: TinygradBackend) -> None:
    atom = backend.atom(5)
    unit = backend.array([5], shape=())

    assert atom.tensor.shape == ()
    assert unit.tensor.shape == ()
    assert atom.to_host() == HostValue(atom=True, shape=(), data=(5.0,))
    assert unit.to_host() == HostValue(atom=False, shape=(), data=(5.0,))


def test_invalid_array_bound_is_rejected(backend: TinygradBackend) -> None:
    with pytest.raises(DomainError, match="requires 6 values"):
        backend.array([1, 2], shape=(2, 3))


@pytest.mark.parametrize("shape", [(-1,), (2, 1.5), (True,)])
def test_invalid_shape_is_rejected(backend: TinygradBackend, shape: tuple[object, ...]) -> None:
    with pytest.raises(DomainError):
        backend.array([], shape=shape)  # type: ignore[arg-type]


def test_non_numeric_values_are_rejected(backend: TinygradBackend) -> None:
    with pytest.raises(DomainError, match="real numeric"):
        backend.array(["a"], shape=(1,))  # type: ignore[list-item]


def test_unsupported_primitive_is_explicit(backend: TinygradBackend) -> None:
    with pytest.raises(UnsupportedPrimitive):
        backend.call("!", backend.atom(1), backend.atom(2))


def test_wrong_valence_is_explicit(backend: TinygradBackend) -> None:
    with pytest.raises(UnsupportedPrimitive, match="valence 0"):
        backend.call("+")


def test_shape_that_only_trailing_broadcasts_is_rejected(backend: TinygradBackend) -> None:
    w = backend.array([1, 2], shape=(2, 1))
    x = backend.array([1, 2, 3], shape=(3,))
    with pytest.raises(ShapeError, match="leading-axis"):
        backend.add(w, x)
