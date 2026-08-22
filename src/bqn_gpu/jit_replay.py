"""Replay a captured tinygrad graph without re-deriving its input metadata.

tinygrad derives buffer references and variable bindings from the argument
tensors on every JIT invocation. That derivation rewrites the input UOp graph
and, on this backend's fixed-shape programs, costs more than executing the
kernels: it is about two thirds of a small program's measured time.

The derivation is a pure function of the argument tensors, so its result can be
reused while the very same tensors, still backed by the very same buffers, are
supplied again. Any other argument takes tinygrad's ordinary path, which also
re-establishes the cache. This mirrors what tinygrad itself does on replay and
changes no result.

The metadata helper is internal to the pinned tinygrad revision, so its absence
disables the fast path instead of breaking execution.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on the pinned tinygrad revision
    from tinygrad.engine.jit import _prepare_jit_inputs as prepare_jit_inputs
except ImportError:  # pragma: no cover - keeps execution working without it
    prepare_jit_inputs = None


class ReplayCache:
    """Cached tinygrad JIT input metadata for one compiled program."""

    __slots__ = ("_tensors", "_uops", "_buffers", "_variables")

    def __init__(self) -> None:
        self._tensors: tuple[Any, ...] | None = None
        self._uops: tuple[Any, ...] = ()
        self._buffers: Any = None
        self._variables: Any = None

    def remember(self, jitted: Any, tensors: tuple[Any, ...]) -> None:
        """Record the metadata tinygrad derived for these argument tensors."""

        self._tensors = None
        if prepare_jit_inputs is None or getattr(jitted, "captured", None) is None:
            return
        buffers, variables, _names, _info = prepare_jit_inputs(tensors, {})
        self._tensors = tensors
        self._uops = tuple(tensor.uop for tensor in tensors)
        self._buffers = buffers
        self._variables = variables

    def execute(self, jitted: Any, tensors: tuple[Any, ...]) -> Any | None:
        """Run the captured graph, or return None when the cache does not apply."""

        remembered = self._tensors
        if remembered is None or len(remembered) != len(tensors):
            return None
        for cached, supplied in zip(remembered, tensors):
            if cached is not supplied:
                return None
        for supplied, uop in zip(tensors, self._uops):
            if supplied.uop is not uop:
                return None
        captured = getattr(jitted, "captured", None)
        if captured is None:
            return None
        return captured(self._buffers, self._variables)
