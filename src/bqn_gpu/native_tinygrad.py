"""Independent direct-tinygrad corpus programs used as performance baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tinygrad import Device, Tensor, TinyJit, dtypes

from .corpus import Program
from .errors import DeviceError
from .host_value import HostValue
from .tinygrad_backend import TinygradBackend


def dense_shift(x: Tensor, glyph: str, w: Tensor | float | None = None) -> Tensor:
    """Independent direct-tinygrad implementation of dense Shift."""

    if len(x.shape) == 0:
        raise ValueError("Shift requires a right argument with at least one axis")
    length = int(x.shape[0])
    if w is None:
        if length == 0:
            return x
        inserted = Tensor.zeros(
            *((1,) + tuple(int(item) for item in x.shape[1:])),
            dtype=dtypes.float64,
            device=x.device,
        )
    else:
        if not isinstance(w, Tensor):
            w = Tensor(w, dtype=dtypes.float64, device=x.device)
    if w is not None and len(w.shape) == len(x.shape) - 1:
        inserted = w.reshape((1,) + tuple(int(item) for item in w.shape))
    elif w is not None and len(w.shape) == len(x.shape):
        inserted = w
    elif w is not None:
        raise ValueError("Shift arguments have incompatible ranks")
    if tuple(inserted.shape[1:]) != tuple(x.shape[1:]):
        raise ValueError("Shift arguments have incompatible cell shapes")
    if length == 0 or int(inserted.shape[0]) == 0:
        return x
    inserted_length = int(inserted.shape[0])
    if inserted_length >= length:
        return inserted[:length] if glyph == "»" else inserted[-length:]
    if glyph == "»":
        return inserted.cat(x[: length - inserted_length], dim=0)
    return x[inserted_length:].cat(inserted, dim=0)


def major_cell_self_search(x: Tensor, glyph: str) -> Tensor:
    """Independent direct-tinygrad implementation of monadic self-search."""

    if len(x.shape) == 0:
        raise ValueError("self-search requires an array with at least one axis")
    count = int(x.shape[0])
    if count == 0:
        return x if glyph == "⍷" else Tensor.empty(0, dtype=dtypes.float64, device=x.device)
    cells = x.reshape((count, -1))
    cell_size = int(cells.shape[1])
    equal = (
        Tensor.ones(count, count, dtype=dtypes.bool, device=x.device)
        if cell_size == 0
        else (
            cells.reshape((count, 1, cell_size))
            == cells.reshape((1, count, cell_size))
        ).all(axis=2)
    )
    positions = Tensor.arange(count, device=x.device).reshape((1, count))
    if glyph == "⊐":
        first_positions = equal.where(positions, count).min(axis=1)
        own_positions = Tensor.arange(count, device=x.device)
        firsts = first_positions == own_positions
        class_numbers = firsts.cast(dtypes.float64).cumsum(axis=0) - 1
        return class_numbers[first_positions]
    rows = Tensor.arange(count, device=x.device).reshape((count, 1))
    occurrences = (equal * (positions < rows)).sum(axis=1).cast(dtypes.float64)
    firsts = occurrences == 0
    if glyph == "⍷":
        indices = [
            index for index, first in enumerate(firsts.tolist()) if bool(first)
        ]
        if indices:
            return x[Tensor(indices, dtype=dtypes.int32, device=x.device)]
        return Tensor.empty(
            *((0,) + tuple(int(length) for length in x.shape[1:])),
            dtype=dtypes.float64,
            device=x.device,
        )
    return firsts.cast(dtypes.float64) if glyph == "∊" else occurrences


class NativeTinygradRuntime:
    """Run generated direct tinygrad sources without the BQN frontend/backend."""

    def __init__(self, device: str = "CPU") -> None:
        requested = device.upper()
        try:
            Device[requested]
        except Exception as error:
            raise DeviceError(f"tinygrad device {requested!r} is unavailable: {error}") from error
        self.device = requested
        self.execution_mode = "native-eager"

    def from_host(self, value: HostValue) -> Tensor:
        tensor = Tensor(value.data, dtype=dtypes.float64, device=self.device)
        return tensor.reshape(()) if value.atom else tensor.reshape(value.shape)

    def compile(
        self,
        program: Program,
        arguments: Mapping[str, Tensor],
    ) -> Callable[[Mapping[str, Tensor]], Tensor]:
        names = ("x",) if program.arity == 1 else ("w", "x")
        if set(arguments) != set(names):
            raise ValueError(f"{program.id}: native tinygrad argument mismatch")
        function = eval(  # noqa: S307 - generated, tracked corpus source
            program.native_tinygrad,
            {
                "Tensor": Tensor,
                "dtypes": dtypes,
                "dense_shift": dense_shift,
                "major_cell_self_search": major_cell_self_search,
            },
        )
        for value in arguments.values():
            value.realize()
        if not TinygradBackend.can_compile(program.native_expression):
            self.execution_mode = "native-eager"

            def execute_eager(supplied: Mapping[str, Tensor]) -> Tensor:
                return function(*(supplied[name] for name in names))

            return execute_eager

        jitted = TinyJit(function)
        self.execution_mode = "native-jit-captured"

        def execute(supplied: Mapping[str, Tensor]) -> Tensor:
            return jitted(*(supplied[name] for name in names))

        return execute

    def realize(self, value: Tensor) -> None:
        value.realize()

    def synchronize(self) -> None:
        Device[self.device].synchronize()

    @staticmethod
    def to_host(value: Tensor, *, atom: bool) -> HostValue:
        if atom:
            return HostValue.from_atom(float(value.item()))
        data = () if value.numel() == 0 else tuple(float(item) for item in value.flatten().tolist())
        return HostValue.from_array(data, tuple(int(length) for length in value.shape))
