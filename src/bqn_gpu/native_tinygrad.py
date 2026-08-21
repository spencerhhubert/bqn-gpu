"""Independent direct-tinygrad corpus programs used as performance baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tinygrad import Device, Tensor, TinyJit, dtypes

from .corpus import Program
from .errors import DeviceError
from .host_value import HostValue


class NativeTinygradRuntime:
    """Run generated direct tinygrad sources without the BQN frontend/backend."""

    def __init__(self, device: str = "CPU") -> None:
        requested = device.upper()
        try:
            Device[requested]
        except Exception as error:
            raise DeviceError(f"tinygrad device {requested!r} is unavailable: {error}") from error
        self.device = requested

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
            {"Tensor": Tensor, "dtypes": dtypes},
        )
        for value in arguments.values():
            value.realize()
        jitted = TinyJit(function)

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
