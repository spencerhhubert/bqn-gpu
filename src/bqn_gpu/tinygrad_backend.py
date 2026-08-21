"""tinygrad adapter for a deliberately small BQN primitive surface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

from tinygrad import Device, Tensor, dtypes

from .errors import DeviceError, DomainError, ShapeError, UnsupportedPrimitive
from .host_value import HostValue, Shape


@dataclass(frozen=True)
class TinygradValue:
    """A dense real BQN value resident on a tinygrad device."""

    tensor: Tensor
    atom: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, Tensor):
            raise DomainError("TinygradValue requires a tinygrad Tensor")
        if self.tensor.dtype != dtypes.float64:
            raise DomainError(f"TinygradValue requires float64, got {self.tensor.dtype}")
        if self.atom and len(self.tensor.shape) != 0:
            raise DomainError("an atom must use a zero-dimensional tensor")

    @property
    def shape(self) -> Shape:
        return tuple(int(length) for length in self.tensor.shape)

    def to_host(self) -> HostValue:
        if self.atom:
            return HostValue.from_atom(float(self.tensor.item()))
        if self.tensor.numel() == 0:
            data: tuple[float, ...] = ()
        else:
            data = tuple(float(value) for value in self.tensor.flatten().tolist())
        return HostValue.from_array(data, self.shape)


class TinygradBackend:
    """Execute the supported BQN primitive surface with tinygrad."""

    def __init__(self, device: str = "CPU") -> None:
        requested = device.upper()
        try:
            Device[requested]
        except Exception as error:
            raise DeviceError(f"tinygrad device {requested!r} is unavailable: {error}") from error
        self.device = requested

    def atom(self, value: Real) -> TinygradValue:
        return self.from_host(HostValue.from_atom(value))

    def array(self, values: Iterable[Real], shape: Sequence[int]) -> TinygradValue:
        return self.from_host(HostValue.from_array(values, shape))

    def from_host(self, value: HostValue) -> TinygradValue:
        tensor = Tensor(value.data, dtype=dtypes.float64, device=self.device)
        tensor = tensor.reshape(()) if value.atom else tensor.reshape(value.shape)
        return TinygradValue(tensor=tensor, atom=value.atom)

    def call(self, glyph: str, *arguments: TinygradValue) -> TinygradValue:
        if glyph != "+":
            raise UnsupportedPrimitive(f"primitive {glyph!r} is not implemented")
        if len(arguments) == 1:
            return self.conjugate(arguments[0])
        if len(arguments) == 2:
            return self.add(arguments[0], arguments[1])
        raise UnsupportedPrimitive(f"primitive '+' does not have valence {len(arguments)}")

    def conjugate(self, x: TinygradValue) -> TinygradValue:
        self._check_device(x)
        return TinygradValue(tensor=x.tensor, atom=x.atom)

    def add(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        self._check_device(w)
        self._check_device(x)
        w_tensor, x_tensor = self._leading_axis_agreement(w, x)
        return TinygradValue(tensor=w_tensor + x_tensor, atom=w.atom and x.atom)

    def _check_device(self, value: TinygradValue) -> None:
        if value.tensor.device != self.device:
            raise DeviceError(
                f"value is on {value.tensor.device}, backend executes on {self.device}"
            )

    @staticmethod
    def _leading_axis_agreement(
        w: TinygradValue, x: TinygradValue
    ) -> tuple[Tensor, Tensor]:
        if w.atom or x.atom:
            return w.tensor, x.tensor

        w_shape = w.shape
        x_shape = x.shape
        lower, higher = (
            (w_shape, x_shape) if len(w_shape) <= len(x_shape) else (x_shape, w_shape)
        )
        if higher[: len(lower)] != lower:
            raise ShapeError(
                "BQN leading-axis agreement requires the lower-rank shape "
                f"to prefix the higher-rank shape, got {w_shape} and {x_shape}"
            )

        if len(w_shape) < len(x_shape):
            return w.tensor.reshape(w_shape + (1,) * (len(x_shape) - len(w_shape))), x.tensor
        if len(x_shape) < len(w_shape):
            return w.tensor, x.tensor.reshape(x_shape + (1,) * (len(w_shape) - len(x_shape)))
        return w.tensor, x.tensor
