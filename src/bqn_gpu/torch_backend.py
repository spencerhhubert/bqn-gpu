"""PyTorch adapter for the currently supported BQN primitive surface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import torch

from .errors import DeviceError, DomainError, ShapeError, UnsupportedPrimitive
from .host_value import HostValue, Shape


@dataclass(frozen=True)
class TorchValue:
    """A dense real BQN value resident on a PyTorch device."""

    tensor: torch.Tensor
    atom: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise DomainError("TorchValue requires a torch.Tensor")
        if self.tensor.dtype != torch.float64:
            raise DomainError(f"TorchValue requires float64, got {self.tensor.dtype}")
        if self.atom and self.tensor.ndim != 0:
            raise DomainError("an atom must use a zero-dimensional tensor")

    @property
    def shape(self) -> Shape:
        return tuple(int(length) for length in self.tensor.shape)

    def to_host(self) -> HostValue:
        tensor = self.tensor.detach().cpu()
        if self.atom:
            return HostValue.from_atom(float(tensor.item()))
        data = tuple(float(value) for value in tensor.flatten().tolist())
        return HostValue.from_array(data, self.shape)


class TorchBackend:
    """Execute the supported BQN primitive surface with PyTorch."""

    def __init__(self, device: str = "CPU") -> None:
        try:
            requested = torch.device(device.lower())
        except (RuntimeError, ValueError) as error:
            raise DeviceError(f"invalid PyTorch device {device!r}: {error}") from error
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise DeviceError("PyTorch CUDA execution was requested but CUDA is unavailable")
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        self.torch_device = requested
        self.device = str(requested)

    def atom(self, value: Real) -> TorchValue:
        return self.from_host(HostValue.from_atom(value))

    def array(self, values: Iterable[Real], shape: Sequence[int]) -> TorchValue:
        return self.from_host(HostValue.from_array(values, shape))

    def from_host(self, value: HostValue) -> TorchValue:
        tensor = torch.tensor(value.data, dtype=torch.float64, device=self.torch_device)
        tensor = tensor.reshape(()) if value.atom else tensor.reshape(value.shape)
        return TorchValue(tensor=tensor, atom=value.atom)

    def call(self, glyph: str, *arguments: TorchValue) -> TorchValue:
        if len(arguments) == 1:
            return self._call_monadic(glyph, arguments[0])
        if len(arguments) == 2:
            return self._call_dyadic(glyph, arguments[0], arguments[1])
        raise UnsupportedPrimitive(
            f"primitive {glyph!r} does not have supported valence {len(arguments)}"
        )

    def _call_monadic(self, glyph: str, x: TorchValue) -> TorchValue:
        self._check_device(x)
        if glyph == "=":
            return self.atom(len(x.shape))
        if glyph == "≠":
            return self.atom(1 if len(x.shape) == 0 else x.shape[0])
        if glyph == "≢":
            return self.array(x.shape, (len(x.shape),))
        if glyph == "↕":
            if not x.atom:
                raise DomainError("Range is currently supported only for a natural atom")
            count_value = float(x.tensor.item())
            count = int(count_value)
            if count_value != count or count < 0:
                raise DomainError("Range requires a natural-number atom")
            tensor = torch.arange(
                count, dtype=torch.float64, device=self.torch_device
            )
            return TorchValue(tensor=tensor, atom=False)
        operations = {
            "+": lambda tensor: tensor,
            "-": torch.neg,
            "×": torch.sign,
            "÷": torch.reciprocal,
            "⋆": torch.exp,
            "√": torch.sqrt,
            "⌊": torch.floor,
            "⌈": torch.ceil,
            "|": torch.abs,
        }
        try:
            tensor = operations[glyph](x.tensor)
        except KeyError:
            raise UnsupportedPrimitive(
                f"monadic primitive {glyph!r} is not implemented"
            ) from None
        return TorchValue(tensor=tensor, atom=x.atom)

    def _call_dyadic(self, glyph: str, w: TorchValue, x: TorchValue) -> TorchValue:
        self._check_device(w)
        self._check_device(x)
        w_tensor, x_tensor = self._leading_axis_agreement(w, x)
        if glyph == "+":
            tensor = w_tensor + x_tensor
        elif glyph == "-":
            tensor = w_tensor - x_tensor
        elif glyph == "×":
            tensor = w_tensor * x_tensor
        elif glyph == "÷":
            tensor = w_tensor / x_tensor
        elif glyph == "⋆":
            tensor = torch.pow(w_tensor, x_tensor)
        elif glyph == "√":
            tensor = torch.pow(x_tensor, torch.reciprocal(w_tensor))
        elif glyph == "|":
            tensor = x_tensor - w_tensor * torch.floor(x_tensor / w_tensor)
        elif glyph == "⌊":
            tensor = torch.minimum(w_tensor, x_tensor)
        elif glyph == "⌈":
            tensor = torch.maximum(w_tensor, x_tensor)
        elif glyph == "=":
            tensor = torch.eq(w_tensor, x_tensor).to(torch.float64)
        elif glyph == "≠":
            tensor = torch.ne(w_tensor, x_tensor).to(torch.float64)
        elif glyph == "<":
            tensor = torch.lt(w_tensor, x_tensor).to(torch.float64)
        elif glyph == ">":
            tensor = torch.gt(w_tensor, x_tensor).to(torch.float64)
        elif glyph == "≤":
            tensor = torch.le(w_tensor, x_tensor).to(torch.float64)
        elif glyph == "≥":
            tensor = torch.ge(w_tensor, x_tensor).to(torch.float64)
        else:
            raise UnsupportedPrimitive(f"dyadic primitive {glyph!r} is not implemented")
        return TorchValue(tensor=tensor, atom=w.atom and x.atom)

    def reduce(self, glyph: str, argument: TorchValue) -> TorchValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) != 1:
            raise DomainError("BQN Fold is currently supported only for numeric lists")
        if glyph == "+":
            tensor = argument.tensor.sum()
        elif glyph == "×":
            tensor = argument.tensor.prod()
        elif glyph == "⌊":
            if argument.shape[0] == 0:
                raise DomainError("Minimum Fold of an empty list has no supported fill")
            tensor = argument.tensor.min()
        elif glyph == "⌈":
            if argument.shape[0] == 0:
                raise DomainError("Maximum Fold of an empty list has no supported fill")
            tensor = argument.tensor.max()
        else:
            raise UnsupportedPrimitive(f"Fold with {glyph!r} is not implemented")
        return TorchValue(tensor=tensor, atom=True)

    def synchronize(self) -> None:
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def _check_device(self, value: TorchValue) -> None:
        if value.tensor.device != self.torch_device:
            raise DeviceError(
                f"value is on {value.tensor.device}, backend executes on {self.torch_device}"
            )

    @staticmethod
    def _leading_axis_agreement(
        w: TorchValue, x: TorchValue
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
