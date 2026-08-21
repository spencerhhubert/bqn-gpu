"""tinygrad adapter for a deliberately small BQN primitive surface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence

from tinygrad import Device, Tensor, TinyJit, dtypes

from .errors import DeviceError, DomainError, ShapeError, UnsupportedPrimitive
from .host_value import HostValue, Shape
from .ir import Expression, evaluate, has_tensor_compute


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
        if len(arguments) == 1:
            return self._call_monadic(glyph, arguments[0])
        if len(arguments) == 2:
            return self._call_dyadic(glyph, arguments[0], arguments[1])
        raise UnsupportedPrimitive(
            f"primitive {glyph!r} does not have supported valence {len(arguments)}"
        )

    def _call_monadic(self, glyph: str, x: TinygradValue) -> TinygradValue:
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
            tensor = Tensor.arange(count, device=self.device).cast(dtypes.float64)
            return TinygradValue(tensor=tensor, atom=False)
        if glyph == "+":
            tensor = x.tensor
        elif glyph == "-":
            tensor = -x.tensor
        elif glyph == "×":
            tensor = x.tensor.sign()
        elif glyph == "÷":
            tensor = 1.0 / x.tensor
        elif glyph == "⋆":
            tensor = x.tensor.exp()
        elif glyph == "√":
            tensor = x.tensor.sqrt()
        elif glyph == "⌊":
            tensor = x.tensor.floor()
        elif glyph == "⌈":
            tensor = x.tensor.ceil()
        elif glyph == "|":
            tensor = x.tensor.abs()
        else:
            raise UnsupportedPrimitive(f"monadic primitive {glyph!r} is not implemented")
        return TinygradValue(tensor=tensor, atom=x.atom)

    def _call_dyadic(
        self, glyph: str, w: TinygradValue, x: TinygradValue
    ) -> TinygradValue:
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
            tensor = w_tensor**x_tensor
        elif glyph == "√":
            tensor = x_tensor ** (1.0 / w_tensor)
        elif glyph == "|":
            tensor = x_tensor - w_tensor * (x_tensor / w_tensor).floor()
        elif glyph == "⌊":
            tensor = w_tensor.minimum(x_tensor)
        elif glyph == "⌈":
            tensor = w_tensor.maximum(x_tensor)
        elif glyph == "=":
            tensor = (w_tensor == x_tensor).cast(dtypes.float64)
        elif glyph == "≠":
            tensor = (w_tensor != x_tensor).cast(dtypes.float64)
        elif glyph == "<":
            tensor = (w_tensor < x_tensor).cast(dtypes.float64)
        elif glyph == ">":
            tensor = (w_tensor > x_tensor).cast(dtypes.float64)
        elif glyph == "≤":
            tensor = (w_tensor <= x_tensor).cast(dtypes.float64)
        elif glyph == "≥":
            tensor = (w_tensor >= x_tensor).cast(dtypes.float64)
        else:
            raise UnsupportedPrimitive(f"dyadic primitive {glyph!r} is not implemented")
        return TinygradValue(tensor=tensor, atom=w.atom and x.atom)

    def conjugate(self, x: TinygradValue) -> TinygradValue:
        return self._call_monadic("+", x)

    def add(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        return self._call_dyadic("+", w, x)

    def reduce(self, glyph: str, argument: TinygradValue) -> TinygradValue:
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
        return TinygradValue(tensor=tensor, atom=True)

    def synchronize(self) -> None:
        Device[self.device].synchronize()

    def compile(
        self,
        expression: Expression,
        arguments: Mapping[str, TinygradValue],
    ) -> Callable[[Mapping[str, TinygradValue]], TinygradValue]:
        """Capture a reusable tinygrad graph for one source/shape signature."""

        names = tuple(sorted(arguments))
        atom_kinds = tuple(arguments[name].atom for name in names)
        output_atom: list[bool] = []
        for argument in arguments.values():
            argument.tensor.realize()

        def execute_tensors(*tensors: Tensor) -> Tensor:
            values = {
                name: TinygradValue(tensor=tensor, atom=atom)
                for name, tensor, atom in zip(names, tensors, atom_kinds, strict=True)
            }
            result = evaluate(expression, self, values)
            if not output_atom:
                output_atom.append(result.atom)
            return result.tensor

        jitted = TinyJit(execute_tensors)

        def execute_compiled(
            supplied: Mapping[str, TinygradValue],
        ) -> TinygradValue:
            if tuple(sorted(supplied)) != names:
                raise DomainError(
                    f"compiled program expected arguments {names}, got {tuple(sorted(supplied))}"
                )
            tensor = jitted(*(supplied[name].tensor for name in names))
            return TinygradValue(tensor=tensor, atom=output_atom[0])

        return execute_compiled

    @staticmethod
    def can_compile(expression: Expression) -> bool:
        """Whether the expression has fixed shape and launches tensor work."""

        return TinygradBackend._fixed_output_shape(expression) and has_tensor_compute(
            expression
        )

    @staticmethod
    def _fixed_output_shape(expression: Expression) -> bool:
        """Whether output shape is fixed by the input tensor signatures."""

        operation = expression["op"]
        if operation == "call":
            if expression["glyph"] == "↕":
                return False
            return all(
                TinygradBackend._fixed_output_shape(child)
                for child in expression["arguments"]
            )
        if operation == "fold":
            return TinygradBackend._fixed_output_shape(expression["argument"])
        return True

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
