"""tinygrad adapter for a deliberately small BQN primitive surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence

from tinygrad import Device, Tensor, TinyJit, dtypes

from .errors import DeviceError, DomainError, ShapeError, UnsupportedPrimitive
from .host_value import HostValue, Shape
from .ir import (
    Expression,
    evaluate,
    expand_combinator,
    expand_repeat,
    expand_train,
    has_tensor_compute,
)
from .mapping import plan_mapping
from .optimizer import OptimizationResult, optimize


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


@dataclass(frozen=True)
class ExecutionPlan:
    """An explainable dispatch choice for one specialized expression."""

    mode: str
    reason: str


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

    def call_scalar(
        self,
        glyph: str,
        scalar: Real,
        scalar_left: bool,
        argument: TinygradValue,
    ) -> TinygradValue:
        self._check_device(argument)
        value = float(scalar)
        left, right = (
            (value, argument.tensor) if scalar_left else (argument.tensor, value)
        )
        operations = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "×": lambda: left * right,
            "÷": lambda: left / right,
        }
        try:
            tensor = operations[glyph]()
        except KeyError:
            raise UnsupportedPrimitive(
                f"literal-scalar primitive {glyph!r} is not implemented"
            ) from None
        return TinygradValue(tensor=tensor, atom=argument.atom)

    def call_static(
        self,
        glyph: str,
        left_values: Sequence[int],
        left_atom: bool,
        argument: TinygradValue,
    ) -> TinygradValue:
        values = tuple(int(value) for value in left_values)
        if glyph == "↑":
            return self._take_or_drop_counts(values, argument, take=True)
        if glyph == "↓":
            return self._take_or_drop_counts(values, argument, take=False)
        if glyph == "⌽":
            return self._rotate_counts(values, argument)
        if glyph == "/":
            return self._replicate_counts(values, left_atom, argument)
        if glyph == "↕":
            return self._windows_sizes(values, argument)
        raise UnsupportedPrimitive(f"static primitive {glyph!r} is not implemented")

    def _call_monadic(self, glyph: str, x: TinygradValue) -> TinygradValue:
        self._check_device(x)
        if glyph == "⋆⁼":
            return TinygradValue(tensor=x.tensor.log(), atom=x.atom)
        if glyph in {"»", "«"}:
            return self._shift(glyph, None, x)
        if glyph in {"∧", "∨"}:
            if len(x.shape) != 1:
                raise DomainError("Sort is currently supported for numeric lists")
            tensor = x.tensor.sort(dim=0, descending=glyph == "∨")[0]
            return TinygradValue(tensor=tensor, atom=False)
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
        if glyph == "≡":
            return self.atom(0 if x.atom else 1)
        if glyph in {"⊣", "⊢"}:
            return x
        if glyph == "⥊":
            return TinygradValue(tensor=x.tensor.reshape((x.tensor.numel(),)), atom=False)
        if glyph == "≍":
            return TinygradValue(tensor=x.tensor.reshape((1,) + x.shape), atom=False)
        if glyph == "⌽":
            if len(x.shape) == 0:
                raise DomainError("Reverse requires an array with at least one axis")
            return TinygradValue(tensor=x.tensor.flip(0), atom=False)
        if glyph == "⍉":
            if x.atom:
                return TinygradValue(tensor=x.tensor, atom=False)
            if len(x.shape) <= 1:
                return x
            axes = tuple(range(1, len(x.shape))) + (0,)
            return TinygradValue(tensor=x.tensor.permute(axes), atom=False)
        if glyph == "/":
            counts = self._whole_numbers(x, "Indices", natural=True)
            indices = [index for index, count in enumerate(counts) for _ in range(count)]
            return self._index_list(indices)
        if glyph in {"⍋", "⍒"}:
            if len(x.shape) != 1:
                raise DomainError("Grade is currently supported for numeric lists")
            tensor = x.tensor.argsort(dim=0, descending=glyph == "⍒").cast(dtypes.float64)
            return TinygradValue(tensor=tensor, atom=False)
        if glyph == "⊏":
            if len(x.shape) == 0 or x.shape[0] == 0:
                raise DomainError("First Cell requires a nonempty array with an axis")
            return TinygradValue(tensor=x.tensor[0], atom=False)
        if glyph == "⊑":
            if x.tensor.numel() == 0:
                raise DomainError("First requires a nonempty value")
            return TinygradValue(tensor=x.tensor.reshape((-1,))[0], atom=True)
        if glyph in {"⊐", "⊒", "∊", "⍷"}:
            return self._self_search(glyph, x)
        if glyph == "⋈":
            if not x.atom:
                raise DomainError("dense Enlist is currently supported only for atoms")
            return TinygradValue(tensor=x.tensor.reshape((1,)), atom=False)
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
        elif glyph == "¬":
            tensor = 1.0 - x.tensor
        else:
            raise UnsupportedPrimitive(f"monadic primitive {glyph!r} is not implemented")
        return TinygradValue(tensor=tensor, atom=x.atom)

    def _call_dyadic(
        self, glyph: str, w: TinygradValue, x: TinygradValue
    ) -> TinygradValue:
        self._check_device(w)
        self._check_device(x)
        if glyph in {"≡", "≢"}:
            matches = w.atom == x.atom and w.shape == x.shape
            if not matches:
                return self.atom(1 if glyph == "≢" else 0)
            equal = (w.tensor == x.tensor).all().cast(dtypes.float64)
            tensor = 1.0 - equal if glyph == "≢" else equal
            return TinygradValue(tensor=tensor, atom=True)
        if glyph == "⊣":
            return w
        if glyph == "⊢":
            return x
        if glyph == "⥊":
            return self._reshape(w, x)
        if glyph == "∾":
            return self._join_to(w, x)
        if glyph == "≍":
            return self._couple(w, x)
        if glyph == "⋈":
            if not w.atom or not x.atom:
                raise DomainError("dense Pair is currently supported only for atoms")
            return TinygradValue(tensor=w.tensor.stack(x.tensor, dim=0), atom=False)
        if glyph == "↑":
            return self._take_or_drop(w, x, take=True)
        if glyph == "↓":
            return self._take_or_drop(w, x, take=False)
        if glyph == "⌽":
            return self._rotate(w, x)
        if glyph == "⍉":
            return self._reorder_axes(w, x)
        if glyph == "/":
            return self._replicate(w, x)
        if glyph == "↕":
            return self._windows(w, x)
        if glyph in {"»", "«"}:
            return self._shift(glyph, w, x)
        if glyph == "⊏":
            return self._select(w, x)
        if glyph == "⊑":
            return self._pick(w, x)
        if glyph in {"⍋", "⍒"}:
            return self._bins(glyph, w, x)
        if glyph in {"⊐", "⊒", "∊", "⍷"}:
            return self._search(glyph, w, x)
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
        elif glyph == "∧":
            tensor = w_tensor * x_tensor
        elif glyph == "∨":
            tensor = w_tensor + x_tensor - w_tensor * x_tensor
        elif glyph == "¬":
            tensor = 1.0 + w_tensor - x_tensor
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
        tensor = self._reduce_tensor(glyph, argument.tensor, axis=0)
        return TinygradValue(tensor=tensor, atom=True)

    def insert(self, glyph: str, argument: TinygradValue) -> TinygradValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) == 0:
            raise DomainError("Insert requires an array with at least one axis")
        tensor = self._reduce_tensor(glyph, argument.tensor, axis=0)
        return TinygradValue(
            tensor=tensor,
            atom=len(argument.shape) == 1 and argument.shape[0] != 0,
        )

    def scan(self, glyph: str, argument: TinygradValue) -> TinygradValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) == 0:
            raise DomainError("Scan requires an array with at least one axis")
        if argument.shape[0] == 0:
            return argument
        if glyph == "+":
            tensor = argument.tensor.cumsum(axis=0)
        elif glyph in {"×", "∧"}:
            tensor = argument.tensor.cumprod(axis=0)
        elif glyph == "∨":
            tensor = 1.0 - (1.0 - argument.tensor).cumprod(axis=0)
        elif glyph == "⌊":
            tensor = -(-argument.tensor).cummax(axis=0)[0]
        elif glyph == "⌈":
            tensor = argument.tensor.cummax(axis=0)[0]
        else:
            raise UnsupportedPrimitive(f"Scan with {glyph!r} is not implemented")
        return TinygradValue(tensor=tensor, atom=False)

    def map_function(
        self,
        modifier: str,
        rank_specification: Sequence[float],
        operand: object,
        arguments: Sequence[TinygradValue],
        function: Callable[[Sequence[TinygradValue]], TinygradValue],
    ) -> TinygradValue:
        for argument in arguments:
            self._check_device(argument)
        primitive = (
            operand.get("glyph")
            if isinstance(operand, dict) and operand.get("kind") == "primitive"
            else None
        )
        monadic_pervasive = {"+", "-", "×", "÷", "⋆", "√", "⌊", "⌈", "|", "¬"}
        dyadic_pervasive = {
            "+", "-", "×", "÷", "⋆", "√", "⌊", "⌈", "|", "¬",
            "=", "≠", "<", ">", "≤", "≥", "∧", "∨",
        }
        if len(arguments) == 1 and primitive in monadic_pervasive:
            result = self.call(primitive, arguments[0])
            return TinygradValue(tensor=result.tensor, atom=False)
        if modifier == "¨" and len(arguments) == 2 and primitive in dyadic_pervasive:
            result = self.call(primitive, *arguments)
            return TinygradValue(tensor=result.tensor, atom=False)
        if modifier == "⌜" and len(arguments) == 2 and primitive in dyadic_pervasive:
            left, right = arguments
            output_shape = left.shape + right.shape
            left_tensor = left.tensor.reshape(left.shape + (1,) * len(right.shape)).expand(output_shape)
            right_tensor = right.tensor.reshape((1,) * len(left.shape) + right.shape).expand(output_shape)
            result = self.call(
                primitive,
                TinygradValue(left_tensor, atom=False),
                TinygradValue(right_tensor, atom=False),
            )
            return TinygradValue(tensor=result.tensor, atom=False)

        plan = plan_mapping(
            modifier,
            [argument.shape for argument in arguments],
            rank_specification,
        )
        if (
            modifier == "⎉"
            and len(arguments) == 2
            and plan.cell_ranks == (1, 1)
            and self._is_sum_product_operand(operand)
        ):
            left, right = arguments
            left_cell = left.shape[-1:]
            right_cell = right.shape[-1:]
            if left_cell != right_cell:
                raise ShapeError(
                    "ranked sum-product requires equal vector cell shapes, got "
                    f"{left_cell} and {right_cell}"
                )
            output_shape = plan.frame_shape + left_cell
            tensors = []
            for argument, frame in zip(arguments, plan.argument_frames, strict=True):
                shape = frame + (1,) * (len(plan.frame_shape) - len(frame)) + left_cell
                tensors.append(argument.tensor.reshape(shape).expand(output_shape))
            return TinygradValue(
                tensor=(tensors[0] * tensors[1]).sum(axis=len(plan.frame_shape)),
                atom=False,
            )
        if any(length == 0 for length in plan.frame_shape):
            raise DomainError("mapping over an empty frame is not implemented yet")
        results: list[TinygradValue] = []
        for indices in plan.indices():
            cells = []
            for argument, index, cell_rank in zip(
                arguments, indices, plan.cell_ranks, strict=True
            ):
                tensor = argument.tensor if not index else argument.tensor[index]
                cells.append(
                    TinygradValue(
                        tensor=tensor,
                        atom=(
                            argument.atom
                            if not index and argument.atom
                            else modifier in {"¨", "⌜"} and cell_rank == 0
                        ),
                    )
                )
            results.append(function(cells))
        return self._combine_mapped_results(plan.frame_shape, results)

    @staticmethod
    def _is_sum_product_operand(operand: object) -> bool:
        if not isinstance(operand, dict):
            return False
        left = operand.get("left")
        right = operand.get("right")
        return (
            operand.get("kind") == "modifier"
            and operand.get("modifier") == "∘"
            and isinstance(left, dict)
            and left.get("kind") == "fold"
            and left.get("modifier") == "˝"
            and left.get("glyph") == "+"
            and isinstance(right, dict)
            and right.get("kind") == "primitive"
            and right.get("glyph") == "×"
        )

    def _combine_mapped_results(
        self,
        frame_shape: Shape,
        results: Sequence[TinygradValue],
    ) -> TinygradValue:
        if not results:
            raise DomainError("mapping produced no result cells")
        result_shape = results[0].shape
        if any(result.shape != result_shape for result in results[1:]):
            raise ShapeError("mapped function returned incompatible dense result shapes")
        if not frame_shape:
            return TinygradValue(
                tensor=results[0].tensor.reshape(result_shape),
                atom=False,
            )
        if len(results) == 1:
            tensor = results[0].tensor.reshape((1,) + result_shape)
        else:
            tensor = results[0].tensor.stack(
                *(result.tensor for result in results[1:]),
                dim=0,
            )
        return TinygradValue(tensor=tensor.reshape(frame_shape + result_shape), atom=False)

    def _reduce_tensor(self, glyph: str, tensor: Tensor, *, axis: int) -> Tensor:
        if tensor.shape[axis] == 0:
            identities = {"+": 0.0, "×": 1.0, "∧": 1.0, "∨": 0.0, "⌊": math.inf, "⌈": -math.inf}
            try:
                identity = identities[glyph]
            except KeyError:
                raise UnsupportedPrimitive(f"reduction with {glyph!r} is not implemented") from None
            shape = tuple(length for index, length in enumerate(tensor.shape) if index != axis)
            return Tensor.full(shape, identity, dtype=dtypes.float64, device=self.device)
        if glyph == "+":
            return tensor.sum(axis=axis)
        if glyph in {"×", "∧"}:
            return tensor.prod(axis=axis)
        if glyph == "∨":
            return 1.0 - (1.0 - tensor).prod(axis=axis)
        if glyph == "⌊":
            return tensor.min(axis=axis)
        if glyph == "⌈":
            return tensor.max(axis=axis)
        raise UnsupportedPrimitive(f"reduction with {glyph!r} is not implemented")

    def _reshape(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        shape = self._whole_numbers(w, "Reshape", natural=True)
        if len(w.shape) > 1:
            raise DomainError("Reshape requires a natural-number atom or list")
        target_size = math.prod(shape)
        source = x.tensor.reshape((x.tensor.numel(),))
        if target_size == 0:
            tensor = Tensor([], dtype=dtypes.float64, device=self.device).reshape(shape)
        elif source.numel() == 0:
            raise DomainError("Reshape cannot fill a nonempty result from an empty array")
        else:
            repetitions = math.ceil(target_size / source.numel())
            tensor = source.repeat((repetitions,))[:target_size].reshape(shape)
        return TinygradValue(tensor=tensor, atom=False)

    def _join_to(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        result_rank = max(1, len(w.shape), len(x.shape))
        if result_rank - len(w.shape) > 1 or result_rank - len(x.shape) > 1:
            raise ShapeError("Join To arguments may differ in rank by at most one")
        w_tensor = w.tensor.reshape((1,) + w.shape) if len(w.shape) < result_rank else w.tensor
        x_tensor = x.tensor.reshape((1,) + x.shape) if len(x.shape) < result_rank else x.tensor
        if tuple(w_tensor.shape[1:]) != tuple(x_tensor.shape[1:]):
            raise ShapeError("Join To requires matching trailing cell shapes")
        return TinygradValue(tensor=w_tensor.cat(x_tensor, dim=0), atom=False)

    def _shift(
        self,
        glyph: str,
        w: TinygradValue | None,
        x: TinygradValue,
    ) -> TinygradValue:
        if len(x.shape) == 0:
            raise DomainError("Shift requires a right argument with at least one axis")
        length = x.shape[0]
        if w is None:
            if length == 0:
                return x
            inserted = Tensor.zeros(
                *((1,) + x.shape[1:]),
                dtype=dtypes.float64,
                device=self.device,
            )
        elif len(w.shape) == len(x.shape) - 1:
            inserted = w.tensor.reshape((1,) + w.shape)
        elif len(w.shape) == len(x.shape):
            inserted = w.tensor
        else:
            raise DomainError(
                "Shift left argument rank must equal the right rank or be one less"
            )
        if tuple(inserted.shape[1:]) != x.shape[1:]:
            raise ShapeError("Shift requires matching trailing cell shapes")
        if length == 0:
            return x
        inserted_length = int(inserted.shape[0])
        if inserted_length == 0:
            return x
        if inserted_length >= length:
            output = inserted[:length] if glyph == "»" else inserted[-length:]
        elif glyph == "»":
            output = inserted.cat(x.tensor[: length - inserted_length], dim=0)
        else:
            output = x.tensor[inserted_length:].cat(inserted, dim=0)
        return TinygradValue(tensor=output, atom=False)

    def _couple(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        if w.atom != x.atom or w.shape != x.shape:
            raise DomainError("dense Couple requires arguments with the same kind and shape")
        return TinygradValue(tensor=w.tensor.stack(x.tensor, dim=0), atom=False)

    def _take_or_drop(
        self, w: TinygradValue, x: TinygradValue, *, take: bool
    ) -> TinygradValue:
        counts = self._whole_numbers(w, "Take" if take else "Drop")
        return self._take_or_drop_counts(counts, x, take=take)

    def _take_or_drop_counts(
        self, counts: Sequence[int], x: TinygradValue, *, take: bool
    ) -> TinygradValue:
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        shape = tuple(int(length) for length in tensor.shape)
        if len(counts) > len(shape):
            raise DomainError("dense Take/Drop does not yet add leading unit axes")
        slices: list[slice] = [slice(None)] * len(shape)
        for axis, count in enumerate(counts):
            length = shape[axis]
            if take:
                if abs(count) > length:
                    raise DomainError("fill-expanding Take is outside the dense tier")
                slices[axis] = slice(0, count) if count >= 0 else slice(length + count, length)
            else:
                removed = min(abs(count), length)
                slices[axis] = slice(removed, None) if count >= 0 else slice(0, length - removed)
        return TinygradValue(tensor=tensor[tuple(slices)], atom=False)

    def _rotate(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        rotations = self._whole_numbers(w, "Rotate")
        return self._rotate_counts(rotations, x)

    def _rotate_counts(
        self, rotations: Sequence[int], x: TinygradValue
    ) -> TinygradValue:
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        if len(rotations) > len(tensor.shape):
            raise DomainError("Rotate specifies more axes than the right argument")
        if not rotations:
            return TinygradValue(tensor=tensor, atom=False)
        shifts = tuple(-rotation for rotation in rotations)
        return TinygradValue(
            tensor=tensor.roll(shifts, dims=tuple(range(len(rotations)))), atom=False
        )

    def _reorder_axes(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        destinations = list(self._whole_numbers(w, "Reorder Axes", natural=True))
        tensor = x.tensor
        rank = len(x.shape)
        if len(destinations) > rank:
            raise DomainError("Reorder Axes specifies more axes than the right argument")
        unused = [axis for axis in range(rank) if axis not in destinations]
        destinations.extend(unused[: rank - len(destinations)])
        if sorted(destinations) != list(range(rank)):
            raise DomainError("dense Reorder Axes currently requires a permutation")
        if rank == 0:
            return TinygradValue(tensor=tensor, atom=False)
        axes = tuple(sorted(range(rank), key=destinations.__getitem__))
        return TinygradValue(tensor=tensor.permute(axes), atom=False)

    def _replicate(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        counts = self._whole_numbers(w, "Replicate", natural=True)
        return self._replicate_counts(counts, w.atom, x)

    def _replicate_counts(
        self,
        counts: Sequence[int],
        left_atom: bool,
        x: TinygradValue,
    ) -> TinygradValue:
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        if len(tensor.shape) == 0:
            raise DomainError("Replicate requires an array axis")
        counts = tuple(counts)
        if any(count < 0 for count in counts):
            raise DomainError("Replicate requires natural numbers")
        if left_atom:
            counts = counts * int(tensor.shape[0])
        if len(counts) != tensor.shape[0]:
            raise ShapeError("Replicate counts must match the first-axis length")
        indices = [index for index, count in enumerate(counts) for _ in range(count)]
        if not indices:
            shape = (0,) + tuple(int(length) for length in tensor.shape[1:])
            output = Tensor.empty(*shape, dtype=dtypes.float64, device=self.device)
        else:
            output = tensor[self._integer_indices(indices)]
        return TinygradValue(tensor=output, atom=False)

    def _windows(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        sizes = self._whole_numbers(w, "Windows", natural=True)
        return self._windows_sizes(sizes, x)

    def _windows_sizes(
        self, sizes: Sequence[int], x: TinygradValue
    ) -> TinygradValue:
        sizes = tuple(sizes)
        if any(size < 0 for size in sizes):
            raise DomainError("Windows requires natural numbers")
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        rank = len(tensor.shape)
        if len(sizes) > rank:
            raise DomainError("Windows specifies more axes than the right argument")
        if any(size == 0 for size in sizes):
            raise DomainError("zero-sized Windows are not yet in the dense tier")
        output_lengths = tuple(max(0, int(tensor.shape[axis]) - size + 1) for axis, size in enumerate(sizes))
        if any(length == 0 for length in output_lengths):
            shape = output_lengths + sizes + tuple(int(length) for length in tensor.shape[len(sizes):])
            output = Tensor.empty(*shape, dtype=dtypes.float64, device=self.device)
            return TinygradValue(tensor=output, atom=False)
        output = tensor
        for axis, size in enumerate(sizes):
            output = output.unfold(axis, size, 1)
        if sizes:
            axes = (
                tuple(range(len(sizes)))
                + tuple(range(rank, rank + len(sizes)))
                + tuple(range(len(sizes), rank))
            )
            output = output.permute(axes)
        return TinygradValue(tensor=output, atom=False)

    def _select(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        if x.atom or len(x.shape) == 0:
            raise DomainError("Select requires a right argument with a first axis")
        indices = self._whole_numbers(w, "Select", natural=True)
        if any(index >= x.shape[0] for index in indices):
            raise DomainError("Select index is outside the first axis")
        if w.atom:
            output = x.tensor[indices[0]]
        elif len(w.shape) <= 1:
            if not indices:
                shape = w.shape + x.shape[1:]
                output = Tensor.empty(*shape, dtype=dtypes.float64, device=self.device)
            else:
                output = x.tensor[self._integer_indices(indices)].reshape(w.shape + x.shape[1:])
        else:
            raise DomainError("dense Select currently accepts one index array")
        return TinygradValue(tensor=output, atom=False)

    def _pick(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        if x.atom:
            raise DomainError("Pick requires an array right argument")
        indices = self._whole_numbers(w, "Pick", natural=True)
        if not indices:
            return x
        if len(indices) > len(x.shape):
            raise DomainError("Pick index has higher rank than the right argument")
        if any(index >= x.shape[axis] for axis, index in enumerate(indices)):
            raise DomainError("Pick index is outside the right argument")
        output = x.tensor[tuple(indices)]
        return TinygradValue(tensor=output, atom=len(indices) == len(x.shape))

    def _bins(
        self, glyph: str, w: TinygradValue, x: TinygradValue
    ) -> TinygradValue:
        if len(w.shape) != 1:
            raise DomainError("Bins currently requires a numeric list left argument")
        principal = w.tensor.reshape((1, w.shape[0]))
        queries = x.tensor.reshape((x.tensor.numel(), 1))
        ordered = principal <= queries if glyph == "⍋" else principal >= queries
        output = ordered.sum(axis=1).cast(dtypes.float64).reshape(x.shape)
        return TinygradValue(tensor=output, atom=x.atom)

    def _self_search(self, glyph: str, x: TinygradValue) -> TinygradValue:
        if len(x.shape) == 0:
            raise DomainError("self-search requires an array with at least one axis")
        count = x.shape[0]
        if count == 0:
            if glyph == "⍷":
                return x
            return self.array((), (0,))
        cells = x.tensor.reshape((count, -1))
        cell_size = cells.shape[1]
        if cell_size == 0:
            equal = Tensor.ones(
                count,
                count,
                dtype=dtypes.bool,
                device=self.device,
            )
        else:
            equal = (
                cells.reshape((count, 1, cell_size))
                == cells.reshape((1, count, cell_size))
            ).all(axis=2)
        positions = Tensor.arange(count, device=self.device).reshape((1, count))
        if glyph == "⊐":
            first_positions = equal.where(positions, count).min(axis=1)
            own_positions = Tensor.arange(count, device=self.device)
            firsts = first_positions == own_positions
            class_numbers = firsts.cast(dtypes.float64).cumsum(axis=0) - 1
            output = class_numbers[first_positions]
        else:
            rows = Tensor.arange(count, device=self.device).reshape((count, 1))
            earlier = positions < rows
            occurrences = (equal * earlier).sum(axis=1).cast(dtypes.float64)
            firsts = occurrences == 0
            if glyph == "⍷":
                indices = [
                    index
                    for index, first in enumerate(firsts.tolist())
                    if bool(first)
                ]
                output_shape = (len(indices),) + x.shape[1:]
                output = (
                    x.tensor[self._integer_indices(indices)]
                    if indices
                    else Tensor.empty(
                        *output_shape,
                        dtype=dtypes.float64,
                        device=self.device,
                    )
                )
                return TinygradValue(tensor=output, atom=False)
            output = firsts.cast(dtypes.float64) if glyph == "∊" else occurrences
        return TinygradValue(tensor=output, atom=False)

    def _search(
        self, glyph: str, w: TinygradValue, x: TinygradValue
    ) -> TinygradValue:
        if glyph == "⍷":
            return self._find(w, x)
        if glyph == "∊":
            principal, queries, result = x, w, w
        else:
            principal, queries, result = w, x, x
        if len(principal.shape) != 1:
            raise DomainError("dense search currently requires a numeric list principal argument")
        principal_values = principal.tensor.reshape((principal.shape[0],))
        query_values = queries.tensor.reshape((queries.tensor.numel(),))
        if glyph == "⊒":
            principal_host = tuple(float(item) for item in principal_values.tolist())
            query_host = tuple(float(item) for item in query_values.tolist())
            locations: dict[float, list[int]] = {}
            for index, value in enumerate(principal_host):
                locations.setdefault(value, []).append(index)
            used: dict[float, int] = {}
            output_host = []
            for value in query_host:
                occurrence = used.get(value, 0)
                matches = locations.get(value, [])
                output_host.append(matches[occurrence] if occurrence < len(matches) else len(principal_host))
                used[value] = occurrence + 1
            output = Tensor(output_host, dtype=dtypes.float64, device=self.device).reshape(result.shape)
            return TinygradValue(tensor=output, atom=result.atom)
        matches = query_values.reshape((-1, 1)) == principal_values.reshape((1, -1))
        if glyph == "∊":
            output = matches.any(axis=1).cast(dtypes.float64)
        elif principal.shape[0] == 0:
            output = Tensor.full(
                (query_values.numel(),),
                0.0,
                dtype=dtypes.float64,
                device=self.device,
            )
        else:
            positions = Tensor.arange(principal.shape[0], device=self.device).reshape((1, -1))
            output = matches.where(positions, principal.shape[0]).min(axis=1).cast(dtypes.float64)
        return TinygradValue(tensor=output.reshape(result.shape), atom=result.atom)

    def _find(self, w: TinygradValue, x: TinygradValue) -> TinygradValue:
        if len(w.shape) != 1 or len(x.shape) != 1:
            raise DomainError("Find is currently supported for numeric lists")
        pattern_length, input_length = w.shape[0], x.shape[0]
        if pattern_length == 0:
            return TinygradValue(tensor=Tensor.ones(input_length + 1, dtype=dtypes.float64, device=self.device), atom=False)
        if pattern_length > input_length:
            return TinygradValue(tensor=Tensor.empty(0, dtype=dtypes.float64, device=self.device), atom=False)
        windows = x.tensor.unfold(0, pattern_length, 1)
        output = (windows == w.tensor.reshape((1, pattern_length))).all(axis=1).cast(dtypes.float64)
        return TinygradValue(tensor=output, atom=False)

    def _integer_indices(self, values: Sequence[int]) -> Tensor:
        return Tensor(values, dtype=dtypes.int32, device=self.device)

    def _index_list(self, values: Sequence[int]) -> TinygradValue:
        tensor = Tensor(values, dtype=dtypes.float64, device=self.device)
        return TinygradValue(tensor=tensor, atom=False)

    @staticmethod
    def _whole_numbers(
        value: TinygradValue, operation: str, *, natural: bool = False
    ) -> tuple[int, ...]:
        if value.atom:
            numbers = (float(value.tensor.item()),)
        elif len(value.shape) <= 1:
            numbers = tuple(float(item) for item in value.tensor.flatten().tolist())
        else:
            raise DomainError(f"{operation} requires an atom or list")
        result: list[int] = []
        for number in numbers:
            integer = int(number)
            if number != integer or (natural and integer < 0):
                qualifier = "natural numbers" if natural else "whole numbers"
                raise DomainError(f"{operation} requires {qualifier}")
            result.append(integer)
        return tuple(result)

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
        optimization = self.optimize(expression, arguments)
        optimized_expression = optimization.expression
        plan = self.execution_plan(optimized_expression)
        output_atom: list[bool] = []
        for argument in arguments.values():
            argument.tensor.realize()

        def execute_tensors(*tensors: Tensor) -> Tensor:
            values = {
                name: TinygradValue(tensor=tensor, atom=atom)
                for name, tensor, atom in zip(names, tensors, atom_kinds, strict=True)
            }
            result = evaluate(optimized_expression, self, values)
            if not output_atom:
                output_atom.append(result.atom)
            return result.tensor

        if plan.mode == "optimized-noop":
            def execute_simplified(
                supplied: Mapping[str, TinygradValue],
            ) -> TinygradValue:
                return evaluate(optimized_expression, self, supplied)

            execute_simplified.execution_mode = "optimized-noop"  # type: ignore[attr-defined]
            execute_simplified.execution_reason = plan.reason  # type: ignore[attr-defined]
            return execute_simplified

        if plan.mode == "specialized-eager":
            def execute_specialized(
                supplied: Mapping[str, TinygradValue],
            ) -> TinygradValue:
                if tuple(sorted(supplied)) != names:
                    raise DomainError(
                        f"specialized program expected arguments {names}, "
                        f"got {tuple(sorted(supplied))}"
                    )
                return evaluate(optimized_expression, self, supplied)

            execute_specialized.execution_mode = "specialized-eager"  # type: ignore[attr-defined]
            execute_specialized.execution_reason = plan.reason  # type: ignore[attr-defined]
            return execute_specialized

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

        execute_compiled.execution_mode = "jit-captured"  # type: ignore[attr-defined]
        execute_compiled.execution_reason = plan.reason  # type: ignore[attr-defined]
        return execute_compiled

    @staticmethod
    def execution_plan(expression: Expression) -> ExecutionPlan:
        """Choose the least-overhead execution strategy for specialized IR."""

        if not has_tensor_compute(expression):
            return ExecutionPlan(
                mode="optimized-noop",
                reason="specialization-erased-all-data-dependent-work",
            )
        if TinygradBackend._is_layout_only(expression):
            return ExecutionPlan(
                mode="specialized-eager",
                reason="layout-only-expression-is-cheaper-without-jit-replay",
            )
        if (
            expression["op"] == "map"
            and expression["modifier"] == "⎉"
            and TinygradBackend._is_sum_product_operand(expression["function"])
        ):
            return ExecutionPlan(
                mode="jit-captured",
                reason="ranked-sum-product-lowers-to-one-batched-reduction",
            )
        if expression["op"] == "map":
            return ExecutionPlan(
                mode="jit-captured",
                reason="uniform-dense-mapping-captured-as-fixed-shape-graph",
            )
        return ExecutionPlan(
            mode="jit-captured",
            reason="fixed-shape-tensor-compute-benefits-from-graph-replay",
        )

    @staticmethod
    def _is_layout_only(expression: Expression) -> bool:
        """Whether evaluation only constructs dense views of existing values."""

        operation = expression["op"]
        if operation in {"argument", "constant", "array"}:
            return True
        if operation != "call":
            return False
        children = expression["arguments"]
        return (
            len(children) == 1
            and expression["glyph"] in {"⥊", "≍", "⌽", "⍉", "⊣", "⊢"}
            and TinygradBackend._is_layout_only(children[0])
        )

    @staticmethod
    def optimize(
        expression: Expression,
        arguments: Mapping[str, TinygradValue],
    ) -> OptimizationResult:
        return optimize(
            expression,
            {name: len(value.shape) for name, value in arguments.items()},
        )

    @staticmethod
    def can_compile(
        expression: Expression,
        arguments: Mapping[str, TinygradValue] | None = None,
    ) -> bool:
        """Whether the expression has fixed shape and launches tensor work."""

        candidate = (
            TinygradBackend.optimize(expression, arguments).expression
            if arguments is not None
            else expression
        )
        return TinygradBackend._fixed_output_shape(candidate) and has_tensor_compute(expression)

    @staticmethod
    def _fixed_output_shape(expression: Expression) -> bool:
        """Whether output shape is fixed by the input tensor signatures."""

        operation = expression["op"]
        if operation == "combinator":
            return TinygradBackend._fixed_output_shape(expand_combinator(expression))
        if operation == "train":
            return TinygradBackend._fixed_output_shape(expand_train(expression))
        if operation == "repeat":
            return TinygradBackend._fixed_output_shape(expand_repeat(expression))
        if operation == "map":
            return all(
                TinygradBackend._fixed_output_shape(child)
                for child in expression["arguments"]
            )
        if operation == "static_call":
            return TinygradBackend._fixed_output_shape(expression["argument"])
        if operation == "scalar_call":
            return TinygradBackend._fixed_output_shape(expression["argument"])
        if operation == "call":
            glyph = expression["glyph"]
            children = expression["arguments"]
            if len(children) == 1 and glyph in {"↕", "/", "⍷"}:
                return False
            if len(children) == 2:
                if glyph in {"⊒", "⍷"}:
                    return False
                if glyph in {"⥊", "⍉", "⊏", "⊑"}:
                    return False
                if glyph in {"↑", "↓", "↕", "⌽", "/"}:
                    return TinygradBackend._literal_whole_numbers_expression(
                        children[0]
                    ) and TinygradBackend._fixed_output_shape(children[1])
            return all(
                TinygradBackend._fixed_output_shape(child)
                for child in children
            )
        if operation == "fold":
            return TinygradBackend._fixed_output_shape(expression["argument"])
        if operation in {"insert", "scan"}:
            return TinygradBackend._fixed_output_shape(expression["argument"])
        return True

    @staticmethod
    def _literal_whole_numbers_expression(expression: Expression) -> bool:
        if expression["op"] == "constant":
            values = (expression["value"],)
        elif expression["op"] == "array":
            values = tuple(expression["values"])
        else:
            return False
        return all(int(value) == value for value in values)

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
