"""PyTorch adapter for the currently supported BQN primitive surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterable, Sequence

import torch

from .errors import DeviceError, DomainError, ShapeError, UnsupportedPrimitive
from .host_value import HostValue, Shape
from .mapping import plan_mapping


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
        if glyph == "⋆⁼":
            return TorchValue(tensor=x.tensor.log(), atom=x.atom)
        if glyph in {"»", "«"}:
            return self._shift(glyph, None, x)
        if glyph in {"∧", "∨"}:
            if len(x.shape) != 1:
                raise DomainError("Sort is currently supported for numeric lists")
            tensor = torch.sort(x.tensor, dim=0, descending=glyph == "∨").values
            return TorchValue(tensor=tensor, atom=False)
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
        if glyph == "≡":
            return self.atom(0 if x.atom else 1)
        if glyph in {"⊣", "⊢"}:
            return x
        if glyph == "⥊":
            return TorchValue(tensor=x.tensor.reshape((x.tensor.numel(),)), atom=False)
        if glyph == "≍":
            return TorchValue(tensor=x.tensor.reshape((1,) + x.shape), atom=False)
        if glyph == "⌽":
            if len(x.shape) == 0:
                raise DomainError("Reverse requires an array with at least one axis")
            return TorchValue(tensor=torch.flip(x.tensor, (0,)), atom=False)
        if glyph == "⍉":
            if x.atom:
                return TorchValue(tensor=x.tensor, atom=False)
            if len(x.shape) <= 1:
                return x
            axes = tuple(range(1, len(x.shape))) + (0,)
            return TorchValue(tensor=x.tensor.permute(axes), atom=False)
        if glyph == "/":
            counts = self._whole_numbers(x, "Indices", natural=True)
            indices = [index for index, count in enumerate(counts) for _ in range(count)]
            return self._index_list(indices)
        if glyph in {"⍋", "⍒"}:
            if len(x.shape) != 1:
                raise DomainError("Grade is currently supported for numeric lists")
            tensor = torch.argsort(x.tensor, dim=0, descending=glyph == "⍒", stable=True).to(torch.float64)
            return TorchValue(tensor=tensor, atom=False)
        if glyph == "⊏":
            if len(x.shape) == 0 or x.shape[0] == 0:
                raise DomainError("First Cell requires a nonempty array with an axis")
            return TorchValue(tensor=x.tensor[0], atom=False)
        if glyph == "⊑":
            if x.tensor.numel() == 0:
                raise DomainError("First requires a nonempty value")
            return TorchValue(tensor=x.tensor.reshape((-1,))[0], atom=True)
        if glyph in {"⊐", "⊒", "∊", "⍷"}:
            return self._self_search(glyph, x)
        if glyph == "⋈":
            if not x.atom:
                raise DomainError("dense Enlist is currently supported only for atoms")
            return TorchValue(tensor=x.tensor.reshape((1,)), atom=False)
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
            "¬": lambda tensor: 1.0 - tensor,
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
        if glyph in {"≡", "≢"}:
            matches = w.atom == x.atom and w.shape == x.shape
            equal = matches and torch.equal(w.tensor, x.tensor)
            return self.atom(float(not equal) if glyph == "≢" else float(equal))
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
            return TorchValue(tensor=torch.stack((w.tensor, x.tensor), dim=0), atom=False)
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
        elif glyph == "∧":
            tensor = w_tensor * x_tensor
        elif glyph == "∨":
            tensor = w_tensor + x_tensor - w_tensor * x_tensor
        elif glyph == "¬":
            tensor = 1.0 + w_tensor - x_tensor
        else:
            raise UnsupportedPrimitive(f"dyadic primitive {glyph!r} is not implemented")
        return TorchValue(tensor=tensor, atom=w.atom and x.atom)

    def reduce(self, glyph: str, argument: TorchValue) -> TorchValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) != 1:
            raise DomainError("BQN Fold is currently supported only for numeric lists")
        tensor = self._reduce_tensor(glyph, argument.tensor, axis=0)
        return TorchValue(tensor=tensor, atom=True)

    def insert(self, glyph: str, argument: TorchValue) -> TorchValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) == 0:
            raise DomainError("Insert requires an array with at least one axis")
        tensor = self._reduce_tensor(glyph, argument.tensor, axis=0)
        return TorchValue(
            tensor=tensor,
            atom=len(argument.shape) == 1 and argument.shape[0] != 0,
        )

    def scan(self, glyph: str, argument: TorchValue) -> TorchValue:
        self._check_device(argument)
        if argument.atom or len(argument.shape) == 0:
            raise DomainError("Scan requires an array with at least one axis")
        if argument.shape[0] == 0:
            return argument
        if glyph == "+":
            tensor = torch.cumsum(argument.tensor, dim=0)
        elif glyph in {"×", "∧"}:
            tensor = torch.cumprod(argument.tensor, dim=0)
        elif glyph == "∨":
            tensor = 1.0 - torch.cumprod(1.0 - argument.tensor, dim=0)
        elif glyph == "⌊":
            tensor = torch.cummin(argument.tensor, dim=0).values
        elif glyph == "⌈":
            tensor = torch.cummax(argument.tensor, dim=0).values
        else:
            raise UnsupportedPrimitive(f"Scan with {glyph!r} is not implemented")
        return TorchValue(tensor=tensor, atom=False)

    def map_function(
        self,
        modifier: str,
        rank_specification: Sequence[float],
        operand: object,
        arguments: Sequence[TorchValue],
        function: Callable[[Sequence[TorchValue]], TorchValue],
    ) -> TorchValue:
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
            return TorchValue(tensor=result.tensor, atom=False)
        if modifier == "¨" and len(arguments) == 2 and primitive in dyadic_pervasive:
            result = self.call(primitive, *arguments)
            return TorchValue(tensor=result.tensor, atom=False)
        if modifier == "⌜" and len(arguments) == 2 and primitive in dyadic_pervasive:
            left, right = arguments
            output_shape = left.shape + right.shape
            left_tensor = left.tensor.reshape(left.shape + (1,) * len(right.shape)).expand(output_shape)
            right_tensor = right.tensor.reshape((1,) * len(left.shape) + right.shape).expand(output_shape)
            result = self.call(
                primitive,
                TorchValue(left_tensor, atom=False),
                TorchValue(right_tensor, atom=False),
            )
            return TorchValue(tensor=result.tensor, atom=False)

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
            return TorchValue(
                tensor=(tensors[0] * tensors[1]).sum(dim=len(plan.frame_shape)),
                atom=False,
            )
        if any(length == 0 for length in plan.frame_shape):
            raise DomainError("mapping over an empty frame is not implemented yet")
        results: list[TorchValue] = []
        for indices in plan.indices():
            cells = []
            for argument, index, cell_rank in zip(
                arguments, indices, plan.cell_ranks, strict=True
            ):
                tensor = argument.tensor if not index else argument.tensor[index]
                cells.append(
                    TorchValue(
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

    @staticmethod
    def _combine_mapped_results(
        frame_shape: Shape,
        results: Sequence[TorchValue],
    ) -> TorchValue:
        if not results:
            raise DomainError("mapping produced no result cells")
        result_shape = results[0].shape
        if any(result.shape != result_shape for result in results[1:]):
            raise ShapeError("mapped function returned incompatible dense result shapes")
        if not frame_shape:
            return TorchValue(
                tensor=results[0].tensor.reshape(result_shape),
                atom=False,
            )
        tensor = (
            results[0].tensor.reshape((1,) + result_shape)
            if len(results) == 1
            else torch.stack(tuple(result.tensor for result in results), dim=0)
        )
        return TorchValue(tensor=tensor.reshape(frame_shape + result_shape), atom=False)

    def _reduce_tensor(
        self, glyph: str, tensor: torch.Tensor, *, axis: int
    ) -> torch.Tensor:
        if tensor.shape[axis] == 0:
            identities = {"+": 0.0, "×": 1.0, "∧": 1.0, "∨": 0.0, "⌊": math.inf, "⌈": -math.inf}
            try:
                identity = identities[glyph]
            except KeyError:
                raise UnsupportedPrimitive(f"reduction with {glyph!r} is not implemented") from None
            shape = tuple(length for index, length in enumerate(tensor.shape) if index != axis)
            return torch.full(shape, identity, dtype=torch.float64, device=self.torch_device)
        if glyph == "+":
            return torch.sum(tensor, dim=axis)
        if glyph in {"×", "∧"}:
            return torch.prod(tensor, dim=axis)
        if glyph == "∨":
            return 1.0 - torch.prod(1.0 - tensor, dim=axis)
        if glyph == "⌊":
            return torch.min(tensor, dim=axis).values
        if glyph == "⌈":
            return torch.max(tensor, dim=axis).values
        raise UnsupportedPrimitive(f"reduction with {glyph!r} is not implemented")

    def _reshape(self, w: TorchValue, x: TorchValue) -> TorchValue:
        shape = self._whole_numbers(w, "Reshape", natural=True)
        if len(w.shape) > 1:
            raise DomainError("Reshape requires a natural-number atom or list")
        target_size = math.prod(shape)
        source = x.tensor.reshape((x.tensor.numel(),))
        if target_size == 0:
            tensor = torch.empty(shape, dtype=torch.float64, device=self.torch_device)
        elif source.numel() == 0:
            raise DomainError("Reshape cannot fill a nonempty result from an empty array")
        else:
            repetitions = math.ceil(target_size / source.numel())
            tensor = source.repeat(repetitions)[:target_size].reshape(shape)
        return TorchValue(tensor=tensor, atom=False)

    def _join_to(self, w: TorchValue, x: TorchValue) -> TorchValue:
        result_rank = max(1, len(w.shape), len(x.shape))
        if result_rank - len(w.shape) > 1 or result_rank - len(x.shape) > 1:
            raise ShapeError("Join To arguments may differ in rank by at most one")
        w_tensor = w.tensor.reshape((1,) + w.shape) if len(w.shape) < result_rank else w.tensor
        x_tensor = x.tensor.reshape((1,) + x.shape) if len(x.shape) < result_rank else x.tensor
        if tuple(w_tensor.shape[1:]) != tuple(x_tensor.shape[1:]):
            raise ShapeError("Join To requires matching trailing cell shapes")
        return TorchValue(tensor=torch.cat((w_tensor, x_tensor), dim=0), atom=False)

    def _shift(
        self,
        glyph: str,
        w: TorchValue | None,
        x: TorchValue,
    ) -> TorchValue:
        if len(x.shape) == 0:
            raise DomainError("Shift requires a right argument with at least one axis")
        length = x.shape[0]
        if w is None:
            if length == 0:
                return x
            inserted = torch.zeros(
                (1,) + x.shape[1:],
                dtype=torch.float64,
                device=self.torch_device,
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
            output = torch.cat((inserted, x.tensor[: length - inserted_length]), dim=0)
        else:
            output = torch.cat((x.tensor[inserted_length:], inserted), dim=0)
        return TorchValue(tensor=output, atom=False)

    def _couple(self, w: TorchValue, x: TorchValue) -> TorchValue:
        if w.atom != x.atom or w.shape != x.shape:
            raise DomainError("dense Couple requires arguments with the same kind and shape")
        return TorchValue(tensor=torch.stack((w.tensor, x.tensor), dim=0), atom=False)

    def _take_or_drop(
        self, w: TorchValue, x: TorchValue, *, take: bool
    ) -> TorchValue:
        counts = self._whole_numbers(w, "Take" if take else "Drop")
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
        return TorchValue(tensor=tensor[tuple(slices)], atom=False)

    def _rotate(self, w: TorchValue, x: TorchValue) -> TorchValue:
        rotations = self._whole_numbers(w, "Rotate")
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        if len(rotations) > tensor.ndim:
            raise DomainError("Rotate specifies more axes than the right argument")
        if not rotations:
            return TorchValue(tensor=tensor, atom=False)
        return TorchValue(
            tensor=torch.roll(
                tensor,
                shifts=tuple(-rotation for rotation in rotations),
                dims=tuple(range(len(rotations))),
            ),
            atom=False,
        )

    def _reorder_axes(self, w: TorchValue, x: TorchValue) -> TorchValue:
        destinations = list(self._whole_numbers(w, "Reorder Axes", natural=True))
        rank = len(x.shape)
        if len(destinations) > rank:
            raise DomainError("Reorder Axes specifies more axes than the right argument")
        unused = [axis for axis in range(rank) if axis not in destinations]
        destinations.extend(unused[: rank - len(destinations)])
        if sorted(destinations) != list(range(rank)):
            raise DomainError("dense Reorder Axes currently requires a permutation")
        if rank == 0:
            return TorchValue(tensor=x.tensor, atom=False)
        axes = tuple(sorted(range(rank), key=destinations.__getitem__))
        return TorchValue(tensor=x.tensor.permute(axes), atom=False)

    def _replicate(self, w: TorchValue, x: TorchValue) -> TorchValue:
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        counts = self._whole_numbers(w, "Replicate", natural=True)
        if w.atom:
            counts = counts * int(tensor.shape[0])
        if len(counts) != tensor.shape[0]:
            raise ShapeError("Replicate counts must match the first-axis length")
        count_tensor = torch.tensor(counts, dtype=torch.int64, device=self.torch_device)
        output = torch.repeat_interleave(tensor, count_tensor, dim=0)
        return TorchValue(tensor=output, atom=False)

    def _windows(self, w: TorchValue, x: TorchValue) -> TorchValue:
        sizes = self._whole_numbers(w, "Windows", natural=True)
        tensor = x.tensor.reshape((1,)) if x.atom else x.tensor
        rank = tensor.ndim
        if len(sizes) > rank:
            raise DomainError("Windows specifies more axes than the right argument")
        if any(size == 0 for size in sizes):
            raise DomainError("zero-sized Windows are not yet in the dense tier")
        output_lengths = tuple(max(0, int(tensor.shape[axis]) - size + 1) for axis, size in enumerate(sizes))
        if any(length == 0 for length in output_lengths):
            shape = output_lengths + sizes + tuple(int(length) for length in tensor.shape[len(sizes):])
            output = torch.empty(shape, dtype=torch.float64, device=self.torch_device)
            return TorchValue(tensor=output, atom=False)
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
        return TorchValue(tensor=output, atom=False)

    def _select(self, w: TorchValue, x: TorchValue) -> TorchValue:
        if x.atom or len(x.shape) == 0:
            raise DomainError("Select requires a right argument with a first axis")
        indices = self._whole_numbers(w, "Select", natural=True)
        if any(index >= x.shape[0] for index in indices):
            raise DomainError("Select index is outside the first axis")
        if w.atom:
            output = x.tensor[indices[0]]
        elif len(w.shape) <= 1:
            index = torch.tensor(indices, dtype=torch.int64, device=self.torch_device)
            output = torch.index_select(x.tensor, 0, index).reshape(w.shape + x.shape[1:])
        else:
            raise DomainError("dense Select currently accepts one index array")
        return TorchValue(tensor=output, atom=False)

    def _pick(self, w: TorchValue, x: TorchValue) -> TorchValue:
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
        return TorchValue(tensor=output, atom=len(indices) == len(x.shape))

    def _bins(self, glyph: str, w: TorchValue, x: TorchValue) -> TorchValue:
        if len(w.shape) != 1:
            raise DomainError("Bins currently requires a numeric list left argument")
        right = x.tensor.reshape((-1,))
        output = torch.searchsorted(w.tensor, right, right=True) if glyph == "⍋" else torch.searchsorted(-w.tensor, -right, right=True)
        return TorchValue(tensor=output.to(torch.float64).reshape(x.shape), atom=x.atom)

    def _self_search(self, glyph: str, x: TorchValue) -> TorchValue:
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
            equal = torch.ones(
                (count, count),
                dtype=torch.bool,
                device=self.torch_device,
            )
        else:
            equal = torch.all(
                cells.reshape((count, 1, cell_size))
                == cells.reshape((1, count, cell_size)),
                dim=2,
            )
        positions = torch.arange(count, device=self.torch_device).reshape((1, count))
        if glyph == "⊐":
            first_positions = torch.where(equal, positions, count).min(dim=1).values
            own_positions = torch.arange(count, device=self.torch_device)
            firsts = first_positions == own_positions
            class_numbers = torch.cumsum(firsts.to(torch.float64), dim=0) - 1
            output = class_numbers[first_positions]
        else:
            rows = torch.arange(count, device=self.torch_device).reshape((count, 1))
            occurrences = (equal & (positions < rows)).sum(dim=1).to(torch.float64)
            firsts = occurrences == 0
            if glyph == "⍷":
                return TorchValue(tensor=x.tensor[firsts], atom=False)
            output = firsts.to(torch.float64) if glyph == "∊" else occurrences
        return TorchValue(tensor=output, atom=False)

    def _search(self, glyph: str, w: TorchValue, x: TorchValue) -> TorchValue:
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
            output = torch.tensor(output_host, dtype=torch.float64, device=self.torch_device).reshape(result.shape)
            return TorchValue(tensor=output, atom=result.atom)
        matches = query_values.reshape((-1, 1)) == principal_values.reshape((1, -1))
        if glyph == "∊":
            output = matches.any(dim=1).to(torch.float64)
        elif principal.shape[0] == 0:
            output = torch.full((query_values.numel(),), 0.0, dtype=torch.float64, device=self.torch_device)
        else:
            positions = torch.arange(principal.shape[0], device=self.torch_device).reshape((1, -1))
            output = torch.where(matches, positions, principal.shape[0]).min(dim=1).values.to(torch.float64)
        return TorchValue(tensor=output.reshape(result.shape), atom=result.atom)

    def _find(self, w: TorchValue, x: TorchValue) -> TorchValue:
        if len(w.shape) != 1 or len(x.shape) != 1:
            raise DomainError("Find is currently supported for numeric lists")
        pattern_length, input_length = w.shape[0], x.shape[0]
        if pattern_length == 0:
            return TorchValue(tensor=torch.ones(input_length + 1, dtype=torch.float64, device=self.torch_device), atom=False)
        if pattern_length > input_length:
            return TorchValue(tensor=torch.empty(0, dtype=torch.float64, device=self.torch_device), atom=False)
        windows = x.tensor.unfold(0, pattern_length, 1)
        output = torch.all(windows == w.tensor.reshape((1, pattern_length)), dim=1).to(torch.float64)
        return TorchValue(tensor=output, atom=False)

    def _index_list(self, values: Sequence[int]) -> TorchValue:
        tensor = torch.tensor(values, dtype=torch.float64, device=self.torch_device)
        return TorchValue(tensor=tensor, atom=False)

    @staticmethod
    def _whole_numbers(
        value: TorchValue, operation: str, *, natural: bool = False
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
