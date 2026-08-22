"""Independent direct-PyTorch corpus programs used as performance baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from .corpus import Program
from .errors import DeviceError
from .host_value import HostValue


def major_cell_self_search(x: torch.Tensor, glyph: str) -> torch.Tensor:
    """Independent direct-Torch implementation of monadic self-search."""

    if len(x.shape) == 0:
        raise ValueError("self-search requires an array with at least one axis")
    count = int(x.shape[0])
    if count == 0:
        return x if glyph == "⍷" else torch.empty(0, dtype=torch.float64, device=x.device)
    cells = x.reshape((count, -1))
    cell_size = int(cells.shape[1])
    equal = (
        torch.ones((count, count), dtype=torch.bool, device=x.device)
        if cell_size == 0
        else torch.all(
            cells.reshape((count, 1, cell_size))
            == cells.reshape((1, count, cell_size)),
            dim=2,
        )
    )
    positions = torch.arange(count, device=x.device).reshape((1, count))
    if glyph == "⊐":
        first_positions = torch.where(equal, positions, count).min(dim=1).values
        own_positions = torch.arange(count, device=x.device)
        firsts = first_positions == own_positions
        class_numbers = torch.cumsum(firsts.to(torch.float64), dim=0) - 1
        return class_numbers[first_positions]
    rows = torch.arange(count, device=x.device).reshape((count, 1))
    occurrences = (equal & (positions < rows)).sum(dim=1).to(torch.float64)
    firsts = occurrences == 0
    if glyph == "⍷":
        return x[firsts]
    return firsts.to(torch.float64) if glyph == "∊" else occurrences


class NativeTorchRuntime:
    """Run generated direct PyTorch sources without the BQN frontend/backend."""

    def __init__(self, device: str = "CPU") -> None:
        try:
            requested = torch.device(device.lower())
        except (RuntimeError, ValueError) as error:
            raise DeviceError(f"invalid PyTorch device {device!r}: {error}") from error
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise DeviceError("PyTorch CUDA execution was requested but CUDA is unavailable")
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        self.device = requested

    def from_host(self, value: HostValue) -> torch.Tensor:
        tensor = torch.tensor(value.data, dtype=torch.float64, device=self.device)
        return tensor.reshape(()) if value.atom else tensor.reshape(value.shape)

    @staticmethod
    def compile(
        program: Program,
        arguments: Mapping[str, torch.Tensor],
    ) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
        names = ("x",) if program.arity == 1 else ("w", "x")
        if set(arguments) != set(names):
            raise ValueError(f"{program.id}: native Torch argument mismatch")
        function = eval(  # noqa: S307 - generated, tracked corpus source
            program.native_torch,
            {"torch": torch, "major_cell_self_search": major_cell_self_search},
        )

        def execute(supplied: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return function(*(supplied[name] for name in names))

        return execute

    @staticmethod
    def realize(_value: torch.Tensor) -> None:
        return None

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def to_host(value: torch.Tensor, *, atom: bool) -> HostValue:
        tensor = value.detach().cpu()
        if atom:
            return HostValue.from_atom(float(tensor.item()))
        data = tuple(float(item) for item in tensor.flatten().tolist())
        return HostValue.from_array(data, tuple(int(length) for length in tensor.shape))
