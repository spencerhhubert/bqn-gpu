"""Public API for bqn-gpu's source frontend and execution backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import (
    BQNGPUError,
    DeviceError,
    DomainError,
    ShapeError,
    SourceError,
    UnsupportedPrimitive,
)
from .host_value import HostValue
from .source import CompiledProgram, compile_bqn, execute

if TYPE_CHECKING:
    from .tinygrad_backend import TinygradBackend, TinygradValue
    from .torch_backend import TorchBackend, TorchValue


__all__ = [
    "BQNGPUError",
    "CompiledProgram",
    "DeviceError",
    "DomainError",
    "HostValue",
    "ShapeError",
    "SourceError",
    "TinygradBackend",
    "TinygradValue",
    "TorchBackend",
    "TorchValue",
    "UnsupportedPrimitive",
    "compile_bqn",
    "execute",
]


def __getattr__(name: str) -> Any:
    if name in {"TinygradBackend", "TinygradValue"}:
        from .tinygrad_backend import TinygradBackend, TinygradValue

        return {"TinygradBackend": TinygradBackend, "TinygradValue": TinygradValue}[name]
    if name in {"TorchBackend", "TorchValue"}:
        from .torch_backend import TorchBackend, TorchValue

        return {"TorchBackend": TorchBackend, "TorchValue": TorchValue}[name]
    raise AttributeError(name)
