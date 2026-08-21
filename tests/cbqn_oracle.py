"""Small ctypes wrapper around cBQN's public embedding interface."""

from __future__ import annotations

import ctypes
from pathlib import Path

from bqn_gpu import HostValue


BQNV = ctypes.c_uint64


class CBQNOracle:
    def __init__(self, library_path: Path) -> None:
        if not library_path.is_file():
            raise FileNotFoundError(
                f"cBQN oracle not found at {library_path}; run scripts/build_cbqn.sh"
            )
        self.library = ctypes.CDLL(str(library_path))
        self._configure_api()
        self.library.bqn_init()
        self._functions: dict[str, int] = {}

    def _configure_api(self) -> None:
        api = self.library
        api.bqn_init.argtypes = []
        api.bqn_init.restype = None
        api.bqn_free.argtypes = [BQNV]
        api.bqn_free.restype = None
        api.bqn_evalCStr.argtypes = [ctypes.c_char_p]
        api.bqn_evalCStr.restype = BQNV
        api.bqn_type.argtypes = [BQNV]
        api.bqn_type.restype = ctypes.c_int
        api.bqn_call1.argtypes = [BQNV, BQNV]
        api.bqn_call1.restype = BQNV
        api.bqn_call2.argtypes = [BQNV, BQNV, BQNV]
        api.bqn_call2.restype = BQNV
        api.bqn_makeF64.argtypes = [ctypes.c_double]
        api.bqn_makeF64.restype = BQNV
        api.bqn_makeF64Arr.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double),
        ]
        api.bqn_makeF64Arr.restype = BQNV
        api.bqn_readF64.argtypes = [BQNV]
        api.bqn_readF64.restype = ctypes.c_double
        api.bqn_rank.argtypes = [BQNV]
        api.bqn_rank.restype = ctypes.c_size_t
        api.bqn_bound.argtypes = [BQNV]
        api.bqn_bound.restype = ctypes.c_size_t
        api.bqn_shape.argtypes = [BQNV, ctypes.POINTER(ctypes.c_size_t)]
        api.bqn_shape.restype = None
        api.bqn_readF64Arr.argtypes = [BQNV, ctypes.POINTER(ctypes.c_double)]
        api.bqn_readF64Arr.restype = None

    def close(self) -> None:
        for function in self._functions.values():
            self.library.bqn_free(function)
        self._functions.clear()

    def call(self, glyph: str, *arguments: HostValue) -> HostValue:
        if len(arguments) not in (1, 2):
            raise ValueError("cBQN oracle wrapper supports monadic and dyadic calls")
        function = self._functions.get(glyph)
        if function is None:
            function = int(self.library.bqn_evalCStr(glyph.encode("utf-8")))
            self._functions[glyph] = function

        encoded = [self._make_value(value) for value in arguments]
        try:
            if len(encoded) == 1:
                result = int(self.library.bqn_call1(function, encoded[0]))
            else:
                result = int(self.library.bqn_call2(function, encoded[0], encoded[1]))
            try:
                return self._read_value(result)
            finally:
                self.library.bqn_free(result)
        finally:
            for value in encoded:
                self.library.bqn_free(value)

    def _make_value(self, value: HostValue) -> int:
        if value.atom:
            return int(self.library.bqn_makeF64(value.data[0]))

        shape_buffer = (ctypes.c_size_t * max(1, len(value.shape)))(*value.shape)
        data_buffer = (ctypes.c_double * max(1, len(value.data)))(*value.data)
        return int(
            self.library.bqn_makeF64Arr(
                len(value.shape),
                shape_buffer,
                data_buffer,
            )
        )

    def _read_value(self, value: int) -> HostValue:
        value_type = self.library.bqn_type(value)
        if value_type == 1:
            return HostValue.from_atom(self.library.bqn_readF64(value))
        if value_type != 0:
            raise TypeError(f"expected cBQN numeric value, got •Type {value_type}")

        rank = int(self.library.bqn_rank(value))
        shape_buffer = (ctypes.c_size_t * max(1, rank))()
        self.library.bqn_shape(value, shape_buffer)
        shape = tuple(int(shape_buffer[index]) for index in range(rank))
        bound = int(self.library.bqn_bound(value))
        data_buffer = (ctypes.c_double * max(1, bound))()
        self.library.bqn_readF64Arr(value, data_buffer)
        data = tuple(float(data_buffer[index]) for index in range(bound))
        return HostValue.from_array(data, shape)
