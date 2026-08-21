from __future__ import annotations

import os
from pathlib import Path

import pytest

from bqn_gpu import TinygradBackend
from bqn_gpu.cbqn import CBQN


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def backend(cbqn: CBQN) -> TinygradBackend:
    # cBQN reserves JIT address space during initialization. Initialize it
    # before tensor runtimes start worker threads or allocate code mappings.
    return TinygradBackend(os.environ.get("BQN_GPU_TEST_DEVICE", "CPU"))


@pytest.fixture(scope="session")
def cbqn() -> CBQN:
    path = Path(os.environ.get("CBQN_LIB", ROOT / ".build/cbqn/libcbqn.so"))
    oracle = CBQN(path)
    yield oracle
    oracle.close()
