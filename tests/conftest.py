from __future__ import annotations

import os
from pathlib import Path

import pytest

from bqn_gpu import TinygradBackend
from cbqn_oracle import CBQNOracle


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def backend() -> TinygradBackend:
    return TinygradBackend(os.environ.get("BQN_GPU_TEST_DEVICE", "CPU"))


@pytest.fixture(scope="session")
def cbqn() -> CBQNOracle:
    path = Path(os.environ.get("CBQN_LIB", ROOT / ".build/cbqn/libcbqn.so"))
    oracle = CBQNOracle(path)
    yield oracle
    oracle.close()
