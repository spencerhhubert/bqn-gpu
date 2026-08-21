from __future__ import annotations

import os
from typing import Any

import pytest

from bqn_gpu.cbqn import CBQN
from bqn_gpu.corpus import assert_close, generate_inputs, load_programs
from bqn_gpu.errors import DeviceError
from bqn_gpu.source import compile_bqn


@pytest.fixture(scope="module")
def torch_module(cbqn: CBQN) -> Any:
    # cBQN reserves JIT address space at initialization. Loading PyTorch first
    # can occupy that range, so the fixture explicitly depends on cBQN.
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def torch_backend(torch_module: Any):
    from bqn_gpu.torch_backend import TorchBackend

    return TorchBackend(os.environ.get("BQN_GPU_TORCH_TEST_DEVICE", "CPU"))


def test_torch_executes_entire_bqn_source_corpus(
    torch_backend,
    cbqn: CBQN,
) -> None:
    for program in load_programs():
        inputs = generate_inputs(program, size=31)
        actual = compile_bqn(program.bqn).execute(torch_backend, **inputs)
        arguments = (
            (inputs["x"],)
            if program.arity == 1
            else (inputs["w"], inputs["x"])
        )
        expected = cbqn.call(program.bqn, *arguments)
        try:
            assert_close(actual, expected, program)
        except AssertionError as error:
            raise AssertionError(f"PyTorch corpus failure in {program.id}") from error


def test_torch_backend_reports_unavailable_cuda(torch_module: Any) -> None:
    from bqn_gpu.torch_backend import TorchBackend

    if torch_module.cuda.is_available():
        pytest.skip("CUDA is available on this test host")
    with pytest.raises(DeviceError, match="CUDA is unavailable"):
        TorchBackend("CUDA")
