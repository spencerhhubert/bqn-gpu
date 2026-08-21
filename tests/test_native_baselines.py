from __future__ import annotations

import pytest

from bqn_gpu.corpus import assert_close, generate_inputs, load_programs
from bqn_gpu.native_tinygrad import NativeTinygradRuntime


@pytest.mark.parametrize("program", load_programs(), ids=lambda program: program.id)
def test_native_tinygrad_program_matches_cbqn(program, cbqn) -> None:
    inputs = generate_inputs(program, size=37)
    expected = cbqn.call(
        program.bqn,
        *((inputs["x"],) if program.arity == 1 else (inputs["w"], inputs["x"])),
    )
    runtime = NativeTinygradRuntime("CPU")
    device_inputs = {name: runtime.from_host(value) for name, value in inputs.items()}
    executable = runtime.compile(program, device_inputs)
    actual = executable(device_inputs)
    runtime.realize(actual)
    assert_close(runtime.to_host(actual, atom=expected.atom), expected, program)


def test_native_sources_are_independent_of_displayed_bqn() -> None:
    program = next(item for item in load_programs() if item.id == "pair.pipeline.10.naive")
    assert "𝕩" not in program.native_tinygrad
    assert "𝕩" not in program.native_torch
    assert "torch" not in program.native_tinygrad


def test_every_native_torch_program_matches_cbqn(cbqn) -> None:
    # cBQN reserves JIT address space and must initialize before importing Torch.
    from bqn_gpu.native_torch import NativeTorchRuntime

    runtime = NativeTorchRuntime("CPU")
    for program in load_programs():
        inputs = generate_inputs(program, size=37)
        expected = cbqn.call(
            program.bqn,
            *((inputs["x"],) if program.arity == 1 else (inputs["w"], inputs["x"])),
        )
        device_inputs = {name: runtime.from_host(value) for name, value in inputs.items()}
        executable = runtime.compile(program, device_inputs)
        actual = executable(device_inputs)
        assert_close(runtime.to_host(actual, atom=expected.atom), expected, program)
