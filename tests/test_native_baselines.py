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
    # Benchmarks call one compiled program repeatedly, which is when a captured
    # graph is recorded and replayed. Single calls hide capture failures.
    for _ in range(3):
        actual = executable(device_inputs)
        runtime.realize(actual)
        assert_close(runtime.to_host(actual, atom=expected.atom), expected, program)


def test_native_sources_are_independent_of_displayed_bqn() -> None:
    program = next(item for item in load_programs() if item.id == "pair.pipeline.10.naive")
    assert "𝕩" not in program.native_tinygrad
    assert "𝕩" not in program.native_torch
    assert "torch" not in program.native_tinygrad


def test_native_tinygrad_noop_uses_eager_dispatch() -> None:
    program = next(item for item in load_programs() if item.id == "glyph.conjugate.vector")
    runtime = NativeTinygradRuntime("CPU")
    inputs = generate_inputs(program, size=37)
    device_inputs = {name: runtime.from_host(value) for name, value in inputs.items()}

    executable = runtime.compile(program, device_inputs)

    assert runtime.execution_mode == "native-eager"
    assert executable(device_inputs) is device_inputs["x"]


@pytest.mark.parametrize(
    "program_id",
    ("dense.identity_left.monadic_vector", "dense.deshape.monadic_matrix"),
)
def test_native_tinygrad_kernel_free_program_falls_back_to_eager(program_id: str) -> None:
    program = next(item for item in load_programs() if item.id == program_id)
    runtime = NativeTinygradRuntime("CPU")
    inputs = generate_inputs(program, size=37)
    device_inputs = {name: runtime.from_host(value) for name, value in inputs.items()}

    executable = runtime.compile(program, device_inputs)
    for _ in range(3):
        executable(device_inputs).realize()

    assert runtime.execution_mode == "native-eager"


def test_native_tinygrad_dynamic_deduplicate_uses_eager_dispatch() -> None:
    program = next(
        item
        for item in load_programs()
        if item.id == "dense.deduplicate_major_cells.monadic_matrix"
    )
    runtime = NativeTinygradRuntime("CPU")
    inputs = generate_inputs(program, size=37)
    device_inputs = {name: runtime.from_host(value) for name, value in inputs.items()}

    executable = runtime.compile(program, device_inputs)

    assert runtime.execution_mode == "native-eager"
    assert executable(device_inputs).shape[1:] == device_inputs["x"].shape[1:]


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
