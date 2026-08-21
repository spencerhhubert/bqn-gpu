from __future__ import annotations

from bqn_gpu import HostValue, TinygradBackend, compile_bqn
from bqn_gpu.ir import evaluate


def test_compiled_tinygrad_program_reuses_source_graph(
    backend: TinygradBackend,
) -> None:
    program = compile_bqn("{a←|𝕩 ⋄ b←a×a ⋄ +´b}")
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, -2, 3], (3,)))
    }
    executable = backend.compile(program.expression, arguments)

    # TinyJit's first call executes normally, its second captures, and later
    # calls replay. All phases must preserve the complete BQN value.
    for _ in range(4):
        assert executable(arguments).to_host() == HostValue.from_atom(14)


def test_compiled_and_eager_tinygrad_results_match(backend: TinygradBackend) -> None:
    program = compile_bqn("{𝕩+2×|𝕩-1}")
    arguments = {
        "x": backend.from_host(HostValue.from_array([-3, 0, 5], (3,)))
    }
    expected = evaluate(program.expression, backend, arguments).to_host()
    executable = backend.compile(program.expression, arguments)
    for _ in range(3):
        assert executable(arguments).to_host() == expected
