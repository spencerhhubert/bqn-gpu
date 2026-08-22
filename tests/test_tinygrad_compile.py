from __future__ import annotations

from bqn_gpu import HostValue, TinygradBackend, compile_bqn
from bqn_gpu.ir import evaluate
from bqn_gpu.optimizer import optimize


def test_only_fixed_shape_tensor_work_is_compiled() -> None:
    assert TinygradBackend.can_compile(compile_bqn("{𝕩+2×|𝕩-1}").expression)
    assert not TinygradBackend.can_compile(compile_bqn("{+𝕩}").expression)
    assert not TinygradBackend.can_compile(compile_bqn("{≢𝕩}").expression)
    assert not TinygradBackend.can_compile(compile_bqn("{↕𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{⌽⌽𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{⍉𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{1↓𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{3↕𝕩}").expression)
    assert not TinygradBackend.can_compile(compile_bqn("{𝕨↓𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{⊐𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{⊒𝕩}").expression)
    assert TinygradBackend.can_compile(compile_bqn("{∊𝕩}").expression)
    assert not TinygradBackend.can_compile(compile_bqn("{⍷𝕩}").expression)


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


def test_literal_structural_arguments_compile_once(backend: TinygradBackend) -> None:
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (4,)))
    }
    for source in ("{1↓𝕩}", "{3↕𝕩}", "{1⌽𝕩}", "{2/𝕩}"):
        program = compile_bqn(source)
        expected = evaluate(program.expression, backend, arguments).to_host()
        executable = backend.compile(program.expression, arguments)
        for _ in range(3):
            assert executable(arguments).to_host() == expected


def test_shape_specialized_optimizer_explains_structural_rewrites() -> None:
    cases = [
        ("{⌽⌽𝕩}", 1, "double-reverse"),
        ("{1⌽¯1⌽𝕩}", 1, "cancel-rotates"),
        ("{⍉⍉𝕩}", 2, "double-transpose-rank-2"),
        ("{⌽∨𝕩}", 1, "reverse-sorted-list"),
    ]
    for source, rank, rule in cases:
        original = compile_bqn(source).expression
        result = optimize(original, {"x": rank})
        assert result.expression != original
        assert rule in {event.rule for event in result.events}

    atom_reverse = compile_bqn("{⌽⌽𝕩}").expression
    assert optimize(atom_reverse, {"x": 0}).expression == atom_reverse


def test_compile_decision_follows_shape_specialized_rewrites(
    backend: TinygradBackend,
) -> None:
    expression = compile_bqn("{1⌽¯1⌽𝕩}").expression
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3], (3,)))
    }
    assert backend.can_compile(expression)
    assert backend.can_compile(expression, arguments)


def test_compiled_double_reverse_is_erased(backend: TinygradBackend) -> None:
    program = compile_bqn("{⌽⌽𝕩}")
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3], (3,)))
    }
    optimization = backend.optimize(program.expression, arguments)
    assert [event.rule for event in optimization.events] == ["double-reverse"]

    executable = backend.compile(program.expression, arguments)
    assert executable(arguments) is arguments["x"]
    assert executable.execution_mode == "optimized-noop"  # type: ignore[attr-defined]


def test_layout_only_programs_avoid_jit_replay(backend: TinygradBackend) -> None:
    vector_arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (4,)))
    }
    matrix_arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (2, 2)))
    }
    for source, arguments in (
        ("{⌽𝕩}", vector_arguments),
        ("{⍉𝕩}", matrix_arguments),
    ):
        executable = backend.compile(compile_bqn(source).expression, arguments)
        expected = evaluate(compile_bqn(source).expression, backend, arguments).to_host()
        assert executable(arguments).to_host() == expected
        assert executable.execution_mode == "specialized-eager"  # type: ignore[attr-defined]
        assert "without-jit" in executable.execution_reason  # type: ignore[attr-defined]


def test_literal_drop_keeps_jit_replay(backend: TinygradBackend) -> None:
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (4,)))
    }
    executable = backend.compile(compile_bqn("{1↓𝕩}").expression, arguments)
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]


def test_tensor_consumers_still_use_jit_replay(backend: TinygradBackend) -> None:
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (2, 2)))
    }
    executable = backend.compile(compile_bqn("{1+⍉𝕩}").expression, arguments)
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]


def test_combinator_is_inlined_before_jit_capture(backend: TinygradBackend) -> None:
    expression = compile_bqn("{+´∘|𝕩}").expression
    arguments = {
        "x": backend.from_host(HostValue.from_array([-1, 2, -3], (3,)))
    }
    optimization = backend.optimize(expression, arguments)
    assert optimization.events[0].rule == "inline-atop"
    executable = backend.compile(expression, arguments)
    assert executable(arguments).to_host() == HostValue.from_atom(6)
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]


def test_function_train_is_inlined_before_jit_capture(
    backend: TinygradBackend,
) -> None:
    expression = compile_bqn("{(⊢-+´÷≠)𝕩}").expression
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3, 4], (4,)))
    }
    optimization = backend.optimize(expression, arguments)
    assert optimization.events[0].rule == "inline-function-train"
    executable = backend.compile(expression, arguments)
    assert executable(arguments).to_host() == HostValue.from_array(
        [-1.5, -0.5, 0.5, 1.5], (4,)
    )
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]


def test_static_repeat_is_unrolled_before_execution_planning(
    backend: TinygradBackend,
) -> None:
    expression = compile_bqn("{1⊸+⍟4𝕩}").expression
    arguments = {
        "x": backend.from_host(HostValue.from_array([1, 2, 3], (3,)))
    }
    optimization = backend.optimize(expression, arguments)
    assert optimization.events[0].rule == "unroll-static-repeat"
    executable = backend.compile(expression, arguments)
    assert executable(arguments).to_host() == HostValue.from_array([5, 6, 7], (3,))
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]

    zero = compile_bqn("{-⍟0𝕩}").expression
    zero_executable = backend.compile(zero, arguments)
    assert zero_executable(arguments) is arguments["x"]
    assert zero_executable.execution_mode == "optimized-noop"  # type: ignore[attr-defined]


def test_ranked_matrix_vector_product_is_jit_captured(
    backend: TinygradBackend,
) -> None:
    expression = compile_bqn("{𝕨+˝∘×⎉1𝕩}").expression
    assert TinygradBackend._is_sum_product_operand(expression["function"])
    arguments = {
        "w": backend.from_host(HostValue.from_array([1, 2, 3, 4, 5, 6], (2, 3))),
        "x": backend.from_host(HostValue.from_array([10, 20, 30], (3,))),
    }
    executable = backend.compile(expression, arguments)
    for _ in range(3):
        assert executable(arguments).to_host() == HostValue.from_array(
            [140, 320], (2,)
        )
    assert executable.execution_mode == "jit-captured"  # type: ignore[attr-defined]
    assert executable.execution_reason == (  # type: ignore[attr-defined]
        "ranked-sum-product-lowers-to-one-batched-reduction"
    )
