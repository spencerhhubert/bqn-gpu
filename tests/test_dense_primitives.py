from __future__ import annotations

import os
from typing import Any

import pytest

from bqn_gpu import HostValue, TinygradBackend, compile_bqn
from bqn_gpu.cbqn import CBQN
from bqn_gpu.json_values import decode_host_value


V = decode_host_value([3, 1, 2, 1])
M = decode_host_value([[1, 2, 3], [4, 5, 6]])
E = decode_host_value([])


@pytest.fixture(scope="module", params=("tinygrad", "torch"))
def dense_backend(request: pytest.FixtureRequest, cbqn: CBQN) -> Any:
    if request.param == "tinygrad":
        return TinygradBackend(os.environ.get("BQN_GPU_TEST_DEVICE", "CPU"))
    pytest.importorskip("torch")
    from bqn_gpu.torch_backend import TorchBackend

    return TorchBackend(os.environ.get("BQN_GPU_TORCH_TEST_DEVICE", "CPU"))


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("{∧𝕩}", {"x": V}),
        ("{∨𝕩}", {"x": V}),
        ("{¬𝕩}", {"x": V}),
        ("{≡𝕩}", {"x": V}),
        ("{≡𝕩}", {"x": HostValue.from_atom(3)}),
        ("{⊣𝕩}", {"x": V}),
        ("{⊢𝕩}", {"x": V}),
        ("{⥊𝕩}", {"x": M}),
        ("{≍𝕩}", {"x": M}),
        ("{⌽𝕩}", {"x": M}),
        ("{⍉𝕩}", {"x": M}),
        ("{/𝕩}", {"x": decode_host_value([2, 0, 3])}),
        ("{⍋𝕩}", {"x": V}),
        ("{⍒𝕩}", {"x": V}),
        ("{⊏𝕩}", {"x": M}),
        ("{⊑𝕩}", {"x": M}),
        ("{⊐𝕩}", {"x": V}),
        ("{⊒𝕩}", {"x": V}),
        ("{∊𝕩}", {"x": V}),
        ("{⍷𝕩}", {"x": V}),
        ("{+˝𝕩}", {"x": M}),
        ("{+`𝕩}", {"x": M}),
        ("{×´𝕩}", {"x": V}),
        ("{∧´𝕩}", {"x": V}),
        ("{∨´𝕩}", {"x": V}),
        ("{⌊´𝕩}", {"x": V}),
        ("{⌈´𝕩}", {"x": V}),
        ("{×˝𝕩}", {"x": M}),
        ("{∧˝𝕩}", {"x": M}),
        ("{∨˝𝕩}", {"x": M}),
        ("{⌊˝𝕩}", {"x": M}),
        ("{⌈˝𝕩}", {"x": M}),
        ("{×`𝕩}", {"x": M}),
        ("{∧`𝕩}", {"x": M}),
        ("{∨`𝕩}", {"x": M}),
        ("{⌊`𝕩}", {"x": M}),
        ("{⌈`𝕩}", {"x": M}),
        ("{+˝𝕩}", {"x": E}),
        ("{+`𝕩}", {"x": E}),
        ("{𝕨∧𝕩}", {"w": V, "x": decode_host_value([1, 0, 1, 0])}),
        ("{𝕨∨𝕩}", {"w": V, "x": decode_host_value([1, 0, 1, 0])}),
        ("{𝕨¬𝕩}", {"w": V, "x": decode_host_value([1, 0, 1, 0])}),
        ("{𝕨≡𝕩}", {"w": V, "x": V}),
        ("{𝕨≢𝕩}", {"w": V, "x": decode_host_value([3, 1, 2, 0])}),
        ("{𝕨⊣𝕩}", {"w": V, "x": M}),
        ("{𝕨⊢𝕩}", {"w": V, "x": M}),
        ("{2‿3⥊𝕩}", {"x": V}),
        ("{𝕩∾𝕩}", {"x": V}),
        ("{𝕩≍𝕩}", {"x": V}),
        ("{⋈𝕩}", {"x": HostValue.from_atom(3)}),
        ("{𝕩⋈2}", {"x": HostValue.from_atom(1)}),
        ("{2↑𝕩}", {"x": V}),
        ("{¯2↑𝕩}", {"x": V}),
        ("{2↓𝕩}", {"x": V}),
        ("{¯2↓𝕩}", {"x": V}),
        ("{1⌽𝕩}", {"x": V}),
        ("{1‿0⍉𝕩}", {"x": M}),
        ("{2‿0‿1‿3/𝕩}", {"x": V}),
        ("{3↕𝕩}", {"x": V}),
        ("{2‿0‿2⊏𝕩}", {"x": V}),
        ("{1⊑𝕩}", {"x": V}),
        ("{1‿2⊑𝕩}", {"x": M}),
        ("{𝕨⍋𝕩}", {"w": decode_host_value([1, 2, 4]), "x": V}),
        ("{𝕨⍒𝕩}", {"w": decode_host_value([4, 2, 1]), "x": V}),
        ("{𝕨⊐𝕩}", {"w": V, "x": decode_host_value([1, 3, 4])}),
        ("{𝕨⊒𝕩}", {"w": V, "x": decode_host_value([1, 1, 3, 1])}),
        ("{𝕨∊𝕩}", {"w": decode_host_value([1, 4, 3]), "x": V}),
        ("{𝕨⍷𝕩}", {"w": decode_host_value([1, 2]), "x": V}),
    ],
)
def test_dense_numeric_primitive_matches_cbqn(
    source: str,
    arguments: dict[str, HostValue],
    dense_backend: Any,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(dense_backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    expected = cbqn.call(source, *oracle_arguments)
    assert actual == expected


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("{×˜𝕩}", {"x": HostValue.from_atom(4)}),
        ("{𝕨-˜𝕩}", {"w": HostValue.from_atom(3), "x": HostValue.from_atom(10)}),
        ("{|∘-𝕩}", {"x": HostValue.from_atom(4)}),
        ("{𝕨|∘-𝕩}", {"w": HostValue.from_atom(3), "x": HostValue.from_atom(10)}),
        (
            "{𝕨-○|𝕩}",
            {"w": HostValue.from_atom(-3), "x": HostValue.from_atom(-10)},
        ),
        (
            "{𝕨⋆⊸-𝕩}",
            {"w": HostValue.from_atom(2), "x": HostValue.from_atom(3)},
        ),
        (
            "{𝕨⋆⟜-𝕩}",
            {"w": HostValue.from_atom(2), "x": HostValue.from_atom(3)},
        ),
        ("{¯1⊸⌽𝕩}", {"x": decode_host_value([1, 2, 3, 4])}),
        ("{-⟜1 𝕩}", {"x": HostValue.from_atom(8)}),
        ("{+´∘|𝕩}", {"x": decode_host_value([-1, 2, -3])}),
    ],
)
def test_dense_combinators_match_cbqn(
    source: str,
    arguments: dict[str, HostValue],
    dense_backend: Any,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(dense_backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    assert actual == cbqn.call(source, *oracle_arguments)


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("{(⊢+⌽)𝕩}", {"x": V}),
        ("{(⌽⍉)𝕩}", {"x": M}),
        ("{(+´÷≠)𝕩}", {"x": V}),
        ("{(+´∘|÷≠)𝕩}", {"x": V}),
        ("{(⊢-+´÷≠)𝕩}", {"x": V}),
        ("{(⌽⊢-≠)𝕩}", {"x": V}),
        ("{(⊢+⌽-≠)𝕩}", {"x": V}),
        ("{((⊢+⌽)×≠)𝕩}", {"x": V}),
        ("{(1+⊢)𝕩}", {"x": V}),
        (
            "{𝕨(+⋈-)𝕩}",
            {"w": HostValue.from_atom(10), "x": HostValue.from_atom(3)},
        ),
    ],
)
def test_dense_function_trains_match_cbqn(
    source: str,
    arguments: dict[str, HostValue],
    dense_backend: Any,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(dense_backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    assert actual == cbqn.call(source, *oracle_arguments)


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("{-⍟0𝕩}", {"x": V}),
        ("{-⍟2𝕩}", {"x": V}),
        ("{⌽⍟2𝕩}", {"x": V}),
        ("{|⍟3𝕩}", {"x": V}),
        ("{1⊸+⍟4𝕩}", {"x": V}),
        ("{(⊢-+´÷≠)⍟2𝕩}", {"x": V}),
        ("{𝕨+⍟3𝕩}", {"w": V, "x": decode_host_value([1, 0, 1, 0])}),
    ],
)
def test_dense_static_repeat_matches_cbqn(
    source: str,
    arguments: dict[str, HostValue],
    dense_backend: Any,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(dense_backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    assert actual == cbqn.call(source, *oracle_arguments)


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("{+¨𝕩}", {"x": HostValue.from_atom(3)}),
        ("{-¨𝕩}", {"x": M}),
        (
            "{𝕨+¨𝕩}",
            {"w": decode_host_value([10, 20]), "x": M},
        ),
        (
            "{𝕨×⌜𝕩}",
            {"w": decode_host_value([10, 20]), "x": decode_host_value([1, 2, 3])},
        ),
        ("{⌽˘𝕩}", {"x": M}),
        ("{≢˘𝕩}", {"x": M}),
        ("{≡˘𝕩}", {"x": V}),
        ("{≡¨𝕩}", {"x": V}),
        ("{≡⎉0𝕩}", {"x": HostValue.from_array([3], ())}),
        ("{⌽⎉1𝕩}", {"x": M}),
        ("{⌽⎉¯1𝕩}", {"x": M}),
        ("{⌽⎉∞𝕩}", {"x": M}),
        ("{≢⎉¯∞𝕩}", {"x": M}),
        ("{≢⎉0‿1‿2𝕩}", {"x": M}),
        (
            "{𝕨+⎉1‿1𝕩}",
            {"w": M, "x": decode_host_value([10, 20, 30])},
        ),
        (
            "{𝕨+˘𝕩}",
            {"w": M, "x": HostValue.from_array([10], ())},
        ),
        (
            "{𝕨+⎉1‿2𝕩}",
            {
                "w": M,
                "x": decode_host_value(
                    [
                        [[10, 20], [30, 40], [50, 60]],
                        [[70, 80], [90, 100], [110, 120]],
                    ]
                ),
            },
        ),
        (
            "{𝕨+⎉0‿1‿2𝕩}",
            {
                "w": M,
                "x": decode_host_value(
                    [
                        [[10, 20], [30, 40], [50, 60]],
                        [[70, 80], [90, 100], [110, 120]],
                    ]
                ),
            },
        ),
        (
            "{𝕨+˝∘×⎉1𝕩}",
            {"w": M, "x": decode_host_value([10, 20, 30])},
        ),
        ("{-⟜1¨𝕩}", {"x": V}),
        ("{+´∘|˘𝕩}", {"x": M}),
    ],
)
def test_dense_mapping_modifiers_match_cbqn(
    source: str,
    arguments: dict[str, HostValue],
    dense_backend: Any,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(dense_backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    assert actual == cbqn.call(source, *oracle_arguments)
