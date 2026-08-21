from __future__ import annotations

import pytest

from bqn_gpu import HostValue, TinygradBackend, compile_bqn
from bqn_gpu.cbqn import CBQN
from bqn_gpu.json_values import decode_host_value


V = decode_host_value([3, 1, 2, 1])
M = decode_host_value([[1, 2, 3], [4, 5, 6]])


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
    backend: TinygradBackend,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(source)
    actual = compiled.execute(backend, **arguments)
    oracle_arguments = (
        (arguments["w"], arguments["x"])
        if compiled.arity == 2
        else (arguments["x"],)
    )
    expected = cbqn.call(source, *oracle_arguments)
    assert actual == expected
