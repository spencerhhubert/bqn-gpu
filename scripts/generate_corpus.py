#!/usr/bin/env python3
"""Generate the tracked initial BQN program corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import string
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bqn_gpu.ir import (  # noqa: E402
    Expression,
    argument,
    array_constant,
    constant,
    dyadic,
    fold,
    function_source,
    insert,
    monadic,
    render_bqn,
    scan,
)
from bqn_gpu.native_sources import render_native_source  # noqa: E402
from bqn_gpu.source import compile_bqn  # noqa: E402


DESTINATION = ROOT / "corpus" / "programs.json"


def make_programs() -> list[dict[str, Any]]:
    programs: list[dict[str, Any]] = []

    def add(
        identifier: str,
        category: str,
        variant: str,
        expression: Expression,
        arity: int,
        input_mode: str,
        domains: dict[str, str],
        tags: Iterable[str],
        *,
        source: str | None = None,
        native_expression: Expression | None = None,
        native_tinygrad: str | None = None,
        native_torch: str | None = None,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        native_ir = native_expression or expression
        programs.append(
            {
                "id": identifier,
                "category": category,
                "variant": variant,
                "arity": arity,
                "bqn": source or function_source(expression),
                "native": {
                    "expression": native_ir,
                    "tinygrad": native_tinygrad or render_native_source(
                        native_ir, "tinygrad", arity=arity, input_mode=input_mode
                    ),
                    "torch": native_torch or render_native_source(
                        native_ir, "torch", arity=arity, input_mode=input_mode
                    ),
                },
                "input_mode": input_mode,
                "domains": domains,
                "rtol": rtol,
                "atol": atol,
                "tags": sorted(set(tags)),
            }
        )

    x = argument("x")
    w = argument("w")

    unary_glyphs = [
        ("conjugate", "+", "signed", 0.0),
        ("negate", "-", "signed", 0.0),
        ("sign", "×", "signed", 0.0),
        ("reciprocal", "÷", "nonzero", 1e-15),
        ("exponential", "⋆", "signed", 2e-14),
        ("square_root", "√", "positive", 2e-15),
        ("floor", "⌊", "fractional", 0.0),
        ("ceiling", "⌈", "fractional", 0.0),
        ("absolute", "|", "signed", 0.0),
    ]
    for name, glyph, domain, tolerance in unary_glyphs:
        add(
            f"glyph.{name}.vector",
            "glyph",
            "primitive",
            monadic(glyph, x),
            1,
            "monadic_vector",
            {"x": domain},
            ["glyph", "monadic", name],
            rtol=tolerance,
            atol=tolerance,
        )

    dyadic_glyphs = [
        ("add", "+", "signed", "signed", 0.0),
        ("subtract", "-", "signed", "signed", 0.0),
        ("multiply", "×", "signed", "signed", 0.0),
        ("divide", "÷", "signed", "nonzero", 2e-15),
        ("modulus", "|", "nonzero", "signed", 3e-15),
        ("minimum", "⌊", "signed", "signed", 0.0),
        ("maximum", "⌈", "signed", "signed", 0.0),
        ("power", "⋆", "positive", "signed", 5e-14),
        ("root", "√", "positive", "positive", 5e-14),
        ("equal", "=", "signed", "signed", 0.0),
        ("not_equal", "≠", "signed", "signed", 0.0),
        ("less", "<", "signed", "signed", 0.0),
        ("greater", ">", "signed", "signed", 0.0),
        ("less_equal", "≤", "signed", "signed", 0.0),
        ("greater_equal", "≥", "signed", "signed", 0.0),
    ]
    modes = ("dyadic_same", "left_atom", "right_atom", "leading_left", "leading_right")
    for name, glyph, w_domain, x_domain, tolerance in dyadic_glyphs:
        for mode in modes:
            add(
                f"glyph.{name}.{mode}",
                "glyph",
                "primitive",
                dyadic(glyph, w, x),
                2,
                mode,
                {"w": w_domain, "x": x_domain},
                ["glyph", "dyadic", name, mode],
                rtol=tolerance,
                atol=tolerance,
            )

    def apply_steps(start: Expression, steps: list[tuple[str, str, int | None]]) -> Expression:
        result = start
        for kind, glyph, value in steps:
            if kind == "unary":
                result = monadic(glyph, result)
            elif kind == "right":
                assert value is not None
                result = dyadic(glyph, result, constant(value))
            elif kind == "left":
                assert value is not None
                result = dyadic(glyph, constant(value), result)
            else:
                raise ValueError(kind)
        return result

    safe_steps = [
        ("right", "+", 1),
        ("right", "×", 2),
        ("right", "-", 3),
        ("right", "÷", 2),
        ("unary", "|", None),
        ("right", "+", 4),
        ("right", "×", 3),
        ("right", "÷", 5),
        ("right", "-", 1),
        ("unary", "-", None),
        ("right", "+", 2),
        ("right", "×", 2),
    ]

    for index in range(20):
        rotated = safe_steps[index % len(safe_steps) :] + safe_steps[: index % len(safe_steps)]
        steps = rotated[: 3 + index % 7]
        expression = apply_steps(x, steps)
        add(
            f"phrase.unary_chain.{index + 1:03d}",
            "phrase",
            "composed",
            expression,
            1,
            "monadic_vector",
            {"x": "signed"},
            ["phrase", "elementwise", "fusible", f"depth-{len(steps)}"],
            rtol=2e-14,
            atol=2e-14,
        )

    dyadic_bases = [
        dyadic("+", w, x),
        dyadic("-", w, x),
        dyadic("×", w, x),
        dyadic("+", dyadic("×", w, x), x),
        dyadic("-", dyadic("×", w, x), w),
    ]
    for index in range(10):
        base = dyadic_bases[index % len(dyadic_bases)]
        steps = safe_steps[index : index + 4]
        expression = apply_steps(base, steps)
        add(
            f"phrase.dyadic_chain.{index + 1:03d}",
            "phrase",
            "composed",
            expression,
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
            ["phrase", "dyadic", "elementwise", "fusible"],
            rtol=3e-14,
            atol=3e-14,
        )

    comparisons = ("=", "≠", "<", ">", "≤", "≥")
    for index in range(10):
        comparison = dyadic(comparisons[index % len(comparisons)], w, x)
        expression = dyadic(
            "+",
            dyadic("×", comparison, x),
            dyadic("×", dyadic("=", comparison, constant(0)), w),
        )
        add(
            f"phrase.masked_select.{index + 1:03d}",
            "phrase",
            "composed",
            expression,
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
            ["phrase", "dyadic", "comparison", "masked", "fusible"],
            rtol=0.0,
            atol=0.0,
        )

    comparison_names = (
        ("equal", "="),
        ("not_equal", "≠"),
        ("less", "<"),
        ("greater", ">"),
        ("less_equal", "≤"),
        ("greater_equal", "≥"),
    )
    for name, glyph in comparison_names:
        add(
            f"phrase.compare_self.{name}",
            "phrase",
            "composed",
            dyadic(glyph, x, x),
            1,
            "monadic_vector",
            {"x": "signed"},
            ["phrase", "comparison", "self", name],
        )
        add(
            f"phrase.compare_zero.{name}",
            "phrase",
            "composed",
            dyadic(glyph, x, constant(0)),
            1,
            "monadic_vector",
            {"x": "signed"},
            ["phrase", "comparison", "boundary", name],
        )

    structural_glyphs = (("rank", "="), ("length", "≠"), ("shape", "≢"))
    structural_modes = (
        "monadic_atom",
        "monadic_rank_zero",
        "monadic_vector",
        "monadic_matrix",
        "monadic_empty_vector",
        "monadic_empty_matrix",
    )
    for name, glyph in structural_glyphs:
        for mode in structural_modes:
            add(
                f"glyph.{name}.{mode}",
                "glyph",
                "primitive",
                monadic(glyph, x),
                1,
                mode,
                {"x": "signed"},
                ["glyph", "monadic", "structural", name, mode],
            )

    add(
        "glyph.range.natural_atom",
        "glyph",
        "primitive",
        monadic("↕", x),
        1,
        "monadic_atom",
        {"x": "count"},
        ["glyph", "monadic", "structural", "range"],
    )
    for mode in ("monadic_vector", "monadic_matrix", "monadic_empty_vector"):
        add(
            f"phrase.major_cell_indices.{mode}",
            "phrase",
            "composed",
            monadic("↕", monadic("≠", x)),
            1,
            mode,
            {"x": "signed"},
            ["phrase", "structural", "range", "length", mode],
        )

    dense_monads = [
        ("sort_up", "∧", "monadic_vector"),
        ("sort_down", "∨", "monadic_vector"),
        ("not", "¬", "monadic_vector"),
        ("depth", "≡", "monadic_vector"),
        ("identity_left", "⊣", "monadic_vector"),
        ("identity_right", "⊢", "monadic_vector"),
        ("deshape", "⥊", "monadic_matrix"),
        ("solo", "≍", "monadic_vector"),
        ("reverse_vector", "⌽", "monadic_vector"),
        ("reverse_matrix", "⌽", "monadic_matrix"),
        ("transpose", "⍉", "monadic_matrix"),
        ("nudge_vector", "»", "monadic_vector"),
        ("nudge_matrix", "»", "monadic_matrix"),
        ("nudge_back_vector", "«", "monadic_vector"),
        ("nudge_back_matrix", "«", "monadic_matrix"),
        ("grade_up", "⍋", "monadic_vector"),
        ("grade_down", "⍒", "monadic_vector"),
        ("first_cell", "⊏", "monadic_matrix"),
        ("first", "⊑", "monadic_vector"),
        ("classify_major_cells", "⊐", "monadic_matrix"),
        ("occurrence_count_major_cells", "⊒", "monadic_matrix"),
        ("mark_firsts_major_cells", "∊", "monadic_matrix"),
        ("deduplicate_major_cells", "⍷", "monadic_matrix"),
    ]
    for name, glyph, mode in dense_monads:
        add(
            f"dense.{name}.{mode}",
            "dense-primitive",
            "primitive",
            monadic(glyph, x),
            1,
            mode,
            {"x": "signed"},
            ["dense", "glyph", "monadic", name, mode],
        )

    add(
        "dense.major_cells.unique_count",
        "dense-phrase",
        "idiomatic",
        fold("+", monadic("∊", x)),
        1,
        "monadic_matrix",
        {"x": "signed"},
        ["dense", "major-cell", "reduction", "self-search", "unique"],
        source="{+´∊𝕩}",
    )

    for name, glyph in (("and", "∧"), ("or", "∨"), ("span", "¬")):
        for mode in modes:
            add(
                f"dense.{name}.{mode}",
                "dense-primitive",
                "primitive",
                dyadic(glyph, w, x),
                2,
                mode,
                {"w": "signed", "x": "signed"},
                ["dense", "glyph", "dyadic", name, mode],
                rtol=2e-15 if glyph in {"∨", "¬"} else 0.0,
                atol=2e-15 if glyph in {"∨", "¬"} else 0.0,
            )

    for name, glyph in (("match_self", "≡"), ("not_match_self", "≢")):
        add(
            f"dense.{name}",
            "dense-primitive",
            "primitive",
            dyadic(glyph, x, x),
            1,
            "monadic_vector",
            {"x": "signed"},
            ["dense", "glyph", "dyadic", "match"],
        )

    for name, glyph in (("left", "⊣"), ("right", "⊢")):
        add(
            f"dense.identity_{name}.dyadic_same",
            "dense-primitive",
            "primitive",
            dyadic(glyph, w, x),
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
            ["dense", "glyph", "dyadic", "identity"],
        )

    for name, glyph in (("join_self", "∾"), ("couple_self", "≍")):
        add(
            f"dense.{name}",
            "dense-structural",
            "primitive",
            dyadic(glyph, x, x),
            1,
            "monadic_vector",
            {"x": "signed"},
            ["dense", "structural", name],
        )

    literal_structural = [
        ("drop_first", dyadic("↓", constant(1), x), "monadic_vector"),
        ("take_last", dyadic("↑", constant(-1), x), "monadic_vector"),
        ("rotate_one", dyadic("⌽", constant(1), x), "monadic_vector"),
        ("replicate_two", dyadic("/", constant(2), x), "monadic_vector"),
        ("windows_three", dyadic("↕", constant(3), x), "monadic_vector"),
        ("shift_before_one", dyadic("»", constant(1), x), "monadic_vector"),
        ("shift_after_one", dyadic("«", constant(1), x), "monadic_vector"),
        (
            "select_permutation",
            dyadic("⊏", array_constant([2, 0, 1]), x),
            "monadic_vector",
        ),
        ("pick_first", dyadic("⊑", constant(0), x), "monadic_vector"),
        (
            "transpose_axes",
            dyadic("⍉", array_constant([1, 0]), x),
            "monadic_matrix",
        ),
    ]
    for name, expression, mode in literal_structural:
        add(
            f"dense.{name}.{mode}",
            "dense-structural",
            "primitive",
            expression,
            1,
            mode,
            {"x": "signed"},
            ["dense", "structural", name, mode],
        )

    for name, glyph in (
        ("sum", "+"),
        ("product", "×"),
        ("and", "∧"),
        ("or", "∨"),
        ("minimum", "⌊"),
        ("maximum", "⌈"),
    ):
        insert_domain = "positive" if glyph in {"×", "∧", "∨"} else "signed"
        scan_domain = {
            "×": "near_one",
            "∧": "boolean",
            "∨": "boolean",
        }.get(glyph, "signed")
        add(
            f"dense.insert_{name}.matrix",
            "dense-modifier",
            "primitive",
            insert(glyph, x),
            1,
            "monadic_matrix",
            {"x": insert_domain},
            ["dense", "modifier", "insert", name],
            rtol=3e-12,
            atol=3e-12,
        )
        add(
            f"dense.scan_{name}.vector",
            "dense-modifier",
            "primitive",
            scan(glyph, x),
            1,
            "monadic_vector",
            {"x": scan_domain},
            ["dense", "modifier", "scan", name],
            rtol=2e-10 if glyph in {"+", "×"} else 0.0,
            atol=2e-10 if glyph in {"+", "×"} else 0.0,
        )

    dense_phrases = [
        ("sum_reverse", fold("+", monadic("⌽", x)), "monadic_vector"),
        ("sum_sort", fold("+", monadic("∧", x)), "monadic_vector"),
        ("sum_deshape", fold("+", monadic("⥊", x)), "monadic_matrix"),
        ("insert_transpose", insert("+", monadic("⍉", x)), "monadic_matrix"),
        ("reverse_scan", monadic("⌽", scan("+", x)), "monadic_vector"),
        ("scan_reverse", scan("+", monadic("⌽", x)), "monadic_vector"),
        ("sum_drop_first", fold("+", dyadic("↓", constant(1), x)), "monadic_vector"),
        ("sum_replicate", fold("+", dyadic("/", constant(2), x)), "monadic_vector"),
        ("insert_windows", insert("+", dyadic("↕", constant(3), x)), "monadic_vector"),
        (
            "count_nonzero",
            fold("+", monadic("¬", dyadic("=", x, constant(0)))),
            "monadic_vector",
        ),
    ]
    for name, expression, mode in dense_phrases:
        add(
            f"dense.phrase.{name}",
            "dense-phrase",
            "composed",
            expression,
            1,
            mode,
            {"x": "signed"},
            ["dense", "phrase", "structural", name],
            rtol=5e-12,
            atol=5e-12,
        )

    dense_combinators = [
        (
            "self_square",
            dyadic("×", x, x),
            "{×˜𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "swap_subtract",
            dyadic("-", x, w),
            "{𝕨-˜𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
        (
            "atop_absolute_negate",
            monadic("|", monadic("-", x)),
            "{|∘-𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "atop_absolute_difference",
            monadic("|", dyadic("-", w, x)),
            "{𝕨|∘-𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
        (
            "over_subtract_absolute",
            dyadic("-", monadic("|", w), monadic("|", x)),
            "{𝕨-○|𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
        (
            "before_exp_subtract",
            dyadic("-", monadic("⋆", w), x),
            "{𝕨⋆⊸-𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
        (
            "after_power_negate",
            dyadic("⋆", w, monadic("-", x)),
            "{𝕨⋆⟜-𝕩}",
            2,
            "dyadic_same",
            {"w": "positive", "x": "signed"},
        ),
        (
            "bind_rotate_right",
            dyadic("⌽", constant(-1), x),
            "{¯1⊸⌽𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "bind_subtract_one",
            dyadic("-", x, constant(1)),
            "{-⟜1 𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "atop_sum_absolute",
            fold("+", monadic("|", x)),
            "{+´∘|𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "atop_l1_distance",
            fold("+", monadic("|", dyadic("-", w, x))),
            "{𝕨+´∘|∘-𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
        (
            "valences_monadic_negate",
            monadic("-", x),
            "{-⊘+𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
        ),
        (
            "valences_dyadic_add",
            dyadic("+", w, x),
            "{𝕨-⊘+𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
        ),
    ]
    for name, expression, source, arity, mode, domains in dense_combinators:
        add(
            f"dense.combinator.{name}",
            "dense-combinator",
            "idiomatic",
            expression,
            arity,
            mode,
            domains,
            ["dense", "modifier", "combinator", name],
            source=source,
            rtol=5e-12,
            atol=5e-12,
        )

    train_mean = dyadic("÷", fold("+", x), monadic("≠", x))
    train_reverse_add = dyadic("+", x, monadic("⌽", x))
    dense_trains = [
        {
            "name": "reverse_add",
            "expression": train_reverse_add,
            "source": "{(⊢+⌽)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["3-train"],
        },
        {
            "name": "mean",
            "expression": train_mean,
            "source": "{(+´÷≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["3-train", "reduction"],
        },
        {
            "name": "l1_mean",
            "expression": dyadic(
                "÷", fold("+", monadic("|", x)), monadic("≠", x)
            ),
            "source": "{(+´∘|÷≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["3-train", "combinator", "reduction"],
        },
        {
            "name": "centered",
            "expression": dyadic("-", x, train_mean),
            "source": "{(⊢-+´÷≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["long-train", "reduction"],
        },
        {
            "name": "even_reverse_centered",
            "expression": monadic("⌽", dyadic("-", x, monadic("≠", x))),
            "source": "{(⌽⊢-≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["2-train", "long-train", "structural"],
        },
        {
            "name": "reverse_length_offset",
            "expression": dyadic(
                "+", x, dyadic("-", monadic("⌽", x), monadic("≠", x))
            ),
            "source": "{(⊢+⌽-≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["long-train", "structural"],
        },
        {
            "name": "nested_scaled_reverse_add",
            "expression": dyadic("×", train_reverse_add, monadic("≠", x)),
            "source": "{((⊢+⌽)×≠)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["nested-train", "structural"],
        },
        {
            "name": "reverse_transpose",
            "expression": monadic("⌽", monadic("⍉", x)),
            "source": "{(⌽⍉)𝕩}",
            "arity": 1,
            "mode": "monadic_matrix",
            "tags": ["2-train", "structural"],
        },
        {
            "name": "plus_minus_pair",
            "expression": dyadic("⋈", dyadic("+", w, x), dyadic("-", w, x)),
            "source": "{𝕨(+⋈-)𝕩}",
            "arity": 2,
            "mode": "dyadic_atoms",
            "tags": ["3-train", "dyadic"],
        },
        {
            "name": "sum_times_l1",
            "expression": dyadic(
                "×", fold("+", x), fold("+", monadic("|", x))
            ),
            "source": "{(+´×+´∘|)𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "tags": ["3-train", "combinator", "reduction"],
        },
    ]
    for item in dense_trains:
        add(
            f"dense.train.{item['name']}",
            "dense-train",
            "idiomatic",
            item["expression"],
            item["arity"],
            item["mode"],
            {name: "signed" for name in ("w", "x")[-item["arity"] :]},
            ["dense", "train", *item["tags"]],
            source=item["source"],
            rtol=5e-12,
            atol=5e-12,
        )

    repeated_add = x
    for _ in range(4):
        repeated_add = dyadic("+", constant(1), repeated_add)
    repeated_dyadic_add = x
    for _ in range(3):
        repeated_dyadic_add = dyadic("+", w, repeated_dyadic_add)
    centered_once = dyadic("-", x, train_mean)
    centered_twice = dyadic(
        "-",
        centered_once,
        dyadic("÷", fold("+", centered_once), monadic("≠", centered_once)),
    )
    dense_repeats = [
        ("identity_zero", x, "{-⍟0𝕩}", 1, "monadic_vector", ["zero"]),
        (
            "negate_twice",
            monadic("-", monadic("-", x)),
            "{-⍟2𝕩}",
            1,
            "monadic_vector",
            ["pervasive"],
        ),
        (
            "reverse_twice",
            monadic("⌽", monadic("⌽", x)),
            "{⌽⍟2𝕩}",
            1,
            "monadic_vector",
            ["structural"],
        ),
        (
            "absolute_three",
            monadic("|", monadic("|", monadic("|", x))),
            "{|⍟3𝕩}",
            1,
            "monadic_vector",
            ["pervasive"],
        ),
        (
            "bound_add_four",
            repeated_add,
            "{1⊸+⍟4𝕩}",
            1,
            "monadic_vector",
            ["bind", "pervasive"],
        ),
        (
            "dyadic_add_three",
            repeated_dyadic_add,
            "{𝕨+⍟3𝕩}",
            2,
            "dyadic_same",
            ["dyadic", "pervasive"],
        ),
        (
            "centered_twice",
            centered_twice,
            "{(⊢-+´÷≠)⍟2𝕩}",
            1,
            "monadic_vector",
            ["train", "reduction"],
        ),
    ]
    for name, expression, source, arity, mode, tags in dense_repeats:
        add(
            f"dense.repeat.{name}",
            "dense-repeat",
            "idiomatic",
            expression,
            arity,
            mode,
            {argument_name: "signed" for argument_name in ("w", "x")[-arity:]},
            ["dense", "repeat", "static-count", *tags],
            source=source,
            rtol=5e-12,
            atol=5e-12,
        )

    long_undo = x
    for glyph, value in (("÷", 2), ("-", 1), ("÷", 3), ("-", 4), ("÷", 5), ("-", 6), ("÷", 7), ("-", 8)):
        long_undo = dyadic(glyph, long_undo, constant(value))
    dense_undos = [
        ("identity", x, "{+⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["primitive"], 0.0),
        ("negate", monadic("-", x), "{-⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["primitive"], 0.0),
        ("logarithm", monadic("⋆⁼", x), "{⋆⁼𝕩}", 1, "monadic_vector", {"x": "positive"}, ["primitive", "transcendental"], 5e-14),
        ("square", dyadic("×", x, x), "{√⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["primitive", "self"], 0.0),
        ("reverse", monadic("⌽", x), "{⌽⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["primitive", "structural"], 0.0),
        ("dyadic_add", dyadic("-", x, w), "{𝕨+⁼𝕩}", 2, "dyadic_same", {"w": "signed", "x": "signed"}, ["dyadic"], 0.0),
        ("dyadic_multiply", dyadic("÷", x, w), "{𝕨×⁼𝕩}", 2, "dyadic_same", {"w": "nonzero", "x": "signed"}, ["dyadic"], 5e-14),
        ("bound_scale", dyadic("÷", x, constant(2)), "{2⊸×⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["bind"], 0.0),
        ("self_square", monadic("√", x), "{×˜⁼𝕩}", 1, "monadic_vector", {"x": "positive"}, ["self"], 5e-14),
        ("each_negate", monadic("-", x), "{-¨⁼𝕩}", 1, "monadic_vector", {"x": "signed"}, ["mapping"], 0.0),
        (
            "long_composition",
            long_undo,
            "{((2⊸×)∘(1⊸+)∘(3⊸×)∘(4⊸+)∘(5⊸×)∘(6⊸+)∘(7⊸×)∘(8⊸+))⁼𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
            ["atop", "bind", "long", "program"],
            5e-14,
        ),
    ]
    for name, expression, source, arity, mode, domains, tags, tolerance in dense_undos:
        add(
            f"dense.undo.{name}",
            "dense-undo",
            "idiomatic",
            expression,
            arity,
            mode,
            domains,
            ["dense", "modifier", "undo", *tags],
            source=source,
            rtol=tolerance,
            atol=tolerance,
        )

    log_w = monadic("⋆⁼", w)
    log_x = monadic("⋆⁼", x)
    dense_unders = [
        (
            "log_add",
            monadic("⋆", dyadic("+", log_w, log_x)),
            "{𝕨+⌾(⋆⁼)𝕩}",
            2,
            "dyadic_same",
            {"w": "positive", "x": "positive"},
            ["dyadic", "log-domain", "transcendental"],
            8e-14,
        ),
        (
            "log_subtract",
            monadic("⋆", dyadic("-", log_w, log_x)),
            "{𝕨-⌾(⋆⁼)𝕩}",
            2,
            "dyadic_same",
            {"w": "positive", "x": "positive"},
            ["dyadic", "log-domain", "transcendental"],
            8e-14,
        ),
        (
            "l2_combine",
            monadic("√", dyadic("+", dyadic("×", w, w), dyadic("×", x, x))),
            "{𝕨+⌾(×˜)𝕩}",
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
            ["dyadic", "norm", "self"],
            8e-14,
        ),
        (
            "add_three_under_square",
            monadic("√", dyadic("+", constant(9), dyadic("×", x, x))),
            "{3+⌾(×˜)𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
            ["literal-left", "norm", "self"],
            8e-14,
        ),
        (
            "reverse_negate",
            monadic("⌽", monadic("-", monadic("⌽", x))),
            "{-⌾⌽𝕩}",
            1,
            "monadic_vector",
            {"x": "signed"},
            ["structural", "pervasive"],
            0.0,
        ),
    ]
    for name, expression, source, arity, mode, domains, tags, tolerance in dense_unders:
        add(
            f"dense.under.{name}",
            "dense-under",
            "idiomatic",
            expression,
            arity,
            mode,
            domains,
            ["combinator", "dense", "under", *tags],
            source=source,
            rtol=tolerance,
            atol=tolerance,
        )

    dense_mapping = [
        {
            "name": "each_negate",
            "expression": monadic("-", x),
            "source": "{-¨𝕩}",
            "arity": 1,
            "mode": "monadic_vector",
            "domains": {"x": "signed"},
        },
        {
            "name": "each_leading_add",
            "expression": dyadic("+", w, x),
            "source": "{𝕨+¨𝕩}",
            "arity": 2,
            "mode": "leading_left",
            "domains": {"w": "signed", "x": "signed"},
        },
        {
            "name": "table_multiply",
            "expression": dyadic("×", w, x),
            "source": "{𝕨×⌜𝕩}",
            "arity": 2,
            "mode": "table_vectors",
            "domains": {"w": "signed", "x": "signed"},
            "native_tinygrad": "lambda w, x: w.reshape((w.shape[0], 1)) * x.reshape((1, x.shape[0]))",
            "native_torch": "lambda w, x: w.reshape((w.shape[0], 1)) * x.reshape((1, x.shape[0]))",
        },
        {
            "name": "cells_reverse",
            "expression": monadic("⌽", x),
            "source": "{⌽˘𝕩}",
            "arity": 1,
            "mode": "monadic_matrix",
            "domains": {"x": "signed"},
            "native_tinygrad": "lambda x: x.flip(1)",
            "native_torch": "lambda x: torch.flip(x, dims=(1,))",
        },
        {
            "name": "cells_sum_absolute",
            "expression": fold("+", monadic("|", x)),
            "source": "{+´∘|˘𝕩}",
            "arity": 1,
            "mode": "monadic_matrix",
            "domains": {"x": "signed"},
            "native_tinygrad": "lambda x: x.abs().sum(axis=1)",
            "native_torch": "lambda x: x.abs().sum(dim=1)",
        },
        {
            "name": "rank_add_rows",
            "expression": dyadic("+", w, x),
            "source": "{𝕨+⎉1‿1𝕩}",
            "arity": 2,
            "mode": "matrix_vector",
            "domains": {"w": "signed", "x": "signed"},
            "native_tinygrad": "lambda w, x: w + x.reshape((1, x.shape[0]))",
            "native_torch": "lambda w, x: w + x.reshape((1, x.shape[0]))",
        },
        {
            "name": "rank_matrix_vector",
            "expression": fold("+", dyadic("×", w, x)),
            "source": "{𝕨+˝∘×⎉1𝕩}",
            "arity": 2,
            "mode": "matrix_vector",
            "domains": {"w": "signed", "x": "signed"},
            "native_tinygrad": "lambda w, x: (w * x.reshape((1, x.shape[0]))).sum(axis=1)",
            "native_torch": "lambda w, x: (w * x.reshape((1, x.shape[0]))).sum(dim=1)",
        },
    ]
    for item in dense_mapping:
        add(
            f"dense.mapping.{item['name']}",
            "dense-mapping",
            "idiomatic",
            item["expression"],
            item["arity"],
            item["mode"],
            item["domains"],
            ["dense", "modifier", "mapping", item["name"]],
            source=item["source"],
            native_expression=(
                compile_bqn(item["source"]).expression
                if "native_tinygrad" in item
                else None
            ),
            native_tinygrad=item.get("native_tinygrad"),
            native_torch=item.get("native_torch"),
            rtol=5e-12,
            atol=5e-12,
        )

    dense_pairs = [
        ("double_reverse", x, "{⌽⌽𝕩}", "monadic_vector"),
        ("cancel_rotate", x, "{1⌽¯1⌽𝕩}", "monadic_vector"),
        ("double_transpose", x, "{⍉⍉𝕩}", "monadic_matrix"),
        ("sort_up", monadic("∧", x), "{⌽∨𝕩}", "monadic_vector"),
        ("sort_down", monadic("∨", x), "{⌽∧𝕩}", "monadic_vector"),
    ]
    for name, expression, naive, mode in dense_pairs:
        common = {
            "category": "dense-paired",
            "expression": expression,
            "arity": 1,
            "input_mode": mode,
            "domains": {"x": "signed"},
            "tags": ["dense", "paired", "structural", name],
        }
        add(
            f"dense.pair.{name}.naive",
            variant="naive",
            source=naive,
            **common,
        )
        add(
            f"dense.pair.{name}.idiomatic",
            variant="idiomatic",
            **common,
        )

    def naive_source(steps: list[tuple[str, str, int | None]]) -> str:
        names = iter(string.ascii_lowercase)
        previous = next(names)
        statements = [f"{previous}←𝕩"]
        for kind, glyph, value in steps:
            current = next(names)
            if kind == "unary":
                rhs = f"{glyph}{previous}"
            elif kind == "right":
                rhs = f"{previous}{glyph}{value}"
            elif kind == "left":
                rhs = f"{value}{glyph}{previous}"
            else:
                raise ValueError(kind)
            statements.append(f"{current}←{rhs}")
            previous = current
        statements.append(previous)
        return "{" + " ⋄ ".join(statements) + "}"

    for index in range(12):
        rotated = safe_steps[index:] + safe_steps[:index]
        steps = (rotated * 2)[: 8 + index % 5]
        expression = apply_steps(x, steps)
        common = {
            "category": "paired",
            "expression": expression,
            "arity": 1,
            "input_mode": "monadic_vector",
            "domains": {"x": "signed"},
            "tags": ["paired", "long", "elementwise", "fusible", f"pair-{index + 1:02d}"],
            "rtol": 5e-14,
            "atol": 5e-14,
        }
        add(
            f"pair.pipeline.{index + 1:02d}.naive",
            variant="naive",
            source=naive_source(steps),
            **common,
        )
        add(
            f"pair.pipeline.{index + 1:02d}.idiomatic",
            variant="idiomatic",
            **common,
        )

    reductions = [
        ("sum", fold("+", x), "signed"),
        ("product", fold("×", x), "positive"),
        ("sum_abs", fold("+", monadic("|", x)), "signed"),
        ("sum_squares", fold("+", dyadic("×", x, x)), "signed"),
        ("sum_cubes", fold("+", dyadic("×", dyadic("×", x, x), x)), "signed"),
        (
            "sum_affine",
            fold("+", dyadic("+", dyadic("×", x, constant(3)), constant(2))),
            "signed",
        ),
        ("minimum", fold("⌊", x), "signed"),
        ("maximum", fold("⌈", x), "signed"),
        ("sum_reciprocal", fold("+", monadic("÷", x)), "positive"),
    ]
    for name, expression, domain in reductions:
        add(
            f"reduction.{name}",
            "reduction",
            "idiomatic",
            expression,
            1,
            "monadic_vector",
            {"x": domain},
            ["reduction", name],
            rtol=3e-12,
            atol=3e-12,
        )

    dyadic_reductions = [
        ("dot", fold("+", dyadic("×", w, x))),
        ("l1_distance", fold("+", monadic("|", dyadic("-", w, x)))),
        ("squared_distance", fold("+", dyadic("×", dyadic("-", w, x), dyadic("-", w, x)))),
        ("sum_pair_affine", fold("+", dyadic("+", dyadic("×", w, constant(2)), x))),
    ]
    for name, expression in dyadic_reductions:
        add(
            f"reduction.{name}",
            "reduction",
            "idiomatic",
            expression,
            2,
            "dyadic_same",
            {"w": "signed", "x": "signed"},
            ["reduction", "dyadic", name],
            rtol=5e-12,
            atol=5e-12,
        )

    class LongProgram:
        """Build readable multi-stage BQN and its independent workload IR together."""

        def __init__(self, arity: int) -> None:
            self.arity = arity
            self.expressions: dict[str, Expression] = {"x": x}
            self.sources = {"x": "𝕩"}
            if arity == 2:
                self.expressions["w"] = w
                self.sources["w"] = "𝕨"
            self.statements: list[str] = []

        def value(self, item: str | int | float) -> tuple[str, Expression]:
            if isinstance(item, str):
                return self.sources[item], self.expressions[item]
            expression = constant(item)
            return render_bqn(expression), expression

        def bind(self, name: str, source: str, expression: Expression) -> str:
            if name in self.expressions:
                raise AssertionError(f"duplicate long-program binding {name!r}")
            self.sources[name] = name
            self.expressions[name] = expression
            self.statements.append(f"{name}←{source}")
            return name

        def monad(self, name: str, glyph: str, item: str | int | float) -> str:
            source, expression = self.value(item)
            return self.bind(name, f"{glyph}{source}", monadic(glyph, expression))

        def dyad(
            self,
            name: str,
            left: str | int | float,
            glyph: str,
            right: str | int | float,
        ) -> str:
            left_source, left_expression = self.value(left)
            right_source, right_expression = self.value(right)
            return self.bind(
                name,
                f"{left_source}{glyph}{right_source}",
                dyadic(glyph, left_expression, right_expression),
            )

        def fold(self, name: str, glyph: str, item: str) -> str:
            source, expression = self.value(item)
            return self.bind(name, f"{glyph}´{source}", fold(glyph, expression))

        def insert(self, name: str, glyph: str, item: str) -> str:
            source, expression = self.value(item)
            return self.bind(name, f"{glyph}˝{source}", insert(glyph, expression))

        def scan(self, name: str, glyph: str, item: str) -> str:
            source, expression = self.value(item)
            return self.bind(name, f"{glyph}`{source}", scan(glyph, expression))

        def source(self, result: str) -> str:
            if result not in self.expressions:
                raise AssertionError(f"unknown long-program result {result!r}")
            return "{" + " ⋄ ".join([*self.statements, result]) + "}"

    def add_long(
        identifier: str,
        builder: LongProgram,
        result: str,
        input_mode: str,
        domains: dict[str, str],
        tags: Iterable[str],
        *,
        tolerance: float = 2e-9,
    ) -> None:
        add(
            identifier,
            "long-algorithm",
            "algorithmic",
            builder.expressions[result],
            builder.arity,
            input_mode,
            domains,
            ["algorithm", "long", *tags],
            source=builder.source(result),
            rtol=tolerance,
            atol=tolerance,
        )

    statistics = LongProgram(1)
    statistics.fold("total", "+", "x")
    statistics.monad("count", "≠", "x")
    statistics.dyad("mean", "total", "÷", "count")
    statistics.dyad("center", "x", "-", "mean")
    statistics.monad("distance", "|", "center")
    statistics.fold("lone", "+", "distance")
    statistics.dyad("square", "center", "×", "center")
    statistics.fold("energy", "+", "square")
    statistics.dyad("variance", "energy", "÷", "count")
    statistics.dyad("safevar", "variance", "+", 1)
    statistics.monad("scale", "√", "safevar")
    statistics.dyad("normal", "center", "÷", "scale")
    statistics.monad("absnormal", "|", "normal")
    statistics.fold("normlone", "+", "absnormal")
    statistics.dyad("cube", "square", "×", "center")
    statistics.fold("third", "+", "cube")
    statistics.dyad("fourth", "square", "×", "square")
    statistics.fold("fourthsum", "+", "fourth")
    statistics.dyad("safedenom", "energy", "+", 1)
    statistics.dyad("skewproxy", "third", "÷", "safedenom")
    statistics.dyad("kurtproxy", "fourthsum", "÷", "safedenom")
    statistics.dyad("combined", "normlone", "+", "skewproxy")
    statistics.dyad("score", "combined", "+", "kurtproxy")
    add_long(
        "program.statistics.standardized_moments",
        statistics,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "normalization", "moments", "reduction"],
    )

    softmax = LongProgram(1)
    softmax.fold("maximum", "⌈", "x")
    softmax.dyad("shifted", "x", "-", "maximum")
    softmax.monad("weights", "⋆", "shifted")
    softmax.fold("denom", "+", "weights")
    softmax.dyad("probability", "weights", "÷", "denom")
    softmax.dyad("square", "probability", "×", "probability")
    softmax.dyad("cube", "square", "×", "probability")
    softmax.fold("secondmass", "+", "square")
    softmax.fold("thirdmass", "+", "cube")
    softmax.monad("logprob", "⋆⁼", "probability")
    softmax.dyad("plogp", "probability", "×", "logprob")
    softmax.fold("rawentropy", "+", "plogp")
    softmax.monad("entropy", "-", "rawentropy")
    softmax.fold("peak", "⌈", "probability")
    softmax.fold("floorprob", "⌊", "probability")
    softmax.dyad("spread", "peak", "-", "floorprob")
    softmax.dyad("concentration", "secondmass", "+", "thirdmass")
    softmax.dyad("summary", "entropy", "+", "concentration")
    softmax.dyad("score", "summary", "+", "spread")
    add_long(
        "program.statistics.softmax_concentration",
        softmax,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "softmax", "entropy", "reduction", "transcendental"],
    )

    robust = LongProgram(1)
    robust.fold("total", "+", "x")
    robust.monad("count", "≠", "x")
    robust.dyad("mean", "total", "÷", "count")
    robust.dyad("center", "x", "-", "mean")
    robust.monad("deviation", "|", "center")
    robust.fold("devsum", "+", "deviation")
    robust.dyad("mad", "devsum", "÷", "count")
    robust.dyad("threshold", "mad", "+", 0.5)
    robust.dyad("inlier", "deviation", "≤", "threshold")
    robust.monad("outlier", "¬", "inlier")
    robust.dyad("inside", "center", "×", "inlier")
    robust.dyad("outside", "center", "×", "outlier")
    robust.dyad("insquare", "inside", "×", "inside")
    robust.dyad("outsquare", "outside", "×", "outside")
    robust.fold("inenergy", "+", "insquare")
    robust.fold("outenergy", "+", "outsquare")
    robust.fold("outcount", "+", "outlier")
    robust.dyad("saferatio", "outenergy", "÷", "inenergy")
    robust.dyad("ratio", "saferatio", "÷", "threshold")
    robust.dyad("summary", "ratio", "+", "outcount")
    robust.dyad("score", "summary", "+", "mad")
    add_long(
        "program.statistics.robust_dispersion",
        robust,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "mask", "robust", "reduction"],
        tolerance=5e-9,
    )

    winsor = LongProgram(1)
    winsor.fold("minimum", "⌊", "x")
    winsor.fold("maximum", "⌈", "x")
    winsor.dyad("width", "maximum", "-", "minimum")
    winsor.dyad("safewidth", "width", "+", 1)
    winsor.dyad("shifted", "x", "-", "minimum")
    winsor.dyad("unit", "shifted", "÷", "safewidth")
    winsor.dyad("lowmask", "unit", "<", 0.1)
    winsor.dyad("lowdelta", 0.1, "-", "unit")
    winsor.dyad("lowadjust", "lowdelta", "×", "lowmask")
    winsor.dyad("lowered", "unit", "+", "lowadjust")
    winsor.dyad("highmask", "lowered", ">", 0.9)
    winsor.dyad("highdelta", "lowered", "-", 0.9)
    winsor.dyad("highadjust", "highdelta", "×", "highmask")
    winsor.dyad("clipped", "lowered", "-", "highadjust")
    winsor.dyad("center", "clipped", "-", 0.5)
    winsor.monad("distance", "|", "center")
    winsor.dyad("square", "center", "×", "center")
    winsor.dyad("cube", "square", "×", "center")
    winsor.fold("lone", "+", "distance")
    winsor.fold("energy", "+", "square")
    winsor.fold("third", "+", "cube")
    winsor.monad("count", "≠", "x")
    winsor.dyad("meanenergy", "energy", "÷", "count")
    winsor.dyad("meanlone", "lone", "÷", "count")
    winsor.dyad("summary", "meanenergy", "+", "meanlone")
    winsor.dyad("score", "summary", "+", "third")
    add_long(
        "program.statistics.winsorized_shape",
        winsor,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "clamp", "normalization", "moments"],
    )

    positive_stats = LongProgram(1)
    positive_stats.fold("total", "+", "x")
    positive_stats.monad("count", "≠", "x")
    positive_stats.dyad("mean", "total", "÷", "count")
    positive_stats.dyad("center", "x", "-", "mean")
    positive_stats.dyad("square", "center", "×", "center")
    positive_stats.fold("energy", "+", "square")
    positive_stats.dyad("variance", "energy", "÷", "count")
    positive_stats.dyad("safevariance", "variance", "+", 1)
    positive_stats.monad("deviation", "√", "safevariance")
    positive_stats.dyad("coefficient", "deviation", "÷", "mean")
    positive_stats.monad("reciprocal", "÷", "x")
    positive_stats.fold("recipsum", "+", "reciprocal")
    positive_stats.dyad("harmonic", "count", "÷", "recipsum")
    positive_stats.monad("logvalues", "⋆⁼", "x")
    positive_stats.fold("logsum", "+", "logvalues")
    positive_stats.dyad("logmean", "logsum", "÷", "count")
    positive_stats.monad("geometric", "⋆", "logmean")
    positive_stats.dyad("spreadone", "mean", "-", "harmonic")
    positive_stats.dyad("spreadtwo", "geometric", "-", "harmonic")
    positive_stats.dyad("combined", "coefficient", "+", "spreadone")
    positive_stats.dyad("score", "combined", "+", "spreadtwo")
    add_long(
        "program.statistics.mean_family",
        positive_stats,
        "score",
        "monadic_vector",
        {"x": "positive"},
        ["statistics", "harmonic-mean", "geometric-mean", "transcendental"],
    )

    polynomial = LongProgram(1)
    polynomial.dyad("square", "x", "×", "x")
    polynomial.dyad("cube", "square", "×", "x")
    polynomial.dyad("fourth", "square", "×", "square")
    polynomial.dyad("fifth", "fourth", "×", "x")
    polynomial.dyad("sixth", "cube", "×", "cube")
    polynomial.dyad("termone", "x", "×", 0.5)
    polynomial.dyad("termtwo", "square", "÷", 2)
    polynomial.dyad("termthree", "cube", "÷", 6)
    polynomial.dyad("termfour", "fourth", "÷", 24)
    polynomial.dyad("termfive", "fifth", "÷", 120)
    polynomial.dyad("termsix", "sixth", "÷", 720)
    polynomial.dyad("partialone", "termone", "+", "termtwo")
    polynomial.dyad("partialtwo", "termthree", "+", "termfour")
    polynomial.dyad("partialthree", "termfive", "+", "termsix")
    polynomial.dyad("seriesone", "partialone", "+", "partialtwo")
    polynomial.dyad("series", "seriesone", "+", "partialthree")
    polynomial.monad("magnitude", "|", "series")
    polynomial.dyad("energy", "series", "×", "series")
    polynomial.fold("lone", "+", "magnitude")
    polynomial.fold("ltwo", "+", "energy")
    polynomial.fold("peak", "⌈", "magnitude")
    polynomial.dyad("summary", "lone", "+", "ltwo")
    polynomial.dyad("score", "summary", "+", "peak")
    add_long(
        "program.statistics.polynomial_moments",
        polynomial,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "polynomial", "moments", "fusible"],
        tolerance=2e-8,
    )

    threshold = LongProgram(1)
    threshold.dyad("positive", "x", ">", 0)
    threshold.dyad("negative", "x", "<", 0)
    threshold.monad("magnitude", "|", "x")
    threshold.dyad("positivevalue", "magnitude", "×", "positive")
    threshold.dyad("negativevalue", "magnitude", "×", "negative")
    threshold.dyad("square", "x", "×", "x")
    threshold.dyad("positiveenergy", "square", "×", "positive")
    threshold.dyad("negativeenergy", "square", "×", "negative")
    threshold.fold("positivecount", "+", "positive")
    threshold.fold("negativecount", "+", "negative")
    threshold.fold("positivesum", "+", "positivevalue")
    threshold.fold("negativesum", "+", "negativevalue")
    threshold.fold("posenergy", "+", "positiveenergy")
    threshold.fold("negenergy", "+", "negativeenergy")
    threshold.dyad("safeposcount", "positivecount", "+", 1)
    threshold.dyad("safenegcount", "negativecount", "+", 1)
    threshold.dyad("posmean", "positivesum", "÷", "safeposcount")
    threshold.dyad("negmean", "negativesum", "÷", "safenegcount")
    threshold.dyad("countbalance", "positivecount", "-", "negativecount")
    threshold.dyad("meanbalance", "posmean", "-", "negmean")
    threshold.dyad("energybalance", "posenergy", "-", "negenergy")
    threshold.dyad("summary", "countbalance", "+", "meanbalance")
    threshold.dyad("score", "summary", "+", "energybalance")
    add_long(
        "program.statistics.threshold_balance",
        threshold,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["statistics", "mask", "comparison", "reduction"],
    )

    log_shape = LongProgram(1)
    log_shape.monad("logs", "⋆⁼", "x")
    log_shape.fold("logtotal", "+", "logs")
    log_shape.monad("count", "≠", "x")
    log_shape.dyad("logmean", "logtotal", "÷", "count")
    log_shape.dyad("center", "logs", "-", "logmean")
    log_shape.monad("abscen", "|", "center")
    log_shape.dyad("square", "center", "×", "center")
    log_shape.dyad("cube", "square", "×", "center")
    log_shape.dyad("fourth", "square", "×", "square")
    log_shape.fold("lone", "+", "abscen")
    log_shape.fold("energy", "+", "square")
    log_shape.fold("third", "+", "cube")
    log_shape.fold("fourthsum", "+", "fourth")
    log_shape.dyad("safeenergy", "energy", "+", 1)
    log_shape.dyad("skew", "third", "÷", "safeenergy")
    log_shape.dyad("kurt", "fourthsum", "÷", "safeenergy")
    log_shape.monad("geometric", "⋆", "logmean")
    log_shape.dyad("summaryone", "lone", "+", "skew")
    log_shape.dyad("summarytwo", "kurt", "+", "geometric")
    log_shape.dyad("score", "summaryone", "+", "summarytwo")
    add_long(
        "program.statistics.log_shape",
        log_shape,
        "score",
        "monadic_vector",
        {"x": "positive"},
        ["statistics", "log-domain", "moments", "transcendental"],
        tolerance=5e-9,
    )

    def distance_prefix(builder: LongProgram) -> None:
        builder.dyad("difference", "w", "-", "x")
        builder.monad("absolute", "|", "difference")
        builder.dyad("square", "difference", "×", "difference")
        builder.fold("lone", "+", "absolute")
        builder.fold("ltwo", "+", "square")
        builder.monad("count", "≠", "x")
        builder.dyad("mae", "lone", "÷", "count")
        builder.dyad("mse", "ltwo", "÷", "count")

    correlation = LongProgram(2)
    correlation.fold("wtotal", "+", "w")
    correlation.fold("xtotal", "+", "x")
    correlation.monad("count", "≠", "x")
    correlation.dyad("wmean", "wtotal", "÷", "count")
    correlation.dyad("xmean", "xtotal", "÷", "count")
    correlation.dyad("wc", "w", "-", "wmean")
    correlation.dyad("xc", "x", "-", "xmean")
    correlation.dyad("cross", "wc", "×", "xc")
    correlation.dyad("wsquare", "wc", "×", "wc")
    correlation.dyad("xsquare", "xc", "×", "xc")
    correlation.fold("covsum", "+", "cross")
    correlation.fold("wenergy", "+", "wsquare")
    correlation.fold("xenergy", "+", "xsquare")
    correlation.dyad("energymul", "wenergy", "×", "xenergy")
    correlation.dyad("safeenergy", "energymul", "+", 1)
    correlation.monad("denom", "√", "safeenergy")
    correlation.dyad("corr", "covsum", "÷", "denom")
    correlation.dyad("residual", "wc", "-", "xc")
    correlation.monad("absresid", "|", "residual")
    correlation.dyad("residsq", "residual", "×", "residual")
    correlation.fold("residlone", "+", "absresid")
    correlation.fold("residltwo", "+", "residsq")
    correlation.dyad("summary", "corr", "+", "residlone")
    correlation.dyad("score", "summary", "+", "residltwo")
    add_long(
        "program.similarity.correlation_residual",
        correlation,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "correlation", "normalization", "reduction"],
        tolerance=5e-9,
    )

    cosine = LongProgram(2)
    cosine.dyad("product", "w", "×", "x")
    cosine.dyad("wsquare", "w", "×", "w")
    cosine.dyad("xsquare", "x", "×", "x")
    cosine.fold("dot", "+", "product")
    cosine.fold("wenergy", "+", "wsquare")
    cosine.fold("xenergy", "+", "xsquare")
    cosine.dyad("normproduct", "wenergy", "×", "xenergy")
    cosine.dyad("safenorm", "normproduct", "+", 1)
    cosine.monad("denom", "√", "safenorm")
    cosine.dyad("similarity", "dot", "÷", "denom")
    cosine.dyad("difference", "w", "-", "x")
    cosine.monad("absolute", "|", "difference")
    cosine.dyad("square", "difference", "×", "difference")
    cosine.fold("lone", "+", "absolute")
    cosine.fold("ltwo", "+", "square")
    cosine.monad("count", "≠", "x")
    cosine.dyad("mae", "lone", "÷", "count")
    cosine.dyad("mse", "ltwo", "÷", "count")
    cosine.dyad("error", 1, "-", "similarity")
    cosine.dyad("summary", "error", "+", "mae")
    cosine.dyad("score", "summary", "+", "mse")
    add_long(
        "program.similarity.cosine_error",
        cosine,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "cosine", "distance", "reduction"],
        tolerance=5e-9,
    )

    normalized_error = LongProgram(2)
    distance_prefix(normalized_error)
    normalized_error.monad("wabs", "|", "w")
    normalized_error.monad("xabs", "|", "x")
    normalized_error.dyad("magnitude", "wabs", "+", "xabs")
    normalized_error.dyad("safemag", "magnitude", "+", 1)
    normalized_error.dyad("relative", "absolute", "÷", "safemag")
    normalized_error.dyad("relsquare", "relative", "×", "relative")
    normalized_error.fold("relsum", "+", "relative")
    normalized_error.fold("relenergy", "+", "relsquare")
    normalized_error.fold("relpeak", "⌈", "relative")
    normalized_error.dyad("relmean", "relsum", "÷", "count")
    normalized_error.dyad("relmse", "relenergy", "÷", "count")
    normalized_error.dyad("summaryone", "mae", "+", "mse")
    normalized_error.dyad("summarytwo", "relmean", "+", "relmse")
    normalized_error.dyad("combined", "summaryone", "+", "summarytwo")
    normalized_error.dyad("score", "combined", "+", "relpeak")
    add_long(
        "program.similarity.normalized_error",
        normalized_error,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "relative-error", "distance", "reduction"],
    )

    smooth_lone = LongProgram(2)
    distance_prefix(smooth_lone)
    smooth_lone.dyad("small", "absolute", "≤", 1)
    smooth_lone.monad("large", "¬", "small")
    smooth_lone.dyad("half", "square", "÷", 2)
    smooth_lone.dyad("linear", "absolute", "-", 0.5)
    smooth_lone.dyad("smallloss", "half", "×", "small")
    smooth_lone.dyad("largeloss", "linear", "×", "large")
    smooth_lone.dyad("loss", "smallloss", "+", "largeloss")
    smooth_lone.fold("losssum", "+", "loss")
    smooth_lone.dyad("lossmean", "losssum", "÷", "count")
    smooth_lone.dyad("weighted", "loss", "×", "absolute")
    smooth_lone.fold("weightedtotal", "+", "weighted")
    smooth_lone.fold("maximum", "⌈", "loss")
    smooth_lone.dyad("summary", "lossmean", "+", "weightedtotal")
    smooth_lone.dyad("score", "summary", "+", "maximum")
    add_long(
        "program.similarity.smooth_l1_profile",
        smooth_lone,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "smooth-l1", "mask", "reduction"],
    )

    agreement = LongProgram(2)
    distance_prefix(agreement)
    agreement.dyad("near", "absolute", "≤", 0.25)
    agreement.dyad("mediumraw", "absolute", "≤", 1)
    agreement.dyad("medium", "mediumraw", "-", "near")
    agreement.monad("far", "¬", "mediumraw")
    agreement.fold("nearcount", "+", "near")
    agreement.fold("mediumcount", "+", "medium")
    agreement.fold("farcount", "+", "far")
    agreement.dyad("nearweight", "absolute", "×", "near")
    agreement.dyad("mediumweight", "absolute", "×", "medium")
    agreement.dyad("farweight", "absolute", "×", "far")
    agreement.fold("nearsum", "+", "nearweight")
    agreement.fold("mediumsum", "+", "mediumweight")
    agreement.fold("farsum", "+", "farweight")
    agreement.dyad("countscore", "nearcount", "+", "mediumcount")
    agreement.dyad("countscoretwo", "countscore", "+", "farcount")
    agreement.dyad("weightsum", "nearsum", "+", "mediumsum")
    agreement.dyad("weightsumtwo", "weightsum", "+", "farsum")
    agreement.dyad("summary", "countscoretwo", "+", "weightsumtwo")
    agreement.dyad("score", "summary", "+", "mse")
    add_long(
        "program.similarity.agreement_bands",
        agreement,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "comparison", "mask", "bands"],
    )

    log_distance = LongProgram(2)
    log_distance.monad("wlog", "⋆⁼", "w")
    log_distance.monad("xlog", "⋆⁼", "x")
    log_distance.dyad("difference", "wlog", "-", "xlog")
    log_distance.monad("absolute", "|", "difference")
    log_distance.dyad("square", "difference", "×", "difference")
    log_distance.dyad("cube", "square", "×", "difference")
    log_distance.fold("lone", "+", "absolute")
    log_distance.fold("ltwo", "+", "square")
    log_distance.fold("third", "+", "cube")
    log_distance.monad("count", "≠", "x")
    log_distance.dyad("mae", "lone", "÷", "count")
    log_distance.dyad("mse", "ltwo", "÷", "count")
    log_distance.dyad("signedthird", "third", "÷", "count")
    log_distance.dyad("ratio", "w", "÷", "x")
    log_distance.monad("ratiolog", "⋆⁼", "ratio")
    log_distance.monad("ratioabs", "|", "ratiolog")
    log_distance.fold("ratiosum", "+", "ratioabs")
    log_distance.dyad("ratiomean", "ratiosum", "÷", "count")
    log_distance.dyad("summaryone", "mae", "+", "mse")
    log_distance.dyad("summarytwo", "signedthird", "+", "ratiomean")
    log_distance.dyad("score", "summaryone", "+", "summarytwo")
    add_long(
        "program.similarity.log_ratio_distance",
        log_distance,
        "score",
        "dyadic_same",
        {"w": "positive", "x": "positive"},
        ["similarity", "log-domain", "ratio", "transcendental"],
        tolerance=5e-9,
    )

    regression = LongProgram(2)
    regression.fold("wtotal", "+", "w")
    regression.fold("xtotal", "+", "x")
    regression.monad("count", "≠", "x")
    regression.dyad("wmean", "wtotal", "÷", "count")
    regression.dyad("xmean", "xtotal", "÷", "count")
    regression.dyad("wc", "w", "-", "wmean")
    regression.dyad("xc", "x", "-", "xmean")
    regression.dyad("cross", "wc", "×", "xc")
    regression.dyad("wsquare", "wc", "×", "wc")
    regression.fold("covariance", "+", "cross")
    regression.fold("wvariance", "+", "wsquare")
    regression.dyad("safevariance", "wvariance", "+", 1)
    regression.dyad("slope", "covariance", "÷", "safevariance")
    regression.dyad("slopemean", "slope", "×", "wmean")
    regression.dyad("intercept", "xmean", "-", "slopemean")
    regression.dyad("scaled", "slope", "×", "w")
    regression.dyad("prediction", "scaled", "+", "intercept")
    regression.dyad("residual", "x", "-", "prediction")
    regression.monad("absolute", "|", "residual")
    regression.dyad("square", "residual", "×", "residual")
    regression.fold("lone", "+", "absolute")
    regression.fold("ltwo", "+", "square")
    regression.fold("peak", "⌈", "absolute")
    regression.dyad("summary", "lone", "+", "ltwo")
    regression.dyad("score", "summary", "+", "peak")
    add_long(
        "program.similarity.regression_residual",
        regression,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "regression", "residual", "reduction"],
        tolerance=5e-9,
    )

    combined_norm = LongProgram(2)
    distance_prefix(combined_norm)
    combined_norm.dyad("sum", "w", "+", "x")
    combined_norm.monad("sumabs", "|", "sum")
    combined_norm.dyad("sumsquare", "sum", "×", "sum")
    combined_norm.fold("sumlone", "+", "sumabs")
    combined_norm.fold("sumltwo", "+", "sumsquare")
    combined_norm.dyad("product", "w", "×", "x")
    combined_norm.monad("productabs", "|", "product")
    combined_norm.dyad("productsquare", "product", "×", "product")
    combined_norm.fold("productlone", "+", "productabs")
    combined_norm.fold("productltwo", "+", "productsquare")
    combined_norm.dyad("first", "sumlone", "+", "sumltwo")
    combined_norm.dyad("second", "productlone", "+", "productltwo")
    combined_norm.dyad("third", "mae", "+", "mse")
    combined_norm.dyad("combined", "first", "+", "second")
    combined_norm.dyad("score", "combined", "+", "third")
    add_long(
        "program.similarity.combined_norms",
        combined_norm,
        "score",
        "dyadic_same",
        {"w": "signed", "x": "signed"},
        ["similarity", "norm", "product", "reduction"],
        tolerance=5e-9,
    )

    def lag_program(identifier: str, lag: int) -> None:
        builder = LongProgram(1)
        builder.dyad("front", -lag, "↓", "x")
        builder.dyad("back", lag, "↓", "x")
        builder.dyad("difference", "back", "-", "front")
        builder.monad("absolute", "|", "difference")
        builder.dyad("square", "difference", "×", "difference")
        builder.dyad("cube", "square", "×", "difference")
        builder.dyad("cross", "front", "×", "back")
        builder.dyad("frontsquare", "front", "×", "front")
        builder.dyad("backsquare", "back", "×", "back")
        builder.fold("lone", "+", "absolute")
        builder.fold("ltwo", "+", "square")
        builder.fold("third", "+", "cube")
        builder.fold("crosssum", "+", "cross")
        builder.fold("frontenergy", "+", "frontsquare")
        builder.fold("backenergy", "+", "backsquare")
        builder.dyad("energymul", "frontenergy", "×", "backenergy")
        builder.dyad("safeenergy", "energymul", "+", 1)
        builder.monad("denom", "√", "safeenergy")
        builder.dyad("correlation", "crosssum", "÷", "denom")
        builder.monad("count", "≠", "difference")
        builder.dyad("meanvariation", "lone", "÷", "count")
        builder.dyad("meanenergy", "ltwo", "÷", "count")
        builder.dyad("summaryone", "meanvariation", "+", "meanenergy")
        builder.dyad("summarytwo", "third", "+", "correlation")
        builder.dyad("score", "summaryone", "+", "summarytwo")
        add_long(
            identifier,
            builder,
            "score",
            "monadic_vector",
            {"x": "signed"},
            ["signal", "lag", f"lag-{lag}", "autocorrelation", "difference"],
            tolerance=5e-9,
        )

    for lag in (1, 2, 4, 8):
        lag_program(f"program.signal.lag_{lag}_profile", lag)

    def second_difference(identifier: str, spacing: int) -> None:
        builder = LongProgram(1)
        builder.dyad("left", -(2 * spacing), "↓", "x")
        builder.dyad("withoutlast", -spacing, "↓", "x")
        builder.dyad("middle", spacing, "↓", "withoutlast")
        builder.dyad("right", 2 * spacing, "↓", "x")
        builder.dyad("twomiddle", 2, "×", "middle")
        builder.dyad("edge", "left", "+", "right")
        builder.dyad("curvature", "edge", "-", "twomiddle")
        builder.monad("absolute", "|", "curvature")
        builder.dyad("square", "curvature", "×", "curvature")
        builder.dyad("cube", "square", "×", "curvature")
        builder.fold("lone", "+", "absolute")
        builder.fold("ltwo", "+", "square")
        builder.fold("third", "+", "cube")
        builder.fold("peak", "⌈", "absolute")
        builder.monad("count", "≠", "curvature")
        builder.dyad("meanabsolute", "lone", "÷", "count")
        builder.dyad("meanenergy", "ltwo", "÷", "count")
        builder.dyad("meanthird", "third", "÷", "count")
        builder.dyad("summaryone", "meanabsolute", "+", "meanenergy")
        builder.dyad("summarytwo", "meanthird", "+", "peak")
        builder.dyad("score", "summaryone", "+", "summarytwo")
        add_long(
            identifier,
            builder,
            "score",
            "monadic_vector",
            {"x": "signed"},
            ["signal", "finite-difference", f"spacing-{spacing}", "curvature"],
            tolerance=5e-9,
        )

    second_difference("program.signal.second_difference_1", 1)
    second_difference("program.signal.second_difference_2", 2)

    bidirectional = LongProgram(1)
    bidirectional.scan("forward", "+", "x")
    bidirectional.monad("reverse", "⌽", "x")
    bidirectional.scan("backwardraw", "+", "reverse")
    bidirectional.monad("backward", "⌽", "backwardraw")
    bidirectional.dyad("combined", "forward", "+", "backward")
    bidirectional.fold("total", "+", "x")
    bidirectional.dyad("centered", "combined", "-", "total")
    bidirectional.monad("absolute", "|", "centered")
    bidirectional.dyad("square", "centered", "×", "centered")
    bidirectional.dyad("cube", "square", "×", "centered")
    bidirectional.fold("lone", "+", "absolute")
    bidirectional.fold("ltwo", "+", "square")
    bidirectional.fold("third", "+", "cube")
    bidirectional.fold("peak", "⌈", "absolute")
    bidirectional.monad("count", "≠", "x")
    bidirectional.dyad("meanabsolute", "lone", "÷", "count")
    bidirectional.dyad("meanenergy", "ltwo", "÷", "count")
    bidirectional.dyad("meanthird", "third", "÷", "count")
    bidirectional.dyad("summaryone", "meanabsolute", "+", "meanenergy")
    bidirectional.dyad("summarytwo", "meanthird", "+", "peak")
    bidirectional.dyad("score", "summaryone", "+", "summarytwo")
    add_long(
        "program.signal.bidirectional_prefix",
        bidirectional,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["signal", "scan", "reverse", "prefix"],
        tolerance=2e-8,
    )

    ordered_prefix = LongProgram(1)
    ordered_prefix.monad("ordered", "∧", "x")
    ordered_prefix.scan("prefix", "+", "ordered")
    ordered_prefix.monad("reverse", "⌽", "ordered")
    ordered_prefix.scan("suffixraw", "+", "reverse")
    ordered_prefix.monad("suffix", "⌽", "suffixraw")
    ordered_prefix.dyad("balance", "prefix", "-", "suffix")
    ordered_prefix.monad("absolute", "|", "balance")
    ordered_prefix.dyad("square", "balance", "×", "balance")
    ordered_prefix.dyad("cube", "square", "×", "balance")
    ordered_prefix.fold("lone", "+", "absolute")
    ordered_prefix.fold("ltwo", "+", "square")
    ordered_prefix.fold("third", "+", "cube")
    ordered_prefix.fold("peak", "⌈", "absolute")
    ordered_prefix.fold("minimum", "⌊", "balance")
    ordered_prefix.fold("maximum", "⌈", "balance")
    ordered_prefix.dyad("spread", "maximum", "-", "minimum")
    ordered_prefix.monad("count", "≠", "x")
    ordered_prefix.dyad("meanabsolute", "lone", "÷", "count")
    ordered_prefix.dyad("meanenergy", "ltwo", "÷", "count")
    ordered_prefix.dyad("summaryone", "meanabsolute", "+", "meanenergy")
    ordered_prefix.dyad("summarytwo", "third", "+", "spread")
    ordered_prefix.dyad("score", "summaryone", "+", "summarytwo")
    add_long(
        "program.signal.ordered_prefix_balance",
        ordered_prefix,
        "score",
        "monadic_vector",
        {"x": "signed"},
        ["signal", "scan", "ordering", "prefix"],
        tolerance=2e-8,
    )

    matrix_axis = LongProgram(1)
    matrix_axis.insert("columnsum", "+", "x")
    matrix_axis.monad("transpose", "⍉", "x")
    matrix_axis.insert("rowsum", "+", "transpose")
    matrix_axis.monad("columnabs", "|", "columnsum")
    matrix_axis.monad("rowabs", "|", "rowsum")
    matrix_axis.dyad("columnsquare", "columnsum", "×", "columnsum")
    matrix_axis.dyad("rowsquare", "rowsum", "×", "rowsum")
    matrix_axis.fold("columnlone", "+", "columnabs")
    matrix_axis.fold("rowlone", "+", "rowabs")
    matrix_axis.fold("columnenergy", "+", "columnsquare")
    matrix_axis.fold("rowenergy", "+", "rowsquare")
    matrix_axis.fold("columnpeak", "⌈", "columnabs")
    matrix_axis.fold("rowpeak", "⌈", "rowabs")
    matrix_axis.dyad("lonesummary", "columnlone", "+", "rowlone")
    matrix_axis.dyad("energysummary", "columnenergy", "+", "rowenergy")
    matrix_axis.dyad("peaksummary", "columnpeak", "+", "rowpeak")
    matrix_axis.dyad("first", "lonesummary", "+", "energysummary")
    matrix_axis.dyad("score", "first", "+", "peaksummary")
    add_long(
        "program.matrix.axis_summary",
        matrix_axis,
        "score",
        "monadic_matrix",
        {"x": "signed"},
        ["matrix", "insert", "transpose", "axis"],
        tolerance=2e-8,
    )

    matrix_scan = LongProgram(1)
    matrix_scan.scan("forward", "+", "x")
    matrix_scan.monad("reverse", "⌽", "x")
    matrix_scan.scan("backwardraw", "+", "reverse")
    matrix_scan.monad("backward", "⌽", "backwardraw")
    matrix_scan.dyad("combined", "forward", "+", "backward")
    matrix_scan.monad("flat", "⥊", "combined")
    matrix_scan.monad("absolute", "|", "flat")
    matrix_scan.dyad("square", "flat", "×", "flat")
    matrix_scan.dyad("cube", "square", "×", "flat")
    matrix_scan.fold("lone", "+", "absolute")
    matrix_scan.fold("ltwo", "+", "square")
    matrix_scan.fold("third", "+", "cube")
    matrix_scan.fold("peak", "⌈", "absolute")
    matrix_scan.monad("count", "≠", "flat")
    matrix_scan.dyad("meanabsolute", "lone", "÷", "count")
    matrix_scan.dyad("meanenergy", "ltwo", "÷", "count")
    matrix_scan.dyad("meanthird", "third", "÷", "count")
    matrix_scan.dyad("summaryone", "meanabsolute", "+", "meanenergy")
    matrix_scan.dyad("summarytwo", "meanthird", "+", "peak")
    matrix_scan.dyad("score", "summaryone", "+", "summarytwo")
    add_long(
        "program.matrix.bidirectional_axis_scan",
        matrix_scan,
        "score",
        "monadic_matrix",
        {"x": "signed"},
        ["matrix", "scan", "reverse", "deshape"],
        tolerance=2e-8,
    )

    matrix_layout = LongProgram(1)
    matrix_layout.monad("reverse", "⌽", "x")
    matrix_layout.monad("transpose", "⍉", "reverse")
    matrix_layout.monad("reverseagain", "⌽", "transpose")
    matrix_layout.monad("flat", "⥊", "reverseagain")
    matrix_layout.monad("absolute", "|", "flat")
    matrix_layout.dyad("square", "flat", "×", "flat")
    matrix_layout.dyad("cube", "square", "×", "flat")
    matrix_layout.dyad("fourth", "square", "×", "square")
    matrix_layout.fold("total", "+", "flat")
    matrix_layout.fold("lone", "+", "absolute")
    matrix_layout.fold("ltwo", "+", "square")
    matrix_layout.fold("third", "+", "cube")
    matrix_layout.fold("fourthsum", "+", "fourth")
    matrix_layout.fold("minimum", "⌊", "flat")
    matrix_layout.fold("maximum", "⌈", "flat")
    matrix_layout.dyad("spread", "maximum", "-", "minimum")
    matrix_layout.dyad("first", "total", "+", "lone")
    matrix_layout.dyad("second", "ltwo", "+", "third")
    matrix_layout.dyad("thirdsummary", "fourthsum", "+", "spread")
    matrix_layout.dyad("combined", "first", "+", "second")
    matrix_layout.dyad("score", "combined", "+", "thirdsummary")
    add_long(
        "program.matrix.layout_moments",
        matrix_layout,
        "score",
        "monadic_matrix",
        {"x": "signed"},
        ["matrix", "layout", "transpose", "reverse", "moments"],
        tolerance=2e-8,
    )

    major_cells = LongProgram(1)
    major_cells.monad("firstmask", "∊", "x")
    major_cells.monad("occurrence", "⊒", "x")
    major_cells.monad("classes", "⊐", "x")
    major_cells.monad("unique", "⍷", "x")
    major_cells.fold("uniquecount", "+", "firstmask")
    major_cells.fold("occurrencesum", "+", "occurrence")
    major_cells.fold("classsum", "+", "classes")
    major_cells.dyad("occurrencesquare", "occurrence", "×", "occurrence")
    major_cells.dyad("classsquare", "classes", "×", "classes")
    major_cells.fold("occurrenceenergy", "+", "occurrencesquare")
    major_cells.fold("classenergy", "+", "classsquare")
    major_cells.monad("uniqueflat", "⥊", "unique")
    major_cells.monad("uniqueabs", "|", "uniqueflat")
    major_cells.dyad("uniquesquare", "uniqueflat", "×", "uniqueflat")
    major_cells.fold("uniquelone", "+", "uniqueabs")
    major_cells.fold("uniqueenergy", "+", "uniquesquare")
    major_cells.dyad("first", "uniquecount", "+", "occurrencesum")
    major_cells.dyad("second", "classsum", "+", "occurrenceenergy")
    major_cells.dyad("third", "classenergy", "+", "uniquelone")
    major_cells.dyad("fourth", "third", "+", "uniqueenergy")
    major_cells.dyad("combined", "first", "+", "second")
    major_cells.dyad("score", "combined", "+", "fourth")
    add_long(
        "program.matrix.major_cell_profile",
        major_cells,
        "score",
        "monadic_matrix",
        {"x": "signed"},
        ["matrix", "major-cell", "self-search", "classification", "unique"],
        tolerance=2e-8,
    )

    for index in range(5):
        steps = (safe_steps[index:] + safe_steps[:index]) * 3
        expression = fold("+", apply_steps(x, steps[: 24 + index]))
        add(
            f"program.long_pipeline_reduce.{index + 1:02d}",
            "program",
            "naive",
            expression,
            1,
            "monadic_vector",
            {"x": "signed"},
            ["program", "long", "reduction", "fusible"],
            rtol=2e-11,
            atol=2e-11,
        )

    identifiers = [program["id"] for program in programs]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("generated corpus IDs are not unique")
    if len(programs) < 100:
        raise AssertionError("initial corpus unexpectedly has fewer than 100 programs")
    return programs


def render_manifest() -> str:
    programs = make_programs()
    categories = Counter(program["category"] for program in programs)
    manifest = {
        "schema_version": 2,
        "policy": {
            "growth": "append durable cases as bugs, optimizations, and workloads are discovered",
            "initial_floor": 100,
            "categories": dict(sorted(categories.items())),
        },
        "programs": programs,
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render_manifest()
    if arguments.check:
        if not DESTINATION.exists() or DESTINATION.read_text(encoding="utf-8") != rendered:
            print(f"{DESTINATION.relative_to(ROOT)} is stale", file=sys.stderr)
            return 1
        return 0
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(make_programs())} programs to {DESTINATION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
