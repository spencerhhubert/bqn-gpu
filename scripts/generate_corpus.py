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
    constant,
    dyadic,
    fold,
    function_source,
    monadic,
    render_bqn,
)


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
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        programs.append(
            {
                "id": identifier,
                "category": category,
                "variant": variant,
                "arity": arity,
                "bqn": source or function_source(expression),
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
        "schema_version": 1,
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
