"""Deterministic typed generation and equivalence mutation for dense BQN programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable

from .ir import (
    Expression,
    argument,
    constant,
    dyadic,
    fold,
    function_source,
    monadic,
    render_bqn,
    scan,
)


@dataclass(frozen=True)
class GeneratedProgram:
    """One reproducible candidate produced without sampling source text."""

    id: str
    strategy: str
    seed: int
    expression: Expression
    bqn: str
    equivalent_to_bqn: str | None
    features: tuple[str, ...]
    steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "seed": self.seed,
            "steps": self.steps,
            "bqn": self.bqn,
            "equivalent_to_bqn": self.equivalent_to_bqn,
            "features": list(self.features),
            "expression": self.expression,
        }


def generate_programs(
    *,
    seed: int,
    count: int,
    min_steps: int = 8,
    max_steps: int = 32,
    strategies: Iterable[str] = ("grammar", "mutation", "combinator"),
) -> list[GeneratedProgram]:
    """Generate unique, typed dense-list programs from stable per-case seeds."""

    if count < 1:
        raise ValueError("count must be positive")
    if min_steps < 1 or max_steps < min_steps:
        raise ValueError("step bounds must satisfy 1 <= min_steps <= max_steps")
    enabled = tuple(strategies)
    if not enabled or any(
        item not in {"grammar", "mutation", "combinator"} for item in enabled
    ):
        raise ValueError("strategies must contain grammar, mutation, and/or combinator")

    programs: list[GeneratedProgram] = []
    seen: set[str] = set()
    attempt = 0
    while len(programs) < count:
        case_seed = _case_seed(seed, attempt)
        strategy = enabled[attempt % len(enabled)]
        program = _generate_one(
            case_seed,
            strategy=strategy,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        attempt += 1
        if program.bqn in seen:
            continue
        seen.add(program.bqn)
        programs.append(program)
    return programs


def _generate_one(
    seed: int,
    *,
    strategy: str,
    min_steps: int,
    max_steps: int,
) -> GeneratedProgram:
    randomizer = random.Random(seed)
    steps = randomizer.randint(min_steps, max_steps)
    expression, features = _vector_chain(randomizer, steps)
    equivalent_to: str | None = None
    bqn_override: str | None = None

    if strategy == "mutation":
        equivalent_to = function_source(expression)
        mutation = randomizer.choice(("double-reverse", "cancel-rotate"))
        if mutation == "double-reverse":
            expression = monadic("⌽", monadic("⌽", expression))
        else:
            expression = dyadic(
                "⌽",
                constant(1),
                dyadic("⌽", constant(-1), expression),
            )
        features.add(mutation)
    elif strategy == "combinator":
        expression, body, feature = _combinator_candidate(randomizer, expression)
        equivalent_to = function_source(expression)
        bqn_override = "{" + body + "}"
        features.update(("combinator", feature))
    elif randomizer.random() < 0.35:
        expression = fold(randomizer.choice(("+", "⌊", "⌈")), expression)
        features.add("fold")

    bqn = bqn_override or function_source(expression)
    digest = hashlib.sha256(bqn.encode("utf-8")).hexdigest()[:16]
    return GeneratedProgram(
        id=f"generated.{strategy}.{digest}",
        strategy=strategy,
        seed=seed,
        expression=expression,
        bqn=bqn,
        equivalent_to_bqn=equivalent_to,
        features=tuple(sorted(features)),
        steps=steps,
    )


def _vector_chain(
    randomizer: random.Random,
    steps: int,
) -> tuple[Expression, set[str]]:
    expression = argument("x")
    features: set[str] = {"dense-list"}
    scans = 0
    for _ in range(steps):
        choices = ["pervasive", "scalar-right", "scalar-left", "reverse", "rotate", "sort"]
        if scans < 2:
            choices.append("scan")
        operation = randomizer.choice(choices)
        if operation == "pervasive":
            glyph = randomizer.choice(("+", "-", "×", "|", "⌊", "⌈", "¬"))
            expression = monadic(glyph, expression)
            features.add("pervasive")
        elif operation in {"scalar-right", "scalar-left"}:
            glyph = randomizer.choice(
                ("+", "-", "×", "÷")
                if operation == "scalar-right"
                else ("+", "-", "×")
            )
            value = randomizer.choice((-2, -1, -0.5, 0.5, 1, 2))
            scalar = constant(value)
            expression = (
                dyadic(glyph, expression, scalar)
                if operation == "scalar-right"
                else dyadic(glyph, scalar, expression)
            )
            features.add("scalar-extension")
        elif operation == "reverse":
            expression = monadic("⌽", expression)
            features.add("reverse")
        elif operation == "rotate":
            expression = dyadic("⌽", constant(randomizer.choice((-3, -1, 1, 3))), expression)
            features.add("rotate")
        elif operation == "sort":
            expression = monadic(randomizer.choice(("∧", "∨")), expression)
            features.add("sort")
        else:
            expression = scan(randomizer.choice(("+", "⌊", "⌈")), expression)
            scans += 1
            features.add("scan")
    return expression, features


def _combinator_candidate(
    randomizer: random.Random,
    operand: Expression,
) -> tuple[Expression, str, str]:
    rendered = render_bqn(operand)
    choice = randomizer.choice(
        (
            "self",
            "atop",
            "over",
            "before-bind",
            "after-bind",
            "atop-reduction",
            "atop-chain",
        )
    )
    if choice == "self":
        return dyadic("×", operand, operand), f"×˜{rendered}", "self-swap"
    if choice == "atop":
        return monadic("|", monadic("-", operand)), f"|∘-{rendered}", "atop"
    if choice == "over":
        return monadic("|", monadic("-", operand)), f"|○-{rendered}", "over"
    if choice == "before-bind":
        count = randomizer.choice((-3, -1, 1, 3))
        bqn_count = str(count).replace("-", "¯")
        return dyadic("⌽", constant(count), operand), f"{bqn_count}⊸⌽{rendered}", "before-bind"
    if choice == "after-bind":
        value = randomizer.choice((-2, -1, 1, 2))
        bqn_value = str(value).replace("-", "¯")
        return dyadic("-", operand, constant(value)), f"-⟜{bqn_value} {rendered}", "after-bind"
    if choice == "atop-reduction":
        return fold("+", monadic("|", operand)), f"+´∘|{rendered}", "atop"
    return (
        fold("+", monadic("|", monadic("-", operand))),
        f"+´∘|∘-{rendered}",
        "atop-chain",
    )


def _case_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
