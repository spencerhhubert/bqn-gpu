"""Shape-specialized, semantics-preserving rewrites over BQN semantic IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ir import Expression, monadic, render_bqn


@dataclass(frozen=True)
class OptimizationEvent:
    rule: str
    before: str
    after: str


@dataclass(frozen=True)
class OptimizationResult:
    expression: Expression
    events: tuple[OptimizationEvent, ...]


def optimize(
    expression: Expression,
    argument_ranks: Mapping[str, int],
) -> OptimizationResult:
    """Optimize after argument ranks are known, retaining an explainable trace."""

    events: list[OptimizationEvent] = []

    def visit(node: Expression) -> Expression:
        operation = node["op"]
        if operation == "call":
            rewritten = {
                **node,
                "arguments": [visit(child) for child in node["arguments"]],
            }
        elif operation in {"fold", "insert", "scan"}:
            rewritten = {**node, "argument": visit(node["argument"])}
        else:
            return node

        replacement, rule = _rewrite(rewritten, argument_ranks)
        if rule is not None:
            events.append(
                OptimizationEvent(
                    rule=rule,
                    before=render_bqn(rewritten),
                    after=render_bqn(replacement),
                )
            )
        return replacement

    optimized = visit(expression)
    return OptimizationResult(expression=optimized, events=tuple(events))


def infer_rank(expression: Expression, argument_ranks: Mapping[str, int]) -> int | None:
    operation = expression["op"]
    if operation == "argument":
        return argument_ranks.get(expression["name"])
    if operation == "constant":
        return 0
    if operation == "array":
        return len(expression["shape"])
    if operation in {"fold"}:
        return 0
    if operation in {"insert"}:
        rank = infer_rank(expression["argument"], argument_ranks)
        return None if rank is None else max(0, rank - 1)
    if operation == "scan":
        return infer_rank(expression["argument"], argument_ranks)
    if operation != "call":
        return None

    glyph = expression["glyph"]
    arguments = expression["arguments"]
    if len(arguments) == 2:
        if glyph in {"⌽", "↑", "↓", "⊏", "⊑", "↕", "/", "⍉"}:
            return infer_rank(arguments[1], argument_ranks)
        left = infer_rank(arguments[0], argument_ranks)
        right = infer_rank(arguments[1], argument_ranks)
        return max(left, right) if left is not None and right is not None else None

    child_rank = infer_rank(arguments[0], argument_ranks)
    if glyph in {"=", "≠", "≡", "⊑"}:
        return 0
    if glyph in {"≢", "⥊", "↕", "/", "⍋", "⍒", "⊐", "⊒", "∊", "⍷"}:
        return 1
    if glyph == "≍":
        return None if child_rank is None else child_rank + 1
    return child_rank


def _rewrite(
    expression: Expression,
    argument_ranks: Mapping[str, int],
) -> tuple[Expression, str | None]:
    if expression["op"] != "call":
        return expression, None
    glyph = expression["glyph"]
    arguments = expression["arguments"]
    if len(arguments) == 1 and glyph in {"+", "⊣", "⊢"}:
        return arguments[0], "identity-monad"

    if len(arguments) != 1:
        return _cancel_rotates(expression, argument_ranks)
    child = arguments[0]
    if child["op"] != "call" or len(child["arguments"]) != 1:
        return expression, None
    grandchild = child["arguments"][0]

    if glyph == "⌽" and child["glyph"] == "⌽":
        rank = infer_rank(grandchild, argument_ranks)
        if rank is not None and rank > 0:
            return grandchild, "double-reverse"
    if glyph == "⍉" and child["glyph"] == "⍉":
        if infer_rank(grandchild, argument_ranks) == 2:
            return grandchild, "double-transpose-rank-2"
    if glyph == "⌽" and child["glyph"] in {"∧", "∨"}:
        if infer_rank(grandchild, argument_ranks) == 1:
            opposite = "∨" if child["glyph"] == "∧" else "∧"
            return monadic(opposite, grandchild), "reverse-sorted-list"
    return expression, None


def _cancel_rotates(
    expression: Expression,
    argument_ranks: Mapping[str, int],
) -> tuple[Expression, str | None]:
    glyph = expression["glyph"]
    arguments = expression["arguments"]
    if glyph != "⌽" or len(arguments) != 2:
        return expression, None
    outer = _literal_whole_numbers(arguments[0])
    inner = arguments[1]
    if inner["op"] != "call" or inner["glyph"] != "⌽" or len(inner["arguments"]) != 2:
        return expression, None
    inner_counts = _literal_whole_numbers(inner["arguments"][0])
    target = inner["arguments"][1]
    rank = infer_rank(target, argument_ranks)
    if (
        outer is not None
        and inner_counts is not None
        and len(outer) == len(inner_counts)
        and rank is not None
        and rank >= len(outer)
        and all(left + right == 0 for left, right in zip(outer, inner_counts, strict=True))
    ):
        return target, "cancel-rotates"
    return expression, None


def _literal_whole_numbers(expression: Expression) -> tuple[int, ...] | None:
    if expression["op"] == "constant":
        values = (expression["value"],)
    elif expression["op"] == "array":
        values = tuple(expression["values"])
    else:
        return None
    numbers = tuple(int(value) for value in values)
    return numbers if all(number == value for number, value in zip(numbers, values)) else None
