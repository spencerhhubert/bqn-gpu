"""Small serializable expression IR shared by corpus execution backends."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

from .protocol import ExecutionBackend, ValueT


Expression = dict[str, Any]


def argument(name: str) -> Expression:
    return {"op": "argument", "name": name}


def constant(value: Real) -> Expression:
    return {"op": "constant", "value": value}


def monadic(glyph: str, x: Expression) -> Expression:
    return {"op": "call", "glyph": glyph, "arguments": [x]}


def dyadic(glyph: str, w: Expression, x: Expression) -> Expression:
    return {"op": "call", "glyph": glyph, "arguments": [w, x]}


def fold(glyph: str, x: Expression) -> Expression:
    return {"op": "fold", "glyph": glyph, "argument": x}


def evaluate(
    expression: Expression,
    backend: ExecutionBackend[ValueT],
    arguments: Mapping[str, ValueT],
) -> ValueT:
    operation = expression["op"]
    if operation == "argument":
        return arguments[expression["name"]]
    if operation == "constant":
        return backend.atom(expression["value"])
    if operation == "call":
        values = tuple(evaluate(child, backend, arguments) for child in expression["arguments"])
        return backend.call(expression["glyph"], *values)
    if operation == "fold":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.reduce(expression["glyph"], value)
    raise ValueError(f"unknown IR operation {operation!r}")


def render_bqn(expression: Expression) -> str:
    operation = expression["op"]
    if operation == "argument":
        return {"w": "𝕨", "x": "𝕩"}[expression["name"]]
    if operation == "constant":
        return _render_number(expression["value"])
    if operation == "call":
        children = expression["arguments"]
        if len(children) == 1:
            return f"({expression['glyph']}{render_bqn(children[0])})"
        if len(children) == 2:
            return (
                f"({render_bqn(children[0])}{expression['glyph']}"
                f"{render_bqn(children[1])})"
            )
        raise ValueError("BQN primitive calls must be monadic or dyadic")
    if operation == "fold":
        return f"({expression['glyph']}´{render_bqn(expression['argument'])})"
    raise ValueError(f"unknown IR operation {operation!r}")


def function_source(expression: Expression) -> str:
    return "{" + render_bqn(expression) + "}"


def _render_number(value: Real) -> str:
    number = float(value)
    rendered = repr(number) if not number.is_integer() else str(int(number))
    return "¯" + rendered[1:] if rendered.startswith("-") else rendered
