"""Small serializable expression IR shared by corpus execution backends."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Sequence

from .protocol import ExecutionBackend, ValueT


Expression = dict[str, Any]
FunctionExpression = dict[str, Any]


def argument(name: str) -> Expression:
    return {"op": "argument", "name": name}


def constant(value: Real) -> Expression:
    return {"op": "constant", "value": value}


def array_constant(values: Sequence[Real]) -> Expression:
    return {"op": "array", "values": list(values), "shape": [len(values)]}


def monadic(glyph: str, x: Expression) -> Expression:
    return {"op": "call", "glyph": glyph, "arguments": [x]}


def dyadic(glyph: str, w: Expression, x: Expression) -> Expression:
    return {"op": "call", "glyph": glyph, "arguments": [w, x]}


def fold(glyph: str, x: Expression) -> Expression:
    return {"op": "fold", "glyph": glyph, "argument": x}


def insert(glyph: str, x: Expression) -> Expression:
    return {"op": "insert", "glyph": glyph, "argument": x}


def scan(glyph: str, x: Expression) -> Expression:
    return {"op": "scan", "glyph": glyph, "argument": x}


def primitive_function(glyph: str) -> FunctionExpression:
    return {"kind": "primitive", "glyph": glyph}


def modified_function(
    modifier: str,
    left: FunctionExpression,
    right: FunctionExpression | None = None,
) -> FunctionExpression:
    function = {"kind": "modifier", "modifier": modifier, "left": left}
    if right is not None:
        function["right"] = right
    return function


def constant_function(value: Expression) -> FunctionExpression:
    return {"kind": "constant", "value": value}


def apply_function(
    function: FunctionExpression,
    arguments: Sequence[Expression],
) -> Expression:
    kind = function["kind"]
    if kind == "primitive":
        if len(arguments) == 1:
            return monadic(function["glyph"], arguments[0])
        if len(arguments) == 2:
            return dyadic(function["glyph"], arguments[0], arguments[1])
        raise ValueError("BQN functions must be called with one or two arguments")
    if kind == "constant":
        return function["value"]
    if kind == "fold":
        if len(arguments) != 1:
            raise ValueError("Fold/Insert/Scan initial values are not implemented")
        operation = function["modifier"]
        constructor = {"´": fold, "˝": insert, "`": scan}[operation]
        return constructor(function["glyph"], arguments[0])
    if kind == "modifier":
        return {
            "op": "combinator",
            "modifier": function["modifier"],
            "left": function["left"],
            **({"right": function["right"]} if "right" in function else {}),
            "arguments": list(arguments),
        }
    raise ValueError(f"unknown function IR kind {kind!r}")


def expand_combinator(expression: Expression) -> Expression:
    """Expand a pure combinator call without making backend decisions."""

    if expression["op"] != "combinator":
        raise ValueError("expected a combinator expression")
    modifier = expression["modifier"]
    left = expression["left"]
    right = expression.get("right")
    arguments = expression["arguments"]
    if len(arguments) not in {1, 2}:
        raise ValueError("BQN combinators require one or two arguments")
    x = arguments[-1]
    w = arguments[0] if len(arguments) == 2 else x

    if modifier == "˜":
        return apply_function(left, [x, w])
    if right is None:
        raise ValueError(f"combinator {modifier!r} requires two operands")
    if modifier == "∘":
        inner_arguments = [x] if len(arguments) == 1 else [w, x]
        return apply_function(left, [apply_function(right, inner_arguments)])
    if modifier == "○":
        if len(arguments) == 1:
            return apply_function(left, [apply_function(right, [x])])
        return apply_function(
            left,
            [apply_function(right, [w]), apply_function(right, [x])],
        )
    if modifier == "⊸":
        return apply_function(
            right,
            [apply_function(left, [w]), x],
        )
    if modifier == "⟜":
        return apply_function(
            left,
            [w, apply_function(right, [x])],
        )
    raise ValueError(f"unknown combinator {modifier!r}")


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
    if operation == "array":
        return backend.array(expression["values"], expression["shape"])
    if operation == "call":
        values = tuple(evaluate(child, backend, arguments) for child in expression["arguments"])
        return backend.call(expression["glyph"], *values)
    if operation == "static_call":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.call_static(
            expression["glyph"],
            expression["left_values"],
            expression["left_atom"],
            value,
        )
    if operation == "fold":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.reduce(expression["glyph"], value)
    if operation == "insert":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.insert(expression["glyph"], value)
    if operation == "scan":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.scan(expression["glyph"], value)
    if operation == "combinator":
        return evaluate(expand_combinator(expression), backend, arguments)
    raise ValueError(f"unknown IR operation {operation!r}")


def has_tensor_compute(expression: Expression) -> bool:
    """Whether evaluating an expression launches data-dependent tensor work."""

    operation = expression["op"]
    if operation in {"argument", "constant", "array"}:
        return False
    if operation in {"fold", "insert", "scan"}:
        return True
    if operation == "static_call":
        return True
    if operation == "combinator":
        return has_tensor_compute(expand_combinator(expression))
    if operation == "call":
        glyph = expression["glyph"]
        children = expression["arguments"]
        if glyph in {"=", "≠", "≢", "↕"} and len(children) == 1:
            return False
        if glyph == "+" and len(children) == 1:
            return has_tensor_compute(children[0])
        return True
    raise ValueError(f"unknown IR operation {operation!r}")


def render_bqn(expression: Expression) -> str:
    operation = expression["op"]
    if operation == "argument":
        return {"w": "𝕨", "x": "𝕩"}[expression["name"]]
    if operation == "constant":
        return _render_number(expression["value"])
    if operation == "array":
        return "‿".join(_render_number(value) for value in expression["values"])
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
    if operation == "static_call":
        values = expression["left_values"]
        left = (
            _render_number(values[0])
            if expression["left_atom"]
            else "‿".join(_render_number(value) for value in values)
        )
        return f"({left}{expression['glyph']}{render_bqn(expression['argument'])})"
    if operation == "fold":
        return f"({expression['glyph']}´{render_bqn(expression['argument'])})"
    if operation == "insert":
        return f"({expression['glyph']}˝{render_bqn(expression['argument'])})"
    if operation == "scan":
        return f"({expression['glyph']}`{render_bqn(expression['argument'])})"
    if operation == "combinator":
        function = {
            "kind": "modifier",
            "modifier": expression["modifier"],
            "left": expression["left"],
            **({"right": expression["right"]} if "right" in expression else {}),
        }
        arguments = expression["arguments"]
        rendered_function = render_function(function)
        if len(arguments) == 1:
            return f"({rendered_function} {render_bqn(arguments[0])})"
        return (
            f"({render_bqn(arguments[0])} {rendered_function} "
            f"{render_bqn(arguments[1])})"
        )
    raise ValueError(f"unknown IR operation {operation!r}")


def render_function(function: FunctionExpression) -> str:
    kind = function["kind"]
    if kind == "primitive":
        return str(function["glyph"])
    if kind == "constant":
        return render_bqn(function["value"])
    if kind == "fold":
        return f"{function['glyph']}{function['modifier']}"
    if kind == "modifier":
        left = render_function(function["left"])
        if "right" not in function:
            return f"{left}{function['modifier']}"
        return f"{left}{function['modifier']}{render_function(function['right'])}"
    raise ValueError(f"unknown function IR kind {kind!r}")


def function_source(expression: Expression) -> str:
    return "{" + render_bqn(expression) + "}"


def _render_number(value: Real) -> str:
    number = float(value)
    rendered = repr(number) if not number.is_integer() else str(int(number))
    return "¯" + rendered[1:] if rendered.startswith("-") else rendered
