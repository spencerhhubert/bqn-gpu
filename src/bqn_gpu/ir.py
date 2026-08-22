"""Small serializable expression IR shared by corpus execution backends."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Sequence

from .protocol import ExecutionBackend, ValueT


Expression = dict[str, Any]
FunctionExpression = dict[str, Any]
MAX_REPEAT_IR_NODES = 4096


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


def train_function(functions: Sequence[FunctionExpression]) -> FunctionExpression:
    if len(functions) < 2:
        raise ValueError("a BQN train requires at least two components")
    _validate_train_components(functions)
    return {"kind": "train", "functions": list(functions)}


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
    if kind == "train":
        return {
            "op": "train",
            "functions": function["functions"],
            "arguments": list(arguments),
        }
    if kind == "modifier":
        modifier = function["modifier"]
        if modifier == "⁼":
            undone = {
                "op": "undo",
                "function": function["left"],
                "arguments": list(arguments),
            }
            expand_undo(undone)
            return undone
        if modifier == "⍟":
            right = function.get("right")
            if right is None or right["kind"] != "constant":
                raise ValueError("Repeat currently requires a literal count")
            value = right["value"]
            if value["op"] != "constant":
                raise ValueError("Repeat currently requires one literal count")
            count_value = value["value"]
            count = int(count_value)
            if count_value != count or not 0 <= count <= 64:
                raise ValueError(
                    "Repeat currently requires a natural count no larger than 64"
                )
            repeated = {
                "op": "repeat",
                "function": function["left"],
                "count": count,
                "arguments": list(arguments),
            }
            expand_repeat(repeated)
            return repeated
        if modifier in {"˘", "¨", "⌜", "⎉"}:
            rank_specification: list[Real] = []
            if modifier == "⎉":
                right = function.get("right")
                if right is None or right["kind"] != "constant":
                    raise ValueError("Rank currently requires a literal numeric rank operand")
                value = right["value"]
                if value["op"] == "constant":
                    rank_specification = [value["value"]]
                elif value["op"] == "array":
                    rank_specification = list(value["values"])
                else:
                    raise ValueError("Rank currently requires literal numeric ranks")
            return {
                "op": "map",
                "modifier": modifier,
                "function": function["left"],
                "ranks": rank_specification,
                "arguments": list(arguments),
            }
        return {
            "op": "combinator",
            "modifier": modifier,
            "left": function["left"],
            **({"right": function["right"]} if "right" in function else {}),
            "arguments": list(arguments),
        }
    raise ValueError(f"unknown function IR kind {kind!r}")


def expand_train(expression: Expression) -> Expression:
    """Expand a train call while retaining its source-level semantic IR."""

    if expression["op"] != "train":
        raise ValueError("expected a train expression")
    arguments = expression["arguments"]
    if len(arguments) not in {1, 2}:
        raise ValueError("BQN trains require one or two arguments")
    return _apply_train(expression["functions"], arguments)


def expand_repeat(expression: Expression) -> Expression:
    """Unroll a statically bounded Repeat call into ordinary semantic IR."""

    if expression["op"] != "repeat":
        raise ValueError("expected a repeat expression")
    arguments = expression["arguments"]
    if len(arguments) not in {1, 2}:
        raise ValueError("BQN Repeat requires one or two arguments")
    left = arguments[0] if len(arguments) == 2 else None
    result = arguments[-1]
    for _ in range(expression["count"]):
        repeated_arguments = [result] if left is None else [left, result]
        result = apply_function(expression["function"], repeated_arguments)
        if (
            _bounded_expanded_expression_size(result, MAX_REPEAT_IR_NODES)
            > MAX_REPEAT_IR_NODES
        ):
            raise ValueError(
                f"static Repeat expansion exceeds {MAX_REPEAT_IR_NODES} semantic IR nodes"
            )
    return result


def expand_undo(expression: Expression) -> Expression:
    """Lower a documented Undo subset to ordinary semantic IR."""

    if expression["op"] != "undo":
        raise ValueError("expected an Undo expression")
    arguments = expression["arguments"]
    if len(arguments) not in {1, 2}:
        raise ValueError("BQN Undo requires one or two arguments")
    return _apply_inverse(expression["function"], arguments)


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
    if modifier == "⊘":
        selected = left if len(arguments) == 1 else right
        return apply_function(selected, arguments)
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
    if operation == "scalar_call":
        value = evaluate(expression["argument"], backend, arguments)
        return backend.call_scalar(
            expression["glyph"],
            expression["scalar"],
            expression["scalar_left"],
            value,
        )
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
    if operation == "train":
        return evaluate(expand_train(expression), backend, arguments)
    if operation == "repeat":
        return evaluate(expand_repeat(expression), backend, arguments)
    if operation == "undo":
        return evaluate(expand_undo(expression), backend, arguments)
    if operation == "map":
        values = tuple(
            evaluate(child, backend, arguments) for child in expression["arguments"]
        )

        def invoke(mapped_values: Sequence[ValueT]) -> ValueT:
            names = [f"__mapped_{index}" for index in range(len(mapped_values))]
            call = apply_function(
                expression["function"],
                [argument(name) for name in names],
            )
            return evaluate(call, backend, dict(zip(names, mapped_values, strict=True)))

        return backend.map_function(
            expression["modifier"],
            expression["ranks"],
            expression["function"],
            values,
            invoke,
        )
    raise ValueError(f"unknown IR operation {operation!r}")


def has_tensor_compute(expression: Expression) -> bool:
    """Whether evaluating an expression launches data-dependent tensor work."""

    operation = expression["op"]
    if operation in {"argument", "constant", "array"}:
        return False
    if operation in {"fold", "insert", "scan", "scalar_call"}:
        return True
    if operation == "static_call":
        return True
    if operation == "combinator":
        return has_tensor_compute(expand_combinator(expression))
    if operation == "train":
        return has_tensor_compute(expand_train(expression))
    if operation == "repeat":
        return has_tensor_compute(expand_repeat(expression))
    if operation == "undo":
        return has_tensor_compute(expand_undo(expression))
    if operation == "map":
        return True
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
    if operation == "scalar_call":
        scalar = _render_number(expression["scalar"])
        argument = render_bqn(expression["argument"])
        if expression["scalar_left"]:
            return f"({scalar}{expression['glyph']}{argument})"
        return f"({argument}{expression['glyph']}{scalar})"
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
    if operation == "train":
        function = {"kind": "train", "functions": expression["functions"]}
        arguments = expression["arguments"]
        rendered_function = render_function(function)
        if len(arguments) == 1:
            return f"({rendered_function} {render_bqn(arguments[0])})"
        return (
            f"({render_bqn(arguments[0])} {rendered_function} "
            f"{render_bqn(arguments[1])})"
        )
    if operation == "repeat":
        function = render_function(expression["function"])
        repeated = f"{function}⍟{expression['count']}"
        arguments = expression["arguments"]
        if len(arguments) == 1:
            return f"({repeated} {render_bqn(arguments[0])})"
        return (
            f"({render_bqn(arguments[0])} {repeated} "
            f"{render_bqn(arguments[1])})"
        )
    if operation == "undo":
        function = f"{render_function(expression['function'])}⁼"
        arguments = expression["arguments"]
        if len(arguments) == 1:
            return f"({function} {render_bqn(arguments[0])})"
        return (
            f"({render_bqn(arguments[0])} {function} "
            f"{render_bqn(arguments[1])})"
        )
    if operation == "map":
        function = render_function(expression["function"])
        modifier = expression["modifier"]
        if modifier == "⎉":
            values = expression["ranks"]
            rank = "‿".join(_render_number(value) for value in values)
            function = f"{function}⎉{rank}"
        else:
            function = f"{function}{modifier}"
        arguments = expression["arguments"]
        if len(arguments) == 1:
            return f"({function} {render_bqn(arguments[0])})"
        return (
            f"({render_bqn(arguments[0])} {function} "
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
    if kind == "train":
        return "(" + "".join(
            render_function(component) for component in function["functions"]
        ) + ")"
    raise ValueError(f"unknown function IR kind {kind!r}")


def function_source(expression: Expression) -> str:
    return "{" + render_bqn(expression) + "}"


def _render_number(value: Real) -> str:
    number = float(value)
    if math.isinf(number):
        return "¯∞" if number < 0 else "∞"
    rendered = repr(number) if not number.is_integer() else str(int(number))
    return "¯" + rendered[1:] if rendered.startswith("-") else rendered


def _apply_train(
    functions: Sequence[FunctionExpression],
    arguments: Sequence[Expression],
) -> Expression:
    if len(functions) == 1:
        return apply_function(functions[0], arguments)
    if len(functions) % 2 == 0:
        return apply_function(
            functions[0],
            [_apply_train(functions[1:], arguments)],
        )
    return apply_function(
        functions[1],
        [
            apply_function(functions[0], arguments),
            _apply_train(functions[2:], arguments),
        ],
    )


def _validate_train_components(functions: Sequence[FunctionExpression]) -> None:
    if len(functions) == 1:
        if functions[0]["kind"] == "constant":
            raise ValueError("the final component of a BQN train must be a function")
        return
    if len(functions) % 2 == 0:
        if functions[0]["kind"] == "constant":
            raise ValueError("the left function of a 2-train cannot be a subject")
        _validate_train_components(functions[1:])
        return
    if functions[1]["kind"] == "constant":
        raise ValueError("a train combining position must contain a function")
    _validate_train_components(functions[2:])


def _apply_inverse(
    function: FunctionExpression,
    arguments: Sequence[Expression],
) -> Expression:
    """Apply the portable inverse cases supported by the dense compiler."""

    kind = function["kind"]
    if kind == "primitive":
        return _apply_primitive_inverse(function["glyph"], arguments)
    if kind == "modifier":
        modifier = function["modifier"]
        left = function["left"]
        if modifier == "⁼":
            return apply_function(left, arguments)
        if modifier == "⊘":
            selected = left if len(arguments) == 1 else function["right"]
            return _apply_inverse(selected, arguments)
        if modifier == "∘":
            right = function["right"]
            target = _apply_inverse(left, [arguments[-1]])
            right_arguments = [target] if len(arguments) == 1 else [arguments[0], target]
            return _apply_inverse(right, right_arguments)
        if modifier in {"˘", "¨", "⌜"}:
            if modifier == "⌜" and len(arguments) != 1:
                raise ValueError("Undo of Table is currently monadic only")
            return apply_function(
                {**function, "left": modified_function("⁼", left)},
                arguments,
            )
        if modifier == "⊸" and left["kind"] == "constant":
            return _apply_inverse(function["right"], [left["value"], arguments[-1]])
        if modifier == "˜":
            return _apply_self_inverse(left, arguments)
        raise ValueError(f"Undo for modifier {modifier!r} is not implemented")
    if kind == "train" and len(function["functions"]) == 2:
        outer, inner = function["functions"]
        target = _apply_inverse(outer, [arguments[-1]])
        inner_arguments = [target] if len(arguments) == 1 else [arguments[0], target]
        return _apply_inverse(inner, inner_arguments)
    raise ValueError("Undo is not implemented for this function form")


def _apply_primitive_inverse(
    glyph: str,
    arguments: Sequence[Expression],
) -> Expression:
    x = arguments[-1]
    if len(arguments) == 1:
        if glyph in {"+", "⊣", "⊢"}:
            return x
        if glyph in {"-", "÷", "¬", "⌽"}:
            return monadic(glyph, x)
        if glyph == "√":
            return dyadic("×", x, x)
        if glyph == "⋆":
            return monadic("⋆⁼", x)
        raise ValueError(f"monadic Undo for primitive {glyph!r} is not implemented")

    w = arguments[0]
    if glyph == "+":
        return dyadic("-", x, w)
    if glyph == "-":
        return dyadic("-", w, x)
    if glyph == "×":
        return dyadic("÷", x, w)
    if glyph == "÷":
        return dyadic("÷", w, x)
    if glyph == "√":
        return dyadic("⋆", x, w)
    if glyph == "⋆":
        return dyadic("÷", monadic("⋆⁼", x), monadic("⋆⁼", w))
    if glyph == "⊢":
        return x
    if glyph == "⌽":
        return dyadic("⌽", monadic("-", w), x)
    raise ValueError(f"dyadic Undo for primitive {glyph!r} is not implemented")


def _apply_self_inverse(
    operand: FunctionExpression,
    arguments: Sequence[Expression],
) -> Expression:
    if operand["kind"] != "primitive":
        raise ValueError("Undo of Self currently requires a primitive operand")
    glyph = operand["glyph"]
    x = arguments[-1]
    if len(arguments) == 1:
        if glyph == "+":
            return dyadic("÷", x, constant(2))
        if glyph == "×":
            return monadic("√", x)
        raise ValueError(f"monadic Undo for Self primitive {glyph!r} is not implemented")
    w = arguments[0]
    if glyph == "+":
        return dyadic("-", x, w)
    if glyph == "-":
        return dyadic("+", w, x)
    if glyph == "×":
        return dyadic("÷", x, w)
    if glyph == "÷":
        return dyadic("×", w, x)
    if glyph == "⋆":
        return dyadic("√", w, x)
    raise ValueError(f"dyadic Undo for Self primitive {glyph!r} is not implemented")


def _bounded_expanded_expression_size(expression: Expression, limit: int) -> int:
    """Count execution IR nodes, stopping once ``limit`` has been exceeded.

    Trains and combinators retain compact source-level nodes until optimization,
    so counting their surface representation would let a duplicating train hide
    exponential growth inside Repeat.
    """

    if limit < 1:
        return 1
    operation = expression["op"]
    if operation == "train":
        return _bounded_expanded_expression_size(expand_train(expression), limit)
    if operation == "combinator":
        return _bounded_expanded_expression_size(expand_combinator(expression), limit)
    if operation == "repeat":
        return _bounded_expanded_expression_size(expand_repeat(expression), limit)
    if operation == "undo":
        return _bounded_expanded_expression_size(expand_undo(expression), limit)

    total = 1
    for key in ("argument",):
        child = expression.get(key)
        if isinstance(child, dict):
            total += _bounded_expanded_expression_size(child, limit - total)
            if total > limit:
                return total
    for key in ("arguments",):
        children = expression.get(key, ())
        for child in children:
            total += _bounded_expanded_expression_size(child, limit - total)
            if total > limit:
                return total
    return total
