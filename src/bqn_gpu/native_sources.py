"""Render independent direct-framework implementations of corpus expressions."""

from __future__ import annotations

from typing import Mapping

from .ir import Expression


def render_native_source(
    expression: Expression,
    framework: str,
    *,
    arity: int,
    input_mode: str,
) -> str:
    """Return a direct tinygrad or PyTorch lambda for a corpus expression.

    These sources are generated from the corpus workload specification, not by
    parsing or lowering BQN. They are the independent framework baselines used
    to measure the custom BQN frontend/backend path.
    """

    if framework not in {"tinygrad", "torch"}:
        raise ValueError(f"unknown native framework {framework!r}")
    if arity not in {1, 2}:
        raise ValueError(f"unsupported arity {arity}")
    names = ("x",) if arity == 1 else ("w", "x")
    arguments = {name: name for name in names}
    if input_mode == "leading_left":
        arguments["w"] = "w.reshape(w.shape + (1,))"
    elif input_mode == "leading_right":
        arguments["x"] = "x.reshape(x.shape + (1,))"
    body = _render(expression, framework, arguments)
    return f"lambda {', '.join(names)}: {body}"


def _render(
    expression: Expression,
    framework: str,
    arguments: Mapping[str, str],
) -> str:
    operation = expression["op"]
    if operation == "argument":
        return arguments[expression["name"]]
    if operation == "constant":
        return repr(float(expression["value"]))
    if operation == "array":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.float64" if framework == "tinygrad" else "torch.float64"
        values = repr([float(value) for value in expression["values"]])
        device = next(iter(arguments.values()), "x")
        return f"{constructor}({values}, dtype={dtype}, device=({device}).device)"
    if operation == "call":
        children = [
            _render(child, framework, arguments)
            for child in expression["arguments"]
        ]
        if len(children) == 1:
            return _monadic(expression["glyph"], children[0], framework)
        if len(children) == 2:
            return _dyadic(expression["glyph"], children[0], children[1], framework)
        raise ValueError("native primitive calls must be monadic or dyadic")
    if operation == "fold":
        value = _render(expression["argument"], framework, arguments)
        methods = {"+": "sum", "×": "prod", "⌊": "min", "⌈": "max"}
        try:
            return f"({value}).{methods[expression['glyph']]}()"
        except KeyError:
            raise ValueError(f"unsupported native Fold {expression['glyph']!r}") from None
    if operation in {"insert", "scan"}:
        value = _render(expression["argument"], framework, arguments)
        glyph = expression["glyph"]
        if operation == "insert":
            methods = {"+": "sum", "×": "prod", "⌊": "min", "⌈": "max"}
            try:
                return f"({value}).{methods[glyph]}(axis=0)"
            except KeyError:
                raise ValueError(f"unsupported native Insert {glyph!r}") from None
        methods = {"+": "cumsum", "×": "cumprod"}
        try:
            return f"({value}).{methods[glyph]}(axis=0)"
        except KeyError:
            raise ValueError(f"unsupported native Scan {glyph!r}") from None
    raise ValueError(f"unknown native IR operation {operation!r}")


def _monadic(glyph: str, value: str, framework: str) -> str:
    if glyph == "+":
        return f"({value})"
    if glyph == "-":
        return f"(-({value}))"
    methods = {
        "×": "sign",
        "⋆": "exp",
        "√": "sqrt",
        "⌊": "floor",
        "⌈": "ceil",
        "|": "abs",
    }
    if glyph in methods:
        return f"({value}).{methods[glyph]}()"
    if glyph == "÷":
        return f"(1.0 / ({value}))"
    if glyph == "=":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.float64" if framework == "tinygrad" else "torch.float64"
        return f"{constructor}(float(len(({value}).shape)), dtype={dtype}, device=({value}).device)"
    if glyph == "≠":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.float64" if framework == "tinygrad" else "torch.float64"
        length = f"(1 if len(({value}).shape) == 0 else ({value}).shape[0])"
        return f"{constructor}(float({length}), dtype={dtype}, device=({value}).device)"
    if glyph == "≢":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.float64" if framework == "tinygrad" else "torch.float64"
        return f"{constructor}(list(({value}).shape), dtype={dtype}, device=({value}).device)"
    if glyph == "↕":
        if framework == "tinygrad":
            return f"Tensor.arange(int(({value}).item()), device=({value}).device).cast(dtypes.float64)"
        return f"torch.arange(int(({value}).item()), dtype=torch.float64, device=({value}).device)"
    raise ValueError(f"unsupported native monadic primitive {glyph!r}")


def _dyadic(glyph: str, left: str, right: str, framework: str) -> str:
    operators = {
        "+": "+",
        "-": "-",
        "×": "*",
        "÷": "/",
        "⋆": "**",
    }
    if glyph in operators:
        return f"(({left}) {operators[glyph]} ({right}))"
    if glyph == "√":
        return f"(({right}) ** (1.0 / ({left})))"
    if glyph == "|":
        floor = f"(({right}) / ({left})).floor()"
        return f"(({right}) - ({left}) * ({floor}))"
    if glyph in {"⌊", "⌈"}:
        method = "minimum" if glyph == "⌊" else "maximum"
        if framework == "tinygrad":
            return f"({left}).{method}({right})"
        return f"torch.{method}(({left}), ({right}))"
    comparisons = {
        "=": "==",
        "≠": "!=",
        "<": "<",
        ">": ">",
        "≤": "<=",
        "≥": ">=",
    }
    if glyph in comparisons:
        comparison = f"(({left}) {comparisons[glyph]} ({right}))"
        return (
            f"{comparison}.cast(dtypes.float64)"
            if framework == "tinygrad"
            else f"{comparison}.to(torch.float64)"
        )
    raise ValueError(f"unsupported native dyadic primitive {glyph!r}")
