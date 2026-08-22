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
            structural = _dyadic_structural(
                expression["glyph"],
                expression["arguments"][0],
                children[0],
                children[1],
                framework,
            )
            if structural is not None:
                return structural
            return _dyadic(expression["glyph"], children[0], children[1], framework)
        raise ValueError("native primitive calls must be monadic or dyadic")
    if operation == "fold":
        value = _render(expression["argument"], framework, arguments)
        methods = {
            "+": "sum",
            "×": "prod",
            "∧": "prod",
            "⌊": "min",
            "⌈": "max",
        }
        if expression["glyph"] == "∨":
            return f"(1.0 - (1.0 - ({value})).prod())"
        try:
            return f"({value}).{methods[expression['glyph']]}()"
        except KeyError:
            raise ValueError(f"unsupported native Fold {expression['glyph']!r}") from None
    if operation in {"insert", "scan"}:
        value = _render(expression["argument"], framework, arguments)
        glyph = expression["glyph"]
        if operation == "insert":
            if glyph in {"+", "×", "∧", "∨"}:
                method = {
                    "+": "sum",
                    "×": "prod",
                    "∧": "prod",
                    "∨": "prod",
                }[glyph]
                reduction = f"({value}).{method}(axis=0)"
                if glyph == "∨":
                    reduction = f"(1.0 - (1.0 - ({value})).prod(axis=0))"
                return reduction
            if glyph in {"⌊", "⌈"}:
                method = "min" if glyph == "⌊" else "max"
                result = f"({value}).{method}(axis=0)"
                return result if framework == "tinygrad" else f"({result}).values"
            raise ValueError(f"unsupported native Insert {glyph!r}")
        if glyph in {"+", "×", "∧"}:
            method = "cumsum" if glyph == "+" else "cumprod"
            return f"({value}).{method}(axis=0)"
        if glyph == "∨":
            return f"(1.0 - (1.0 - ({value})).cumprod(axis=0))"
        if glyph == "⌈":
            result = f"({value}).cummax(axis=0)"
            return f"({result})[0]" if framework == "tinygrad" else f"({result}).values"
        if glyph == "⌊":
            if framework == "tinygrad":
                return f"(-(-({value})).cummax(axis=0)[0])"
            return f"(({value}).cummin(axis=0)).values"
        raise ValueError(f"unsupported native Scan {glyph!r}")
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
    if glyph in {"∧", "∨"}:
        result = f"({value}).sort(dim=0, descending={glyph == '∨'})"
        return f"({result})[0]" if framework == "tinygrad" else f"({result}).values"
    if glyph == "¬":
        return f"(1.0 - ({value}))"
    if glyph == "≡":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.float64" if framework == "tinygrad" else "torch.float64"
        depth = f"(0.0 if len(({value}).shape) == 0 else 1.0)"
        return f"{constructor}({depth}, dtype={dtype}, device=({value}).device)"
    if glyph in {"⊣", "⊢"}:
        return f"({value})"
    if glyph == "⥊":
        return f"({value}).reshape((-1,))"
    if glyph == "≍":
        return f"({value}).reshape((1,) + tuple(({value}).shape))"
    if glyph == "⌽":
        return (
            f"({value}).flip(0)"
            if framework == "tinygrad"
            else f"torch.flip(({value}), dims=(0,))"
        )
    if glyph == "⍉":
        axes = f"(tuple(range(1, len(({value}).shape))) + (0,))"
        return f"({value}).permute({axes})"
    if glyph in {"⍋", "⍒"}:
        return f"({value}).argsort(dim=0, descending={glyph == '⍒'})"
    if glyph in {"⊐", "⊒", "∊", "⍷"}:
        return f"major_cell_self_search(({value}), {glyph!r})"
    if glyph in {"»", "«"}:
        return f"dense_shift(({value}), {glyph!r})"
    if glyph == "⊏":
        return f"({value})[0]"
    if glyph == "⊑":
        return f"({value}).reshape((-1,))[0]"
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
    if glyph == "∧":
        return f"(({left}) * ({right}))"
    if glyph == "∨":
        return f"(({left}) + ({right}) - ({left}) * ({right}))"
    if glyph == "¬":
        return f"(1.0 + ({left}) - ({right}))"
    if glyph in {"≡", "≢"}:
        comparison = f"((({left}) == ({right})).all())"
        comparison = (
            f"({comparison}).cast(dtypes.float64)"
            if framework == "tinygrad"
            else f"({comparison}).to(torch.float64)"
        )
        if glyph == "≢":
            comparison = f"(1.0 - ({comparison}))"
        return comparison
    if glyph == "⊣":
        return f"({left})"
    if glyph == "⊢":
        return f"({right})"
    raise ValueError(f"unsupported native dyadic primitive {glyph!r}")


def _dyadic_structural(
    glyph: str,
    left_expression: Expression,
    left: str,
    right: str,
    framework: str,
) -> str | None:
    if glyph in {"»", "«"}:
        return f"dense_shift(({right}), {glyph!r}, ({left}))"
    if glyph == "∾":
        if framework == "tinygrad":
            return f"({left}).cat(({right}), dim=0)"
        return f"torch.cat((({left}), ({right})), dim=0)"
    if glyph in {"≍", "⋈"}:
        if framework == "tinygrad":
            return f"({left}).stack(({right}), dim=0)"
        return f"torch.stack((({left}), ({right})), dim=0)"

    numbers = _literal_whole_numbers(left_expression)
    if numbers is None:
        return None
    if glyph in {"↑", "↓"} and len(numbers) == 1:
        count = numbers[0]
        if glyph == "↑":
            selection = f":{count}" if count >= 0 else f"{count}:"
        else:
            selection = f"{count}:" if count >= 0 else f":{count}"
        return f"({right})[{selection}]"
    if glyph == "⌽":
        shifts = tuple(-number for number in numbers)
        dims = tuple(range(len(numbers)))
        return f"({right}).roll({shifts!r}, dims={dims!r})"
    if glyph == "⍉":
        rank = len(numbers)
        if sorted(numbers) != list(range(rank)):
            raise ValueError("native Reorder Axes requires a full permutation literal")
        axes = tuple(sorted(range(rank), key=numbers.__getitem__))
        return f"({right}).permute({axes!r})"
    if glyph == "/" and len(numbers) == 1:
        return f"({right}).repeat_interleave({numbers[0]}, dim=0)"
    if glyph == "↕" and len(numbers) == 1:
        return f"({right}).unfold(0, {numbers[0]}, 1)"
    if glyph == "⊏":
        constructor = "Tensor" if framework == "tinygrad" else "torch.tensor"
        dtype = "dtypes.int32" if framework == "tinygrad" else "torch.int64"
        index = f"{constructor}({list(numbers)!r}, dtype={dtype}, device=({right}).device)"
        return f"({right})[{index}]"
    if glyph == "⊑":
        return f"({right})[{numbers!r}]"
    return None


def _literal_whole_numbers(expression: Expression) -> tuple[int, ...] | None:
    if expression["op"] == "constant":
        values = (expression["value"],)
    elif expression["op"] == "array":
        values = tuple(expression["values"])
    else:
        return None
    numbers = tuple(int(value) for value in values)
    return numbers if all(number == value for number, value in zip(numbers, values)) else None
