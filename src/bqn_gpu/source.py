"""Compile a small, explicit BQN source subset to the backend-neutral IR."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

from .errors import SourceError
from .host_value import HostValue
from .ir import (
    Expression,
    FunctionExpression,
    apply_function,
    array_constant,
    constant,
    constant_function,
    evaluate,
    modified_function,
    primitive_function,
)
from .protocol import ExecutionBackend, ValueT


_GLYPHS = frozenset(
    "+-×÷⋆√⌊⌈∧∨¬|≤<>≥=≠≡≢⊣⊢⥊∾≍⋈↑↓↕»«⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!"
)
_NUMBER = re.compile(
    r"¯?(?:∞|π|(?:\d+(?:\.\d+)?)(?:[eE]¯?\d+)?)"
)
_MISSING = object()
_ONE_MODIFIERS = {
    "SELF": "˜",
    "CELLS": "˘",
    "EACH": "¨",
    "TABLE": "⌜",
}
_TWO_MODIFIERS = {
    "ATOP": "∘",
    "OVER": "○",
    "BEFORE": "⊸",
    "AFTER": "⟜",
    "RANK_MODIFIER": "⎉",
}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int


@dataclass(frozen=True)
class CompiledProgram:
    """A source program lowered to backend-neutral operations."""

    source: str
    expression: Expression
    arity: int

    def execute(
        self,
        backend: ExecutionBackend[ValueT],
        *,
        x: HostValue | object = _MISSING,
        w: HostValue | object = _MISSING,
    ) -> HostValue:
        supplied_x = x is not _MISSING
        supplied_w = w is not _MISSING
        if self.arity == 0 and (supplied_x or supplied_w):
            raise SourceError("this BQN program does not take arguments")
        if self.arity == 1 and (not supplied_x or supplied_w):
            raise SourceError("this BQN function requires exactly a right argument (--x)")
        if self.arity == 2 and (not supplied_x or not supplied_w):
            raise SourceError("this BQN function requires both --w and --x arguments")

        arguments: dict[str, ValueT] = {}
        if supplied_x:
            if not isinstance(x, HostValue):
                raise TypeError("x must be a HostValue")
            arguments["x"] = backend.from_host(x)
        if supplied_w:
            if not isinstance(w, HostValue):
                raise TypeError("w must be a HostValue")
            arguments["w"] = backend.from_host(w)
        return evaluate(self.expression, backend, arguments).to_host()


def compile_bqn(source: str) -> CompiledProgram:
    """Compile a BQN string containing a function block or bare program body."""

    if not isinstance(source, str):
        raise TypeError("BQN source must be a string")
    expression = _Parser(source, _tokenize(source)).parse()
    arguments = _argument_names(expression)
    arity = 2 if "w" in arguments else 1 if "x" in arguments else 0
    return CompiledProgram(source=source, expression=expression, arity=arity)


def execute(
    source: str,
    backend: ExecutionBackend[ValueT],
    *,
    x: HostValue | object = _MISSING,
    w: HostValue | object = _MISSING,
) -> HostValue:
    """Compile and execute BQN source on an execution backend."""

    return compile_bqn(source).execute(backend, x=x, w=w)


class _Parser:
    def __init__(self, source: str, tokens: Sequence[Token]) -> None:
        self.source = source
        self.tokens = tokens
        self.index = 0
        self.bindings: dict[str, Expression] = {}

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def peek(self, offset: int = 1) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def match(self, kind: str) -> Token | None:
        if self.current.kind != kind:
            return None
        return self.advance()

    def expect(self, kind: str, message: str) -> Token:
        token = self.match(kind)
        if token is None:
            self.fail(message)
        return token

    def fail(self, message: str, token: Token | None = None) -> None:
        location = token or self.current
        raise SourceError(
            f"{message} at line {location.line}, column {location.column} "
            f"(near {location.text!r})"
        )

    def parse(self) -> Expression:
        self._skip_separators()
        if self.match("LBRACE"):
            expression = self._sequence("RBRACE")
            self.expect("RBRACE", "expected '}' to close BQN block")
            self._skip_separators()
            self.expect("EOF", "unexpected source after BQN block")
            return expression
        expression = self._sequence("EOF")
        self.expect("EOF", "unexpected source after BQN program")
        return expression

    def _sequence(self, end_kind: str) -> Expression:
        self._skip_separators()
        if self.current.kind == end_kind:
            self.fail("BQN program body cannot be empty")

        result: Expression | None = None
        while self.current.kind != end_kind:
            if self.current.kind == "EOF":
                if end_kind == "RBRACE":
                    self.fail("expected '}' to close BQN block")
                self.fail(f"expected {end_kind}")
            if self.current.kind == "NAME" and self.peek().kind == "ASSIGN":
                name = _normalize_name(self.advance().text)
                self.advance()
                result = self._expression()
                self.bindings[name] = result
            else:
                result = self._expression()

            if self.current.kind == "EOF" and end_kind == "RBRACE":
                self.fail("expected '}' to close BQN block")
            if self.current.kind not in ("SEP", end_kind):
                self.fail("expected a statement separator")
            self._skip_separators()

        if result is None:
            self.fail("BQN program body cannot be empty")
        return result

    def _skip_separators(self) -> None:
        while self.match("SEP") is not None:
            pass

    def _expression(self) -> Expression:
        if self.current.kind == "GLYPH" or self._starts_literal_bound_function():
            function = self._function()
            if self.current.kind in {"EOF", "RBRACE", "RPAREN", "SEP"}:
                self.fail("derived BQN function requires an argument")
            return apply_function(function, [self._expression()])

        left = self._subject()
        if self.current.kind == "GLYPH":
            function = self._function()
            return apply_function(function, [left, self._expression()])
        return left

    def _function(self) -> FunctionExpression:
        function = self._function_operand()
        while True:
            if self.current.kind in _ONE_MODIFIERS:
                function = modified_function(
                    _ONE_MODIFIERS[self.advance().kind],
                    function,
                )
                continue
            if self.current.kind in _TWO_MODIFIERS:
                modifier = _TWO_MODIFIERS[self.advance().kind]
                function = modified_function(
                    modifier,
                    function,
                    self._function_operand(),
                )
                continue
            return function

    def _function_operand(self) -> FunctionExpression:
        if self.current.kind == "GLYPH":
            glyph = self.advance().text
            if self.current.kind in {"FOLD", "INSERT", "SCAN"}:
                modifier = self.advance().text
                return {"kind": "fold", "glyph": glyph, "modifier": modifier}
            return primitive_function(glyph)
        if self.current.kind == "NUMBER":
            return constant_function(self._numeric_literal())
        self.fail("expected a primitive or numeric operand after modifier")

    def _starts_literal_bound_function(self) -> bool:
        if self.current.kind != "NUMBER":
            return False
        offset = 1
        while self.peek(offset).kind == "STRAND":
            if self.peek(offset + 1).kind != "NUMBER":
                return False
            offset += 2
        return self.peek(offset).kind in _TWO_MODIFIERS

    def _subject(self) -> Expression:
        token = self.current
        if self.current.kind == "NUMBER":
            return self._numeric_literal()
        if self.match("ARG"):
            return {"op": "argument", "name": {"𝕨": "w", "𝕩": "x"}[token.text]}
        if self.match("NAME"):
            name = _normalize_name(token.text)
            try:
                return self.bindings[name]
            except KeyError:
                self.fail(f"unknown local name {token.text!r}", token)
        if self.match("LPAREN"):
            expression = self._expression()
            self.expect("RPAREN", "expected ')' to close expression")
            return expression
        self.fail("expected a numeric value, argument, local name, or parenthesized expression")

    def _numeric_literal(self) -> Expression:
        first = self.expect("NUMBER", "expected a numeric literal")
        values = [_parse_number(first.text)]
        while self.match("STRAND") is not None:
            item = self.expect("NUMBER", "numeric strands currently require literals")
            values.append(_parse_number(item.text))
        return array_constant(values) if len(values) > 1 else constant(values[0])


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    line = 1
    column = 1
    length = len(source)

    def add(kind: str, text: str, token_line: int = line, token_column: int = column) -> None:
        tokens.append(Token(kind, text, token_line, token_column))

    while index < length:
        character = source[index]
        if character in " \t":
            index += 1
            column += 1
            continue
        if character == "#":
            while index < length and source[index] not in "\r\n":
                index += 1
                column += 1
            continue
        if character in "\r\n":
            token_line, token_column = line, column
            if character == "\r" and index + 1 < length and source[index + 1] == "\n":
                index += 2
            else:
                index += 1
            line += 1
            column = 1
            add("SEP", "\n", token_line, token_column)
            continue

        single = {
            "{": "LBRACE",
            "}": "RBRACE",
            "(": "LPAREN",
            ")": "RPAREN",
            "⋄": "SEP",
            ",": "SEP",
            "←": "ASSIGN",
            "´": "FOLD",
            "˝": "INSERT",
            "`": "SCAN",
            "˜": "SELF",
            "∘": "ATOP",
            "○": "OVER",
            "⊸": "BEFORE",
            "⟜": "AFTER",
            "˘": "CELLS",
            "¨": "EACH",
            "⌜": "TABLE",
            "⎉": "RANK_MODIFIER",
            "‿": "STRAND",
            "𝕨": "ARG",
            "𝕩": "ARG",
        }.get(character)
        if single is not None:
            add(single, character)
            index += 1
            column += 1
            continue
        if character in _GLYPHS:
            add("GLYPH", character)
            index += 1
            column += 1
            continue

        number = _NUMBER.match(source, index)
        if number is not None:
            text = number.group(0)
            add("NUMBER", text)
            index = number.end()
            column += len(text)
            continue
        if character.isalpha() or character == "_":
            end = index + 1
            while end < length and (source[end].isalnum() or source[end] == "_"):
                end += 1
            text = source[index:end]
            add("NAME", text)
            index = end
            column += len(text)
            continue
        raise SourceError(
            f"unsupported BQN token {character!r} at line {line}, column {column}"
        )

    tokens.append(Token("EOF", "<end of source>", line, column))
    return tokens


def _parse_number(text: str) -> float:
    negative = text.startswith("¯")
    unsigned = text[1:] if negative else text
    if unsigned == "∞":
        value = math.inf
    elif unsigned == "π":
        value = math.pi
    else:
        value = float(unsigned.replace("¯", "-"))
    return -value if negative else value


def _normalize_name(name: str) -> str:
    return name.replace("_", "").casefold()


def _argument_names(expression: Mapping[str, object]) -> set[str]:
    operation = expression["op"]
    if operation == "argument":
        return {str(expression["name"])}
    if operation == "call":
        result: set[str] = set()
        for child in expression["arguments"]:  # type: ignore[union-attr]
            result.update(_argument_names(child))
        return result
    if operation in {"fold", "insert", "scan"}:
        return _argument_names(expression["argument"])  # type: ignore[arg-type]
    if operation == "combinator":
        result: set[str] = set()
        for child in expression["arguments"]:  # type: ignore[union-attr]
            result.update(_argument_names(child))
        result.update(_function_argument_names(expression["left"]))  # type: ignore[arg-type]
        if "right" in expression:
            result.update(_function_argument_names(expression["right"]))  # type: ignore[arg-type]
        return result
    if operation == "map":
        result: set[str] = set()
        for child in expression["arguments"]:  # type: ignore[union-attr]
            result.update(_argument_names(child))
        result.update(_function_argument_names(expression["function"]))  # type: ignore[arg-type]
        return result
    return set()


def _function_argument_names(function: Mapping[str, object]) -> set[str]:
    if function["kind"] == "constant":
        return _argument_names(function["value"])  # type: ignore[arg-type]
    if function["kind"] == "modifier":
        result = _function_argument_names(function["left"])  # type: ignore[arg-type]
        if "right" in function:
            result.update(_function_argument_names(function["right"]))  # type: ignore[arg-type]
        return result
    return set()
