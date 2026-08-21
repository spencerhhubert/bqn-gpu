"""Loading, input generation, and comparison for the program corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from .host_value import HostValue
from .ir import Expression


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "corpus" / "programs.json"


@dataclass(frozen=True)
class Program:
    id: str
    category: str
    variant: str
    arity: int
    bqn: str
    native_expression: Expression
    native_tinygrad: str
    native_torch: str
    input_mode: str
    domains: dict[str, str]
    rtol: float
    atol: float
    tags: tuple[str, ...]


def load_programs(path: Path = DEFAULT_MANIFEST) -> list[Program]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2:
        raise ValueError(f"unsupported corpus schema {manifest['schema_version']}")
    return [
        Program(
            id=item["id"],
            category=item["category"],
            variant=item["variant"],
            arity=item["arity"],
            bqn=item["bqn"],
            native_expression=dict(item["native"]["expression"]),
            native_tinygrad=item["native"]["tinygrad"],
            native_torch=item["native"]["torch"],
            input_mode=item["input_mode"],
            domains=dict(item["domains"]),
            rtol=float(item["rtol"]),
            atol=float(item["atol"]),
            tags=tuple(item["tags"]),
        )
        for item in manifest["programs"]
    ]


def generate_inputs(program: Program, size: int = 257) -> dict[str, HostValue]:
    if size < 1:
        raise ValueError("corpus input size must be positive")
    shapes = _input_shapes(program.input_mode, size)
    result: dict[str, HostValue] = {}
    for name, shape in shapes.items():
        randomizer = random.Random(_stable_seed(program.id, name, size))
        domain = program.domains[name]
        if shape is None:
            result[name] = HostValue.from_atom(_number(randomizer, domain))
        else:
            data = [_number(randomizer, domain) for _ in range(math.prod(shape))]
            result[name] = HostValue.from_array(data, shape)
    return result


def assert_close(actual: HostValue, expected: HostValue, program: Program) -> None:
    if actual.atom != expected.atom:
        raise AssertionError(
            f"{program.id}: atom mismatch: actual={actual.atom}, expected={expected.atom}"
        )
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{program.id}: shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    if len(actual.data) != len(expected.data):
        raise AssertionError(f"{program.id}: result bounds differ")
    for index, (got, wanted) in enumerate(zip(actual.data, expected.data, strict=True)):
        if math.isnan(wanted):
            if math.isnan(got):
                continue
        elif math.isclose(got, wanted, rel_tol=program.rtol, abs_tol=program.atol):
            continue
        raise AssertionError(
            f"{program.id}: item {index} differs: actual={got!r}, expected={wanted!r}, "
            f"rtol={program.rtol}, atol={program.atol}"
        )


def _input_shapes(mode: str, size: int) -> dict[str, tuple[int, ...] | None]:
    if mode == "monadic_vector":
        return {"x": (size,)}
    if mode == "monadic_atom":
        return {"x": None}
    if mode == "monadic_rank_zero":
        return {"x": ()}
    if mode == "monadic_matrix":
        rows = max(1, min(32, math.isqrt(size)))
        columns = max(1, (size + rows - 1) // rows)
        return {"x": (rows, columns)}
    if mode == "monadic_empty_vector":
        return {"x": (0,)}
    if mode == "monadic_empty_matrix":
        return {"x": (0, 3)}
    if mode == "dyadic_same":
        return {"w": (size,), "x": (size,)}
    if mode == "dyadic_atoms":
        return {"w": None, "x": None}
    if mode in {"matrix_vector", "table_vectors"}:
        rows = max(1, min(32, math.isqrt(size)))
        columns = max(1, (size + rows - 1) // rows)
        if mode == "matrix_vector":
            return {"w": (rows, columns), "x": (columns,)}
        return {"w": (rows,), "x": (columns,)}
    if mode == "left_atom":
        return {"w": None, "x": (size,)}
    if mode == "right_atom":
        return {"w": (size,), "x": None}
    rows = max(1, min(32, math.isqrt(size)))
    columns = max(1, (size + rows - 1) // rows)
    if mode == "leading_left":
        return {"w": (rows,), "x": (rows, columns)}
    if mode == "leading_right":
        return {"w": (rows, columns), "x": (rows,)}
    raise ValueError(f"unknown input mode {mode!r}")


def _stable_seed(program_id: str, name: str, size: int) -> int:
    digest = hashlib.sha256(f"{program_id}\0{name}\0{size}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _number(randomizer: random.Random, domain: str) -> float:
    if domain == "signed":
        return randomizer.uniform(-3.0, 3.0)
    if domain == "positive":
        return randomizer.uniform(0.25, 3.0)
    if domain == "nonzero":
        magnitude = randomizer.uniform(0.5, 3.0)
        return magnitude if randomizer.choice((True, False)) else -magnitude
    if domain == "fractional":
        return randomizer.uniform(-3.0, 3.0) + 0.125
    if domain == "count":
        return float(randomizer.randint(0, 64))
    if domain == "near_one":
        return randomizer.uniform(0.999, 1.001)
    if domain == "boolean":
        return float(randomizer.choice((0, 1)))
    raise ValueError(f"unknown input domain {domain!r}")
