#!/usr/bin/env python3
"""Generate and differentially verify dense BQN discovery candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bqn_gpu import HostValue, TinygradBackend, compile_bqn  # noqa: E402
from bqn_gpu.cbqn import CBQN  # noqa: E402
from bqn_gpu.discovery import generate_programs  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--size", type=int, default=67)
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=("grammar", "mutation", "combinator", "train", "repeat"),
        help="repeat to select strategies; defaults to all five",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".build" / "generated-candidates.json",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    programs = generate_programs(
        seed=arguments.seed,
        count=arguments.count,
        min_steps=arguments.min_steps,
        max_steps=arguments.max_steps,
        strategies=arguments.strategy
        or ("grammar", "mutation", "combinator", "train", "repeat"),
    )
    randomizer = random.Random(arguments.seed)
    host_input = HostValue.from_array(
        (randomizer.uniform(-2.0, 2.0) for _ in range(arguments.size)),
        (arguments.size,),
    )

    cbqn = CBQN(ROOT / ".build/cbqn/libcbqn.so")
    backend = TinygradBackend(arguments.device)
    failures: list[dict[str, str]] = []
    try:
        for index, program in enumerate(programs, 1):
            print(f"[{index}/{len(programs)}] {program.id}", file=sys.stderr)
            try:
                expected = cbqn.call(program.bqn, host_input)
                actual = compile_bqn(program.bqn).execute(backend, x=host_input)
                _assert_close(actual, expected)
                if program.equivalent_to_bqn is not None:
                    equivalent = cbqn.call(program.equivalent_to_bqn, host_input)
                    _assert_close(expected, equivalent)
            except Exception as error:
                failures.append({"id": program.id, "error": str(error)})
    finally:
        cbqn.close()

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "kind": "typed-grammar-equivalence-mutation-combinator-train-and-repeat-expansion",
            "seed": arguments.seed,
            "count": arguments.count,
            "min_steps": arguments.min_steps,
            "max_steps": arguments.max_steps,
            "input_size": arguments.size,
        },
        "verified": len(failures) == 0,
        "failures": failures,
        "candidates": [program.as_dict() for program in programs],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output, file=sys.stderr)
    return 0 if not failures else 1


def _assert_close(actual: HostValue, expected: HostValue) -> None:
    if actual.atom != expected.atom or actual.shape != expected.shape:
        raise AssertionError(
            f"result kind/shape differs: {actual.atom, actual.shape} != "
            f"{expected.atom, expected.shape}"
        )
    for index, (got, wanted) in enumerate(zip(actual.data, expected.data, strict=True)):
        if math.isnan(got) and math.isnan(wanted):
            continue
        if not math.isclose(got, wanted, rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError(f"result item {index} differs: {got!r} != {wanted!r}")


if __name__ == "__main__":
    raise SystemExit(main())
