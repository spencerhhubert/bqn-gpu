"""Command-line entry point for executing BQN files and source strings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .errors import BQNGPUError
from .json_values import dumps_host_value, loads_host_value
from .source import compile_bqn
from .tinygrad_backend import TinygradBackend


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqn-gpu")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a .bqn source file")
    run.add_argument("path", type=Path)
    evaluate = commands.add_parser("eval", help="execute a BQN source string")
    evaluate.add_argument("source")
    for command in (run, evaluate):
        command.add_argument("--backend", choices=("tinygrad",), default="tinygrad")
        command.add_argument(
            "--device", default=os.environ.get("BQN_GPU_DEVICE", "CPU")
        )
        command.add_argument("--x", help="right argument as JSON or @path/to/file.json")
        command.add_argument("--w", help="left argument as JSON or @path/to/file.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        source = (
            arguments.path.read_text(encoding="utf-8")
            if arguments.command == "run"
            else arguments.source
        )
        compiled = compile_bqn(source)
        backend = TinygradBackend(arguments.device)
        values = {}
        if arguments.x is not None:
            values["x"] = loads_host_value(_json_argument(arguments.x))
        if arguments.w is not None:
            values["w"] = loads_host_value(_json_argument(arguments.w))
        result = compiled.execute(backend, **values)
        print(dumps_host_value(result))
        return 0
    except (BQNGPUError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"bqn-gpu: error: {error}", file=sys.stderr)
        return 2


def _json_argument(specification: str) -> str:
    if specification.startswith("@"):
        return Path(specification[1:]).read_text(encoding="utf-8")
    return specification


if __name__ == "__main__":
    raise SystemExit(main())
