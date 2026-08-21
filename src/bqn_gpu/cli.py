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


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqn-gpu")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a .bqn source file")
    run.add_argument("path", type=Path)
    evaluate = commands.add_parser("eval", help="execute a BQN source string")
    evaluate.add_argument("source")
    for command in (run, evaluate):
        command.add_argument(
            "--backend", choices=("tinygrad", "torch"), default="tinygrad"
        )
        command.add_argument(
            "--device", default=os.environ.get("BQN_GPU_DEVICE", "CPU")
        )
        command.add_argument("--x", help="right argument as JSON or @path/to/file.json")
        command.add_argument("--w", help="left argument as JSON or @path/to/file.json")
        command.add_argument(
            "--fallback",
            choices=("cbqn", "error"),
            default="cbqn",
            help="delegate unsupported numeric programs to cBQN when available",
        )
        command.add_argument(
            "--cbqn-lib",
            type=Path,
            default=Path(
                os.environ.get("CBQN_LIB", ROOT / ".build/cbqn/libcbqn.so")
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    oracle = None
    try:
        source = (
            arguments.path.read_text(encoding="utf-8")
            if arguments.command == "run"
            else arguments.source
        )
        if arguments.fallback == "cbqn" and arguments.cbqn_lib.is_file():
            # cBQN reserves JIT address space and must initialize before tensor
            # runtimes map code or start runtime workers.
            from .cbqn import CBQN

            oracle = CBQN(arguments.cbqn_lib)
        if arguments.backend == "tinygrad":
            from .tinygrad_backend import TinygradBackend

            backend = TinygradBackend(arguments.device)
        else:
            from .torch_backend import TorchBackend

            backend = TorchBackend(arguments.device)
        values = {}
        if arguments.x is not None:
            values["x"] = loads_host_value(_json_argument(arguments.x))
        if arguments.w is not None:
            values["w"] = loads_host_value(_json_argument(arguments.w))
        try:
            result = compile_bqn(source).execute(backend, **values)
        except BQNGPUError as acceleration_error:
            if arguments.fallback == "error":
                raise
            if oracle is None:
                raise BQNGPUError(
                    f"{acceleration_error}; cBQN fallback is unavailable at "
                    f"{arguments.cbqn_lib}"
                ) from acceleration_error
            cbqn_arguments = (
                (values["w"], values["x"])
                if "w" in values
                else (values["x"],)
                if "x" in values
                else ()
            )
            result = (
                oracle.call(source, *cbqn_arguments)
                if cbqn_arguments
                else oracle.evaluate(source)
            )
            print(
                f"bqn-gpu: cBQN fallback: {acceleration_error}",
                file=sys.stderr,
            )
        print(dumps_host_value(result))
        return 0
    except (BQNGPUError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"bqn-gpu: error: {error}", file=sys.stderr)
        return 2
    finally:
        if oracle is not None:
            oracle.close()


def _json_argument(specification: str) -> str:
    if specification.startswith("@"):
        return Path(specification[1:]).read_text(encoding="utf-8")
    return specification


if __name__ == "__main__":
    raise SystemExit(main())
