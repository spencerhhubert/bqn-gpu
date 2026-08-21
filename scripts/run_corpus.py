#!/usr/bin/env python3
"""Correctness-check and benchmark BQN sources across execution backends."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bqn_gpu.cbqn import CBQN  # noqa: E402
from bqn_gpu.corpus import (  # noqa: E402
    Program,
    assert_close,
    generate_inputs,
    load_programs,
)
from bqn_gpu.ir import evaluate  # noqa: E402
from bqn_gpu.errors import BQNGPUError  # noqa: E402
from bqn_gpu.source import CompiledProgram, compile_bqn  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        action="append",
        choices=("cbqn", "tinygrad", "torch"),
        help="backend to run; repeat for more than one (default: all available)",
    )
    parser.add_argument("--device", default=os.environ.get("BQN_GPU_DEVICE", "CPU"))
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--match", default="*", help="program ID glob")
    parser.add_argument("--tag", action="append", help="require a corpus tag")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.size < 1 or arguments.warmup < 0 or arguments.repeat < 1:
        raise SystemExit("size and repeat must be positive; warmup must be non-negative")

    programs = select_programs(
        load_programs(), arguments.match, arguments.tag or (), arguments.limit
    )
    if not programs:
        raise SystemExit("no corpus programs matched")

    requested = list(
        dict.fromkeys(arguments.backend or ["cbqn", "tinygrad", "torch"])
    )
    # cBQN reserves JIT address space during initialization. It must initialize
    # before tensor runtimes allocate code mappings or start runtime workers.
    cbqn = CBQN(ROOT / ".build/cbqn/libcbqn.so")
    backends, skipped = load_backends(requested, arguments.device)
    results: list[dict[str, Any]] = []
    try:
        for index, program in enumerate(programs, 1):
            print(f"[{index}/{len(programs)}] {program.id}", file=sys.stderr)
            inputs = generate_inputs(program, size=arguments.size)
            cbqn_arguments = source_arguments(program, inputs)
            expected = cbqn.call(program.bqn, *cbqn_arguments)
            compiled = compile_bqn(program.bqn)
            for name in requested:
                if name in skipped:
                    continue
                if name == "cbqn":
                    result = benchmark_cbqn(
                        cbqn,
                        program,
                        cbqn_arguments,
                        expected,
                        arguments.warmup,
                        arguments.repeat,
                    )
                else:
                    result = benchmark_backend(
                        name,
                        backends[name],
                        compiled,
                        program,
                        inputs,
                        expected,
                        arguments.warmup,
                        arguments.repeat,
                    )
                result["size"] = arguments.size
                results.append(result)
    finally:
        cbqn.close()

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "repository_dirty": repository_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": arguments.device.upper(),
        "warmup": arguments.warmup,
        "repeat": arguments.repeat,
        "program_count": len(programs),
        "requested_backends": requested,
        "skipped_backends": skipped,
        "versions": backend_versions(backends),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
        print(arguments.output, file=sys.stderr)
    return 0


def select_programs(
    programs: Iterable[Program], pattern: str, tags: Iterable[str], limit: int | None
) -> list[Program]:
    required_tags = set(tags)
    selected = [
        program
        for program in programs
        if fnmatch.fnmatchcase(program.id, pattern)
        and required_tags.issubset(program.tags)
    ]
    return selected if limit is None else selected[:limit]


def load_backends(
    requested: list[str], device: str
) -> tuple[dict[str, Any], dict[str, str]]:
    backends: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in requested:
        if name == "cbqn" or name in backends or name in skipped:
            continue
        try:
            if name == "tinygrad":
                from bqn_gpu.tinygrad_backend import TinygradBackend

                backends[name] = TinygradBackend(device)
            elif name == "torch":
                from bqn_gpu.torch_backend import TorchBackend

                backends[name] = TorchBackend(device)
        except (BQNGPUError, ImportError, RuntimeError) as error:
            skipped[name] = str(error)
    return backends, skipped


def source_arguments(program: Program, inputs: dict[str, Any]) -> tuple[Any, ...]:
    return (inputs["x"],) if program.arity == 1 else (inputs["w"], inputs["x"])


def benchmark_cbqn(
    cbqn: CBQN,
    program: Program,
    arguments: tuple[Any, ...],
    expected: Any,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    cold_start = time.perf_counter_ns()
    actual = cbqn.call(program.bqn, *arguments)
    cold_ns = time.perf_counter_ns() - cold_start
    assert_close(actual, expected, program)
    for _ in range(warmup):
        cbqn.call(program.bqn, *arguments)
    timings = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        cbqn.call(program.bqn, *arguments)
        timings.append(time.perf_counter_ns() - start)
    result = timing_result(program, "cbqn", "CPU", cold_ns, timings)
    result["execution_mode"] = "embedding-call"
    return result


def benchmark_backend(
    name: str,
    backend: Any,
    compiled: CompiledProgram,
    program: Program,
    inputs: dict[str, Any],
    expected: Any,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device_inputs = {key: backend.from_host(value) for key, value in inputs.items()}
    compiler = getattr(backend, "compile", None)
    can_compile = getattr(backend, "can_compile", lambda expression: True)
    executable = (
        compiler(compiled.expression, device_inputs)
        if compiler is not None and can_compile(compiled.expression)
        else None
    )

    def run_once() -> tuple[Any, int]:
        backend.synchronize()
        start = time.perf_counter_ns()
        value = (
            executable(device_inputs)
            if executable is not None
            else evaluate(compiled.expression, backend, device_inputs)
        )
        realize = getattr(value.tensor, "realize", None)
        if realize is not None:
            realize()
        backend.synchronize()
        return value, time.perf_counter_ns() - start

    actual, cold_ns = run_once()
    assert_close(actual.to_host(), expected, program)
    for _ in range(warmup):
        run_once()
    timings = [run_once()[1] for _ in range(repeat)]
    result = timing_result(program, name, str(backend.device).upper(), cold_ns, timings)
    result["execution_mode"] = (
        "jit-captured" if executable is not None else "eager-dispatch"
    )
    return result


def timing_result(
    program: Program,
    backend: str,
    device: str,
    cold_ns: int,
    timings: list[int],
) -> dict[str, Any]:
    return {
        "program_id": program.id,
        "source_sha256": hashlib.sha256(program.bqn.encode("utf-8")).hexdigest(),
        "category": program.category,
        "variant": program.variant,
        "tags": list(program.tags),
        "backend": backend,
        "device": device,
        "correct": True,
        "cold_ns": cold_ns,
        "warm_ns": timings,
        "median_warm_ns": int(statistics.median(timings)),
        "min_warm_ns": min(timings),
        "max_warm_ns": max(timings),
    }


def backend_versions(backends: dict[str, Any]) -> dict[str, str]:
    versions = {"cbqn": (ROOT / "deps/cbqn.rev").read_text().strip()}
    if "tinygrad" in backends:
        versions["tinygrad"] = importlib.metadata.version("tinygrad")
    if "torch" in backends:
        versions["torch"] = importlib.metadata.version("torch")
    return versions


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def repository_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
