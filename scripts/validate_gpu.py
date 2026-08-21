#!/usr/bin/env python3
"""Run CUDA conformance and emit release-grade validation metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from tinygrad import Device, Tensor, dtypes


ROOT = Path(__file__).resolve().parents[1]


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full", "fuzz"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".build/validation")
    arguments = parser.parse_args()

    try:
        Device["CUDA"]
        (Tensor([1.0], dtype=dtypes.float64, device="CUDA") + 1).realize()
    except Exception as error:
        print(f"tinygrad cannot execute on CUDA: {error}", file=sys.stderr)
        return 2

    case_counts = {"smoke": 64, "full": 2048, "fuzz": 20000}
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tinygrad_junit_path = arguments.output_dir / "junit-tinygrad.xml"
    torch_junit_path = arguments.output_dir / "junit-torch.xml"

    environment = os.environ.copy()
    environment.update(
        {
            "BQN_GPU_TEST_DEVICE": "CUDA",
            "BQN_GPU_TORCH_TEST_DEVICE": "CUDA",
            "BQN_GPU_FUZZ_SEED": str(arguments.seed),
            "BQN_GPU_FUZZ_CASES": str(case_counts[arguments.profile]),
        }
    )
    # tinygrad and PyTorch own independent CUDA runtime/context state. Running
    # their full suites in one process can leave PyTorch with stale resource
    # handles after tinygrad tests. Fresh processes also make failures easier
    # to attribute to one adapter.
    tinygrad_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-k",
            "not torch",
            f"--junitxml={tinygrad_junit_path}",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    torch_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_dense_primitives.py",
            "tests/test_torch_backend.py",
            "tests/test_native_baselines.py",
            "-k",
            "torch",
            f"--junitxml={torch_junit_path}",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    test_exit_code = (
        0
        if tinygrad_completed.returncode == 0 and torch_completed.returncode == 0
        else 1
    )

    gpu_fields = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ).split(", ")
    conformance = json.loads((ROOT / "conformance.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result": "pass" if test_exit_code == 0 else "fail",
        "test_exit_code": test_exit_code,
        "suite_exit_codes": {
            "tinygrad": tinygrad_completed.returncode,
            "torch": torch_completed.returncode,
        },
        "profile": arguments.profile,
        "random_seed": arguments.seed,
        "random_cases": case_counts[arguments.profile],
        "repository_commit": command("git", "rev-parse", "HEAD"),
        "repository_dirty": bool(command("git", "status", "--porcelain")),
        "cbqn_commit": (ROOT / "deps/cbqn.rev").read_text(encoding="utf-8").strip(),
        "backends": [
            {
                "name": "tinygrad",
                "revision": conformance["backend"]["revision"],
                "device": "CUDA",
            },
            *torch_backend_metadata(),
        ],
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "gpu": {
            "name": gpu_fields[0],
            "driver_version": gpu_fields[1],
            "memory_mib": int(gpu_fields[2]),
            "compute_capability": gpu_fields[3],
        },
        "artifacts": [
            "gpu-validation.json",
            "junit-tinygrad.xml",
            "junit-torch.xml",
        ],
        "sanitizer": "not run; the current backend dispatches generated tinygrad kernels",
    }
    manifest_path = arguments.output_dir / "gpu-validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return test_exit_code


def torch_backend_metadata() -> list[dict[str, str]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "name": "PyTorch",
            "version": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "device": str(torch.device("cuda", torch.cuda.current_device())),
        }
    ]


if __name__ == "__main__":
    raise SystemExit(main())
