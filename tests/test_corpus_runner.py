from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_corpus_runner_emits_stable_correctness_and_timing_json(tmp_path) -> None:
    output = tmp_path / "results.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_corpus.py",
            "--backend",
            "cbqn",
            "--backend",
            "tinygrad",
            "--match",
            "glyph.add.dyadic_same",
            "--size",
            "17",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["program_count"] == 1
    assert {result["backend"] for result in report["results"]} == {
        "cbqn",
        "tinygrad",
    }
    assert all(result["correct"] for result in report["results"])
    assert all(result["median_warm_ns"] > 0 for result in report["results"])
