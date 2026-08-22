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
            "bqn-gpu-tinygrad",
            "--backend",
            "native-tinygrad",
            "--backend",
            "native-torch",
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
    assert report["schema_version"] == 2
    assert report["program_count"] == 1
    assert {result["backend"] for result in report["results"]} == {
        "cbqn",
        "bqn-gpu-tinygrad",
        "native-tinygrad",
        "native-torch",
    }
    assert all(result["correct"] for result in report["results"])
    assert all(result["median_warm_ns"] > 0 for result in report["results"])
    assert all(result["timing_scope"] == "resident-compute" for result in report["results"])
    assert report["timing_scope"] == "resident-compute"
    assert report["environment"]["fingerprint"]
    assert report["environment"]["cpu"]["threads"] > 0
    assert report["environment"]["software"]["tinygrad-renderer"].endswith(
        ".ClangJITRenderer"
    )
    assert report["environment"]["software"]["tinygrad-compiler"].endswith(
        ".ClangJITCompiler"
    )
    assert next(
        result for result in report["results"] if result["backend"] == "cbqn"
    )["execution_mode"] == "embedding-resident"
    compiled = next(
        result
        for result in report["results"]
        if result["backend"] == "bqn-gpu-tinygrad"
    )
    assert compiled["optimization"]["optimized_bqn"]
    assert isinstance(compiled["optimization"]["rewrites"], list)

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "repository_commit": report["repository_commit"],
                "result": "pass",
                "profile": "test",
                "random_seed": 1,
                "random_cases": 0,
            }
        ),
        encoding="utf-8",
    )
    payload_path = tmp_path / "payload.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/publish_results.py",
            str(output),
            "--suite",
            "test-smoke",
            "--dry-run",
            "--allow-dirty",
            "--validation-manifest",
            str(validation_path),
            "--output-payload",
            str(payload_path),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["commit"]["sha"] == report["repository_commit"]
    assert payload["environment"]["fingerprint"] == report["environment"]["fingerprint"]
    assert len(payload["programs"]) == 1
    assert len(payload["results"]) == 4
    assert all(result["timing_scope"] == "resident-compute" for result in payload["results"])
    assert all(len(result["timings_ns"]) == 1 for result in payload["results"])
    native = next(result for result in payload["results"] if result["backend"] == "native-torch")
    assert native["metadata"]["implementation_kind"] == "native-framework"
    assert payload["programs"][0]["metadata"]["native_implementations"]["torch"].startswith("lambda")
    accelerated = next(
        result
        for result in payload["results"]
        if result["backend"] == "bqn-gpu-tinygrad"
    )
    assert accelerated["metadata"]["optimization"]["optimized_bqn"]
    capability = payload["capability"]
    manifest = json.loads((ROOT / "conformance.json").read_text(encoding="utf-8"))
    assert capability["glyphs_total"] == len(manifest["primitives"]) == 44
    assert capability["metadata"]["monadic_forms_defined"] == 42
    assert capability["metadata"]["dyadic_forms_defined"] == 44
    assert capability["metadata"]["combinators_total"] == 10
    less_equal_monad = next(
        feature
        for feature in capability["features"]
        if feature["glyph"] == "≤" and feature["valence"] == "monadic"
    )
    assert less_equal_monad["metadata"]["language_defined"] is False


def test_development_profile_is_a_multi_size_certification_subset(tmp_path) -> None:
    manifest = json.loads(
        (ROOT / "corpus/benchmark-profiles.json").read_text(encoding="utf-8")
    )
    programs = json.loads(
        (ROOT / "corpus/programs.json").read_text(encoding="utf-8")
    )["programs"]
    program_ids = {program["id"] for program in programs}
    development = manifest["profiles"]["development"]
    certification = manifest["profiles"]["certification"]

    assert set(development["program_ids"]) <= program_ids
    assert certification["program_ids"] is None
    assert set(development["sizes"]) <= set(certification["sizes"])

    output = tmp_path / "profile-results.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_corpus.py",
            "--backend",
            "cbqn",
            "--backend",
            "bqn-gpu-tinygrad",
            "--profile",
            "development",
            "--match",
            "glyph.add.dyadic_same",
            "--size",
            "7",
            "--size",
            "11",
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
    assert report["benchmark_profile"] == "development"
    assert report["sizes"] == [7, 11]
    assert report["program_count"] == 1
    assert len(report["results"]) == 4
    assert {result["size"] for result in report["results"]} == {7, 11}
