#!/usr/bin/env python3
"""Convert a corpus report into a website ingestion bundle and publish it."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bqn_gpu.corpus import Program, load_programs  # noqa: E402


DEFAULT_ENDPOINT = "https://bqn-gpu-website.spencerhhubert.workers.dev/api/v1/ingest"
REPOSITORY = "https://github.com/spencerhhubert/bqn-gpu"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("BQN_GPU_RESULTS_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--token-env", default="BQN_GPU_RESULTS_TOKEN")
    parser.add_argument("--suite", default="corpus")
    parser.add_argument("--run-id")
    parser.add_argument("--ref")
    parser.add_argument("--artifact-url")
    parser.add_argument("--environment-label")
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--output-payload", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    report = read_json(arguments.report)
    if report.get("schema_version") != 2:
        raise SystemExit("publish_results requires a schema version 2 corpus report")
    if report.get("repository_dirty") and not arguments.allow_dirty:
        raise SystemExit("refusing to publish a report measured from a dirty worktree")
    validation = (
        read_json(arguments.validation_manifest)
        if arguments.validation_manifest is not None
        else None
    )
    payload = build_payload(
        report,
        suite=arguments.suite,
        run_id=arguments.run_id,
        ref=arguments.ref,
        artifact_url=arguments.artifact_url,
        environment_label=arguments.environment_label,
        validation=validation,
        junit_path=arguments.junit,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if arguments.output_payload is not None:
        arguments.output_payload.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_payload.write_text(rendered, encoding="utf-8")
        print(arguments.output_payload, file=sys.stderr)
    if arguments.dry_run:
        if arguments.output_payload is None:
            print(rendered, end="")
        return 0

    token = os.environ.get(arguments.token_env)
    if not token:
        raise SystemExit(f"{arguments.token_env} is required unless --dry-run is used")
    response = post_json(arguments.endpoint, rendered.encode("utf-8"), token)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


def build_payload(
    report: dict[str, Any],
    *,
    suite: str,
    run_id: str | None,
    ref: str | None,
    artifact_url: str | None,
    environment_label: str | None,
    validation: dict[str, Any] | None,
    junit_path: Path | None,
) -> dict[str, Any]:
    commit = require_full_commit(report["repository_commit"])
    environment = dict(report["environment"])
    environment["captured_at"] = report["started_at_utc"]
    if environment_label:
        environment["label"] = environment_label
    environment = {key: value for key, value in environment.items() if value is not None}

    programs_by_id = {program.id: program for program in load_programs()}
    selected_ids = list(dict.fromkeys(result["program_id"] for result in report["results"]))
    try:
        selected = [programs_by_id[program_id] for program_id in selected_ids]
    except KeyError as error:
        raise ValueError(f"report references unknown corpus program {error.args[0]}") from error
    for result in report["results"]:
        actual_hash = hashlib.sha256(
            programs_by_id[result["program_id"]].bqn.encode("utf-8")
        ).hexdigest()
        if result["source_sha256"] != actual_hash:
            raise ValueError(f"source hash mismatch for {result['program_id']}")

    status = "pass"
    if any(not result["correct"] for result in report["results"]):
        status = "fail"
    elif report.get("skipped_backends"):
        status = "partial"

    identifier = run_id or default_run_id(commit, suite, report["started_at_utc"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "commit": {
            "sha": commit,
            "repository": REPOSITORY,
            "ref": ref,
            "committed_at": commit_timestamp(commit),
            "url": f"{REPOSITORY}/commit/{commit}",
        },
        "environment": environment,
        "run": {
            "id": identifier,
            "suite": suite,
            "started_at": report["started_at_utc"],
            "finished_at": report["generated_at_utc"],
            "status": status,
            "device": report["device"],
            "dtype": "float64",
            "timing_scope": report["timing_scope"],
            "input_profile": {
                "generator": "bqn-gpu-corpus-v1",
                "size": result_size(report),
                "selection": {
                    "program_count": report["program_count"],
                    "program_ids": selected_ids,
                },
            },
            "warmups": report["warmup"],
            "repetitions": report["repeat"],
            "command": shlex.join(report["command"]),
            "artifact_url": artifact_url,
            "runner_version": "run_corpus.schema-2",
            "metadata": {
                "requested_backends": report["requested_backends"],
                "skipped_backends": report["skipped_backends"],
            },
        },
        "programs": [program_record(program) for program in selected],
        "results": [result_record(result, report["versions"]) for result in report["results"]],
    }
    if validation is not None:
        if validation.get("repository_commit") != commit:
            raise ValueError("validation manifest and benchmark report commits differ")
        payload["capability"] = capability_record(validation, junit_path)
        payload["run"]["seed"] = validation.get("random_seed")
        payload["run"]["metadata"]["validation"] = {
            "profile": validation.get("profile"),
            "random_cases": validation.get("random_cases"),
            "result": validation.get("result"),
        }
        if validation.get("result") != "pass":
            payload["run"]["status"] = "fail"
    return prune_none(payload)


def program_record(program: Program) -> dict[str, Any]:
    return {
        "id": program.id,
        "source": program.bqn,
        "source_sha256": hashlib.sha256(program.bqn.encode("utf-8")).hexdigest(),
        "category": program.category,
        "variant": program.variant,
        "tags": list(program.tags),
        "input_generator": {
            "name": "bqn-gpu-corpus-v1",
            "input_mode": program.input_mode,
            "domains": program.domains,
            "seed": "little-endian first 8 bytes of SHA-256(program_id + NUL + argument_name + NUL + size)",
        },
        "comparison_policy": {
            "atom_and_shape": "exact",
            "relative_tolerance": program.rtol,
            "absolute_tolerance": program.atol,
            "nan": "equal when both values are NaN",
        },
        "metadata": {
            "arity": program.arity,
            "native_implementations": {
                "tinygrad": program.native_tinygrad,
                "torch": program.native_torch,
            },
            "native_expression_sha256": hashlib.sha256(
                json.dumps(
                    program.native_expression,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


def result_record(result: dict[str, Any], versions: dict[str, str]) -> dict[str, Any]:
    timings = list(result["warm_ns"])
    return {
        "program_id": result["program_id"],
        "backend": result["backend"],
        "backend_version": versions.get(result["backend"]),
        "execution_mode": result["execution_mode"],
        "timing_scope": result["timing_scope"],
        "correct": bool(result["correct"]),
        "input_size": result["size"],
        "cold_ns": result["cold_ns"],
        "median_ns": result["median_warm_ns"],
        "min_ns": result["min_warm_ns"],
        "max_ns": result["max_warm_ns"],
        "p95_ns": percentile(timings, 0.95),
        "timings_ns": timings,
        "metadata": {
            "category": result["category"],
            "variant": result["variant"],
            "tags": result["tags"],
            "device": result["device"],
            "source_sha256": result["source_sha256"],
            "implementation_source_sha256": result.get("implementation_source_sha256"),
            "language": result.get("language"),
            "implementation_kind": result.get("implementation_kind"),
            "framework": result.get("framework"),
            "optimization": result.get("optimization"),
        },
    }


def capability_record(
    validation: dict[str, Any], junit_path: Path | None
) -> dict[str, Any]:
    manifest_path = ROOT / "conformance.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    test_counts = junit_counts(junit_path) if junit_path is not None else {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
    }
    features = []
    supported_monadic = 0
    supported_dyadic = 0
    for index, primitive in enumerate(manifest["primitives"]):
        for valence in ("monadic", "dyadic"):
            claim = primitive[valence]
            if claim["status"] == "supported":
                if valence == "monadic":
                    supported_monadic += 1
                else:
                    supported_dyadic += 1
            features.append(
                {
                    "id": f"primitive.{index:03d}.{valence}",
                    "glyph": primitive["glyph"],
                    "name": primitive["name"],
                    "valence": valence,
                    "status": claim["status"],
                    "domain": claim.get("domain"),
                    "behavior": claim.get("behavior"),
                    "evidence": primitive.get("tests", []),
                }
            )
    fold_names = {
        "+": "sum",
        "×": "product",
        "∧": "and",
        "∨": "or",
        "⌊": "minimum",
        "⌈": "maximum",
    }
    supported_folds = 0
    for index, fold in enumerate(manifest["folds"]):
        if fold["status"] == "supported":
            supported_folds += 1
        features.append(
            {
                "id": f"fold.{fold_names.get(fold['glyph'], index)}",
                "glyph": f"{fold['glyph']}´",
                "name": f"{fold_names.get(fold['glyph'], fold['glyph'])} Fold",
                "valence": "monadic-modifier",
                "status": fold["status"],
                "domain": fold.get("domain"),
                "evidence": ["tests/test_corpus.py"],
            }
        )
    modifier_counts = {"insert": 0, "scan": 0}
    for section, modifier in (("inserts", "˝"), ("scans", "`")):
        kind = section.removesuffix("s")
        for index, entry in enumerate(manifest.get(section, [])):
            if entry["status"] == "supported":
                modifier_counts[kind] += 1
            name = fold_names.get(entry["glyph"], entry["glyph"])
            features.append(
                {
                    "id": f"{kind}.{name}",
                    "glyph": f"{entry['glyph']}{modifier}",
                    "name": f"{name} {kind.title()}",
                    "valence": f"{kind}-modifier",
                    "status": entry["status"],
                    "domain": entry.get("domain"),
                    "evidence": ["tests/test_dense_primitives.py"],
                }
            )
    supported_combinators = 0
    for index, entry in enumerate(manifest.get("combinators", [])):
        if entry["status"] == "supported":
            supported_combinators += 1
        features.append(
            {
                "id": f"combinator.{index:03d}",
                "glyph": entry["glyph"],
                "name": entry["name"],
                "valence": "combinator",
                "status": entry["status"],
                "domain": entry.get("domain"),
                "behavior": entry.get("behavior"),
                "evidence": entry.get("tests", []),
            }
        )
    return {
        "backend": manifest["backend"]["name"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "corpus_programs": len(load_programs()),
        "glyphs_total": len(manifest["primitives"]),
        "monadic_supported": supported_monadic,
        "dyadic_supported": supported_dyadic,
        "folds_supported": supported_folds,
        "tests_passed": test_counts["passed"],
        "tests_failed": test_counts["failed"],
        "tests_skipped": test_counts["skipped"],
        "value_domain": f"{manifest['backend']['dtype']} dense real numeric atoms and arrays; see manifest limitations",
        "manifest": manifest,
        "features": features,
        "metadata": {
            "validation_profile": validation.get("profile"),
            "validation_result": validation.get("result"),
            "validation_timestamp": validation.get("timestamp_utc"),
            "inserts_supported": modifier_counts["insert"],
            "scans_supported": modifier_counts["scan"],
            "combinators_supported": supported_combinators,
        },
    }


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(
        int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
        for suite in suites
    )
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {"passed": tests - failed - skipped, "failed": failed, "skipped": skipped}


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def default_run_id(commit: str, suite: str, timestamp: str) -> str:
    safe_suite = re.sub(r"[^A-Za-z0-9._+-]+", "-", suite).strip("-")
    instant = datetime.fromisoformat(timestamp).strftime("%Y%m%dT%H%M%SZ")
    return f"{commit[:12]}.{safe_suite}.{instant}"


def result_size(report: dict[str, Any]) -> int | list[int]:
    sizes = sorted({int(result["size"]) for result in report["results"]})
    return sizes[0] if len(sizes) == 1 else sizes


def require_full_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("repository commit must be a full lowercase SHA-1")
    return value


def commit_timestamp(commit: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", commit],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def post_json(endpoint: str, data: bytes, token: str) -> Any:
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "bqn-gpu-results-publisher/1",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code < 500 or attempt == 2:
                raise RuntimeError(f"ingestion failed with HTTP {error.code}: {body}") from error
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: prune_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [prune_none(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
