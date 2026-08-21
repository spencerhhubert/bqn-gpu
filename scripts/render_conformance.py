#!/usr/bin/env python3
"""Render the public conformance table from conformance.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "conformance.json"
DESTINATION = ROOT / "docs" / "conformance.md"


def render() -> str:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    backend = manifest["backend"]
    lines = [
        "# Conformance",
        "",
        "This file is generated from `conformance.json`. Do not edit it by hand.",
        "",
        f"Semantic oracle: **{manifest['reference']['implementation']}**, pinned by "
        f"`{manifest['reference']['revision_file']}`.",
        "",
        f"Execution backend: **{backend['name']}** at `{backend['revision']}`, "
        f"using `{backend['dtype']}` on {', '.join(f'`{d}`' for d in backend['devices'])}.",
        "",
    ]
    for additional in manifest.get("additional_backends", []):
        lines.extend(
            [
            f"Additional adapter: **{additional['name']}** `{additional['version']}`, "
            f"using `{additional['dtype']}` on "
            f"{', '.join(f'`{d}`' for d in additional['devices'])}; "
            f"tested by {', '.join(f'`{test}`' for test in additional['tests'])}.",
            "",
            ]
        )
    lines.extend(
        [
            "| Primitive | Monad | Dyad | Tests |",
            "|---|---|---|---|",
        ]
    )
    for primitive in manifest["primitives"]:
        tests = "<br>".join(f"`{test}`" for test in primitive["tests"])
        lines.append(
            f"| `{primitive['glyph']}` {primitive['name']} "
            f"| {primitive['monadic']['status']} "
            f"| {primitive['dyadic']['status']} "
            f"| {tests} |"
        )

    for primitive in manifest["primitives"]:
        lines.extend(
            [
                "",
                f"## `{primitive['glyph']}` — {primitive['name']}",
                "",
                "### Monadic",
                "",
                f"Status: **{primitive['monadic']['status']}**",
                "",
                f"Domain: {primitive['monadic']['domain']}",
                "",
                primitive["monadic"]["behavior"],
                "",
                "### Dyadic",
                "",
                f"Status: **{primitive['dyadic']['status']}**",
                "",
                f"Domain: {primitive['dyadic']['domain']}",
                "",
                primitive["dyadic"]["behavior"],
                "",
                "### Limitations",
                "",
            ]
        )
        lines.extend(
            f"- {limitation}"
            for limitation in manifest.get("common_limitations", [])
            + primitive.get("limitations", [])
        )

    for title, key, modifier in (
        ("Fold", "folds", "´"),
        ("Insert", "inserts", "˝"),
        ("Scan", "scans", "`"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Function operand | Status | Domain |",
                "|---|---|---|",
            ]
        )
        for entry in manifest[key]:
            lines.append(
                f"| `{entry['glyph']}{modifier}` | {entry['status']} | {entry['domain']} |"
            )

    source = manifest["source_frontend"]
    lines.extend(
        [
            "",
            "## BQN source frontend",
            "",
            source["summary"],
            "",
            "| Construct | Status | Constraint |",
            "|---|---|---|",
        ]
    )
    for construct in source["constructs"]:
        lines.append(
            f"| {construct['name']} | {construct['status']} | {construct['constraint']} |"
        )
    lines.extend(["", "Source frontend tests:", ""])
    lines.extend(f"- `{test}`" for test in source["tests"])

    lines.extend(
        [
            "",
            "## Meaning of status",
            "",
            "- **supported**: the stated domain is covered by automated differential tests against the pinned cBQN revision.",
            "- **fallback**: the backend deliberately delegates to a semantically correct non-GPU implementation.",
            "- **unsupported**: the backend rejects the operation or domain explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render()
    if arguments.check:
        if not DESTINATION.exists() or DESTINATION.read_text(encoding="utf-8") != rendered:
            print(f"{DESTINATION.relative_to(ROOT)} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
