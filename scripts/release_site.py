#!/usr/bin/env python3
"""Verify and publish the next patch release of the website."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SITE_TAG = re.compile(r"site-v(\d+)\.(\d+)\.(\d+)")
GENERATED_TYPES = SITE / "worker-configuration.d.ts"


def output(*arguments: str) -> str:
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def run(*arguments: str, cwd: Path = ROOT) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def next_site_tag(tags: Iterable[str]) -> str:
    versions = [
        tuple(int(part) for part in match.groups())
        for tag in tags
        if (match := SITE_TAG.fullmatch(tag))
    ]
    if not versions:
        return "site-v0.1.0"
    major, minor, patch = max(versions)
    return f"site-v{major}.{minor}.{patch + 1}"


def normalized_generated_types(contents: bytes) -> bytes:
    """Ignore Wrangler's unstable trailing spaces, but no semantic changes."""

    return b"\n".join(line.rstrip() for line in contents.splitlines())


def check_site() -> None:
    original_types = GENERATED_TYPES.read_bytes()
    try:
        run("npm", "run", "check", cwd=SITE)
    finally:
        generated_types = GENERATED_TYPES.read_bytes()
        if normalized_generated_types(generated_types) == normalized_generated_types(
            original_types
        ):
            GENERATED_TYPES.write_bytes(original_types)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the website and push its next site-v* deployment tag."
    )
    parser.add_argument(
        "--message",
        help="annotated tag message (defaults to 'Deploy <tag>')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every check and print the next tag without creating it",
    )
    arguments = parser.parse_args()

    run("git", "fetch", "origin", "main", "--tags")
    if output("git", "branch", "--show-current") != "main":
        raise SystemExit("website releases must be made from main")
    if output("git", "status", "--porcelain"):
        raise SystemExit("website releases require a clean working tree")
    head = output("git", "rev-parse", "HEAD")
    if head != output("git", "rev-parse", "origin/main"):
        raise SystemExit("local main must exactly match origin/main")

    tags = output("git", "tag", "--list", "site-v*").splitlines()
    tag = next_site_tag(tags)
    check_site()
    if output("git", "status", "--porcelain"):
        raise SystemExit("website checks changed tracked files; commit them first")

    if arguments.dry_run:
        print(f"ready to tag {head} as {tag}")
        return 0

    message = arguments.message or f"Deploy {tag}"
    run("git", "tag", "-a", tag, "-m", message)
    run("git", "push", "origin", f"refs/tags/{tag}")
    print(f"pushed {tag} at {head}; the Website workflow will deploy it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
