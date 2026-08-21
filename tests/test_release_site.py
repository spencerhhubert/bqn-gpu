from __future__ import annotations

from pathlib import Path
from runpy import run_path


ROOT = Path(__file__).resolve().parents[1]
release_site = run_path(str(ROOT / "scripts" / "release_site.py"))
next_site_tag = release_site["next_site_tag"]
normalized_generated_types = release_site["normalized_generated_types"]


def test_first_site_tag_starts_at_point_one() -> None:
    assert next_site_tag([]) == "site-v0.1.0"


def test_next_site_tag_increments_highest_patch() -> None:
    assert next_site_tag(
        ["site-v0.4.1", "site-v0.3.9", "site-v0.4.2", "unrelated-v9"]
    ) == "site-v0.4.3"


def test_next_site_tag_uses_semantic_not_lexical_order() -> None:
    assert next_site_tag(["site-v0.9.9", "site-v0.10.0"]) == "site-v0.10.1"


def test_generated_type_comparison_ignores_only_trailing_whitespace() -> None:
    assert normalized_generated_types(b"type A = \n  string;\n") == (
        normalized_generated_types(b"type A =\n  string;\n")
    )
    assert normalized_generated_types(b"type A = string;\n") != (
        normalized_generated_types(b"type A = number;\n")
    )
