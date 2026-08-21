from __future__ import annotations

from pathlib import Path
from runpy import run_path


ROOT = Path(__file__).resolve().parents[1]
next_site_tag = run_path(str(ROOT / "scripts" / "release_site.py"))["next_site_tag"]


def test_first_site_tag_starts_at_point_one() -> None:
    assert next_site_tag([]) == "site-v0.1.0"


def test_next_site_tag_increments_highest_patch() -> None:
    assert next_site_tag(
        ["site-v0.4.1", "site-v0.3.9", "site-v0.4.2", "unrelated-v9"]
    ) == "site-v0.4.3"


def test_next_site_tag_uses_semantic_not_lexical_order() -> None:
    assert next_site_tag(["site-v0.9.9", "site-v0.10.0"]) == "site-v0.10.1"
