from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from bqn_gpu import TinygradBackend, compile_bqn
from bqn_gpu.cbqn import CBQN
from bqn_gpu.corpus import Program, assert_close, generate_inputs, load_programs


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = load_programs()


def test_tracked_corpus_is_current_and_grows_beyond_its_initial_floor() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_corpus.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    raw = json.loads((ROOT / "corpus/programs.json").read_text(encoding="utf-8"))
    assert raw["policy"]["initial_floor"] == 100
    assert raw["schema_version"] == 2
    assert "append durable cases" in raw["policy"]["growth"]
    assert len(PROGRAMS) >= 100
    assert len({program.id for program in PROGRAMS}) == len(PROGRAMS)
    assert all(program.native_tinygrad.startswith("lambda") for program in PROGRAMS)
    assert all(program.native_torch.startswith("lambda") for program in PROGRAMS)


def test_corpus_has_glyph_phrase_pair_reduction_and_long_program_layers() -> None:
    categories = {program.category for program in PROGRAMS}
    assert {"glyph", "phrase", "paired", "reduction", "program"} <= categories
    assert {
        "dense-primitive",
        "dense-structural",
        "dense-modifier",
        "dense-phrase",
        "dense-paired",
    } <= categories
    assert sum(len("".join(program.bqn.split())) >= 40 for program in PROGRAMS) >= 20
    assert sum(len("".join(program.bqn.split())) >= 80 for program in PROGRAMS) >= 5

    pair_variants: dict[str, set[str]] = {}
    for program in PROGRAMS:
        pair_tags = [tag for tag in program.tags if tag.startswith("pair-")]
        for pair in pair_tags:
            pair_variants.setdefault(pair, set()).add(program.variant)
    assert len(pair_variants) >= 10
    assert all(variants == {"naive", "idiomatic"} for variants in pair_variants.values())


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda program: program.id)
def test_actual_bqn_corpus_source_matches_cbqn(
    program: Program,
    backend: TinygradBackend,
    cbqn: CBQN,
) -> None:
    compiled = compile_bqn(program.bqn)
    assert compiled.arity == program.arity
    inputs = generate_inputs(program, size=31)
    actual = compiled.execute(backend, **inputs)
    arguments = (inputs["x"],) if program.arity == 1 else (inputs["w"], inputs["x"])
    expected = cbqn.call(program.bqn, *arguments)
    assert_close(actual, expected, program)
