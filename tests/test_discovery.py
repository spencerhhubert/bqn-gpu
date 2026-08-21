from __future__ import annotations

import math

from bqn_gpu import HostValue, compile_bqn
from bqn_gpu.discovery import generate_programs


def test_typed_generation_is_reproducible_and_unique() -> None:
    first = generate_programs(seed=17, count=20, min_steps=6, max_steps=12)
    second = generate_programs(seed=17, count=20, min_steps=6, max_steps=12)

    assert [program.as_dict() for program in first] == [
        program.as_dict() for program in second
    ]
    assert len({program.id for program in first}) == 20
    assert {program.strategy for program in first} == {"grammar", "mutation"}
    assert all(program.steps >= 6 for program in first)


def test_generated_programs_match_cbqn_and_equivalent_mutations(
    backend,
    cbqn,
) -> None:
    programs = generate_programs(seed=29, count=12, min_steps=5, max_steps=10)
    value = HostValue.from_array(
        (math.sin(index) for index in range(19)),
        (19,),
    )
    for program in programs:
        actual = compile_bqn(program.bqn).execute(backend, x=value)
        expected = cbqn.call(program.bqn, value)
        assert actual.shape == expected.shape
        assert actual.atom == expected.atom
        assert all(
            math.isclose(got, wanted, rel_tol=1e-11, abs_tol=1e-11)
            for got, wanted in zip(actual.data, expected.data, strict=True)
        )
        if program.equivalent_to_bqn is not None:
            assert cbqn.call(program.equivalent_to_bqn, value) == expected
