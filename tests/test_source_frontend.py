from __future__ import annotations

import json

import pytest

from bqn_gpu import HostValue, SourceError, TinygradBackend, compile_bqn, execute
from bqn_gpu.cli import main
from bqn_gpu.json_values import decode_host_value, encode_host_value


def atom(value: float) -> HostValue:
    return HostValue.from_atom(value)


@pytest.mark.parametrize(
    ("source", "arguments", "expected"),
    [
        ("1-2-3", {}, atom(2)),
        ("{𝕩+1}", {"x": atom(4)}, atom(5)),
        ("{-𝕩+1}", {"x": atom(4)}, atom(-5)),
        ("{𝕨×𝕩+1}", {"w": atom(3), "x": atom(4)}, atom(15)),
        ("{a←|𝕩 ⋄ b←a×a ⋄ +´b}", {"x": decode_host_value([1, -2, 3])}, atom(14)),
        ("{π}", {}, atom(3.141592653589793)),
        ("{¯5e¯1}", {}, atom(-0.5)),
    ],
)
def test_source_executes_with_bqn_right_to_left_rules(
    backend: TinygradBackend,
    source: str,
    arguments: dict[str, HostValue],
    expected: HostValue,
) -> None:
    assert execute(source, backend, **arguments) == expected


def test_multiline_comments_and_local_names(backend: TinygradBackend) -> None:
    source = """{
      magnitudes ← |𝕩 # comments end at a newline separator
      squared ← magnitudes × magnitudes
      +´ squared
    }"""
    actual = execute(source, backend, x=decode_host_value([1, -2, 3]))
    assert actual == atom(14)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("{}", "cannot be empty"),
        ("{𝕩+}", "expected a numeric value"),
        ("{missing+1}", "unknown local name"),
        ("{𝕩≠1}", "unsupported BQN token"),
        ("{𝕩+1", "close BQN block"),
    ],
)
def test_source_errors_include_actionable_diagnostics(source: str, message: str) -> None:
    with pytest.raises(SourceError, match=message):
        compile_bqn(source)


def test_argument_arity_is_enforced(backend: TinygradBackend) -> None:
    with pytest.raises(SourceError, match="right argument"):
        execute("𝕩+1", backend)
    with pytest.raises(SourceError, match="both"):
        execute("𝕨+𝕩", backend, x=atom(1))
    with pytest.raises(SourceError, match="does not take"):
        execute("1+2", backend, x=atom(1))


def test_json_value_interchange_preserves_rank() -> None:
    nested = decode_host_value([[1, 2], [3, 4]])
    assert nested == HostValue.from_array([1, 2, 3, 4], (2, 2))
    rank_zero = decode_host_value({"shape": [], "data": [7]})
    assert rank_zero == HostValue.from_array([7], ())
    assert encode_host_value(rank_zero) == {"shape": [], "data": [7.0]}


def test_cli_evaluates_string(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "{𝕩+1}", "--x", "[1,2,3]"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"shape": [3], "data": [2.0, 3.0, 4.0]}


def test_cli_executes_bqn_file_and_reads_json_file(tmp_path, capsys) -> None:
    source_path = tmp_path / "sum-squares.bqn"
    input_path = tmp_path / "input.json"
    source_path.write_text("{a←𝕩×𝕩 ⋄ +´a}\n", encoding="utf-8")
    input_path.write_text("[1,2,3]", encoding="utf-8")
    assert main(["run", str(source_path), "--x", f"@{input_path}"]) == 0
    assert json.loads(capsys.readouterr().out) == 14.0


def test_cli_falls_back_to_cbqn_for_unsupported_source(capsys) -> None:
    assert main(["eval", "{𝕩≠0}", "--x", "[-1,0,2]"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "shape": [3],
        "data": [1.0, 0.0, 1.0],
    }
    assert "cBQN fallback" in captured.err


def test_cli_can_require_acceleration_instead_of_fallback(capsys) -> None:
    assert main(["eval", "{𝕩≠0}", "--x", "1", "--fallback", "error"]) == 2
    assert "unsupported BQN token" in capsys.readouterr().err
