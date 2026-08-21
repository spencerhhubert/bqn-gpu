# bqn-gpu

An experimental GPU backend for [BQN](https://mlochbaum.github.io/BQN/), developed against [cBQN](https://github.com/dzaima/CBQN) as the executable semantic oracle.

The first execution adapter uses [tinygrad](https://github.com/tinygrad/tinygrad). Its small, visible IR and code generator let this project establish BQN semantics now while retaining a short path to custom CUDA kernels and a raw CUDA backend later. tinygrad is an adapter, not the language specification.

## Current status

The first end-to-end slice implements the BQN `+` primitive for real numeric atoms and dense arrays:

- monadic `+` (Conjugate, currently identity for real numbers);
- dyadic `+` (Add);
- atom extension;
- rank-0 arrays, kept distinct from atoms; and
- BQN leading-axis agreement, translated explicitly to the tensor backend's trailing-axis broadcasting model.

Values use `float64`, matching BQN's numeric model for this initial domain. Nested arrays, characters, arbitrary BQN source execution, automatic cBQN fallback, and transparent cBQN acceleration are not implemented yet. Unsupported primitives and invalid shape agreements fail explicitly.

The precise tested surface is in [docs/conformance.md](docs/conformance.md).

## Example

```python
from bqn_gpu import TinygradBackend

backend = TinygradBackend("CUDA")
w = backend.array([10, 20, 30], shape=(3,))
x = backend.array([1, 2, 3, 4, 5, 6], shape=(3, 2))

result = backend.call("+", w, x)
print(result.to_host())
# HostValue(atom=False, shape=(3, 2), data=(11.0, 12.0, 23.0, 24.0, 35.0, 36.0))
```

The rank-1 left argument is reshaped to `(3, 1)` before execution. This preserves BQN's leading-axis agreement instead of accepting the backend's usual trailing-axis interpretation.

## Development

Python 3.11 or newer is required.

```sh
python -m pip install -e '.[test]'
./scripts/build_cbqn.sh
make test
```

`build_cbqn.sh` clones the cBQN revision pinned in `deps/cbqn.rev`, builds its shared embedding library, and places it under `.build/`. The differential tests use cBQN's public embedding API and never compare formatted text.

To validate on a CUDA machine:

```sh
python scripts/validate_gpu.py --profile smoke
```

This runs the cBQN differential suite on CUDA and writes a machine-readable validation manifest plus JUnit output under `.build/validation/`. GPU-tested releases will attach these results to the exact tagged commit along with the tested hardware and software configuration.

## Design

See [docs/architecture.md](docs/architecture.md) for the value model, backend protocol, cBQN oracle, and path toward raw CUDA and transparent cBQN integration.
