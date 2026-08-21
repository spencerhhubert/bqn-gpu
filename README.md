# bqn-gpu

An experimental GPU execution backend for [BQN](https://mlochbaum.github.io/BQN/), developed against [cBQN](https://github.com/dzaima/CBQN) as the executable semantic oracle.

The first backend uses [tinygrad](https://github.com/tinygrad/tinygrad). Its lazy graph and small code generator let the project test BQN semantics and elementwise fusion now while retaining a short path to custom CUDA kernels and a raw CUDA backend later. tinygrad is an implementation adapter, not the language specification.

## Run BQN

The primary entry point accepts a `.bqn` file or a BQN source string. For example, save this as `sum-squares.bqn`:

```bqn
{
  squares ← 𝕩 × 𝕩
  +´ squares
}
```

Then execute it with a JSON right argument:

```sh
bqn-gpu run sum-squares.bqn --device CUDA --x '[1,2,3,4]'
# 30.0

bqn-gpu eval '{𝕩+1}' --device CPU --x '[1,2,3]'
# {"shape": [3], "data": [2.0, 3.0, 4.0]}

bqn-gpu eval '{𝕩+1}' --backend torch --device CUDA --x '[1,2,3]'
```

`--x` and `--w` accept inline JSON or `@path/to/input.json`. A JSON number is a BQN numeric atom. A rectangular nested list is a dense array. The explicit form `{"shape":[2,2],"data":[1,2,3,4]}` handles exact shapes, including rank-0 and empty arrays. Output is a JSON number for an atom or the explicit shape/data form for an array.

The same path is available from Python:

```python
from bqn_gpu import HostValue, TinygradBackend, execute

result = execute(
    "{𝕩+1}",
    TinygradBackend("CUDA"),
    x=HostValue.from_array([1, 2, 3], (3,)),
)
```

## Current status

The source frontend currently supports headerless function blocks, bare expressions, `𝕨`/`𝕩`, numeric constants, parentheses, BQN right-to-left evaluation, local `←` assignments, statement separators, comments, and Fold over a deliberately small primitive surface. Supported real-number primitives include monadic and dyadic `+ - × ÷ ⌊ ⌈ |`, monadic and dyadic `⋆ √`, dyadic `= ≠ < > ≤ ≥`, monadic Rank `=`, Length `≠`, Shape `≢`, scalar Range `↕`, and Fold with `+ × ⌊ ⌈` on lists.

This is not yet a general BQN compiler. When the pinned cBQN shared library has been built, the CLI delegates unsupported source or backend domains to cBQN and reports the fallback on stderr; the current fallback result must still fit the dense-real numeric boundary. Pass `--fallback error` to require accelerated execution. The exact claimed surface and limitations are generated in [docs/conformance.md](docs/conformance.md), and [docs/source-frontend.md](docs/source-frontend.md) describes the accepted source and data boundary.

The tracked corpus currently contains 200 actual BQN programs and is explicitly designed to grow without a fixed cap. It includes primitive shape cases, phrases, naive/idiomatic pairs, reductions, and long pipelines. Every case is compiled through the BQN source frontend and compared as a complete value against pinned cBQN. See [docs/corpus.md](docs/corpus.md).

Both tinygrad and PyTorch implement the same backend protocol. `scripts/run_corpus.py` correctness-checks and times selected BQN sources across cBQN, tinygrad, and Torch, on CPU or CUDA, and emits stable machine-readable JSON:

```sh
python scripts/run_corpus.py --backend cbqn --backend tinygrad --backend torch \
  --device CUDA --tag paired --size 1048576 --output results.json
```

cBQN is always a CPU reference in this runner. Tensor inputs are transferred before timing and outputs afterward. tinygrad uses a reusable JIT-captured graph after two warmups; Torch currently uses eager dispatch. Every result records its execution mode, cold time, individual warm times, and median. Required device synchronization is inside the timed region, while host/device input and output transfer is excluded.

## Development

Python 3.11 or newer is required.

```sh
python -m pip install -e '.[test]'
./scripts/build_cbqn.sh
make test
```

PyTorch is optional. Install `.[torch]` when it is not already present, or install a CUDA build appropriate for the host before installing this project. tinygrad remains the default backend.

`build_cbqn.sh` clones the revision in `deps/cbqn.rev`, builds its shared embedding library, and places it under `.build/`. Differential tests use cBQN's public embedding API and compare atom/array kind, shape, and numeric data rather than formatted text.

To validate on a CUDA machine:

```sh
python scripts/validate_gpu.py --profile smoke
```

This writes a machine-readable validation manifest and JUnit output under `.build/validation/`. GPU-tested releases can attach those results to the exact tagged commit with the tested hardware and software configuration.

See [docs/architecture.md](docs/architecture.md) for the value model, compiler boundary, backend protocol, cBQN oracle, and path toward raw CUDA and transparent cBQN integration.
