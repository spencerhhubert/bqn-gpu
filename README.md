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

`bqn-gpu explain` reports the semantic IR, argument kind/shape specialization, optimized IR, every named rewrite, and whether tensor work remains. It does not execute the program:

```sh
bqn-gpu explain '{⌽⌽𝕩}' --x '[1,2,3]'
# optimized_bqn: 𝕩
# rewrite: double-reverse
```

## Current status

The source frontend currently supports headerless function blocks, bare
expressions, `𝕨`/`𝕩`, numeric constants and strands, parentheses, BQN
right-to-left evaluation, local `←` assignments, statement separators, and
comments. Its dense-real tier covers arithmetic, logic, comparison, structural,
ordering, search, Fold/Insert/Scan, the pure combinators Self/Swap, Atop,
Over, Before/Bind, and After/Bind, dense mapping modifiers, and parenthesized
function trains. Every claimed valence and domain is listed in
the generated conformance document.

This is not yet a general BQN compiler. When the pinned cBQN shared library has been built, the CLI delegates unsupported source or backend domains to cBQN and reports the fallback on stderr; the current fallback result must still fit the dense-real numeric boundary. Pass `--fallback error` to require accelerated execution. The exact claimed surface and limitations are generated in [docs/conformance.md](docs/conformance.md), and [docs/source-frontend.md](docs/source-frontend.md) describes the accepted source and data boundary.

The tracked corpus currently contains 304 actual BQN programs and is explicitly designed to grow without a fixed cap. It includes primitive shape cases, phrases, naive/idiomatic pairs, reductions, structural transforms, ordering, modifiers, combinators, function trains, dense mapping/rank algorithms, and long pipelines. Every case is compiled through the BQN source frontend and compared as a complete value against pinned cBQN. See [docs/corpus.md](docs/corpus.md).

Deterministic typed-grammar generation and equivalence mutation probe combinations that curated cases may miss. Candidates are checked against cBQN and remain diagnostic until a correctness failure, compiler path, scaling gap, or missed simplification earns promotion into the tracked corpus. See [docs/generative-testing.md](docs/generative-testing.md).

The benchmark corpus keeps three independent forms of every workload: the actual BQN source, direct native tinygrad source, and direct native PyTorch source. The native sources are generated from the workload specification rather than parsed or lowered from BQN. `scripts/run_corpus.py` checks all results against cBQN and records four explicitly named implementations: cBQN on CPU, the custom BQN frontend/backend on tinygrad, direct tinygrad, and direct PyTorch. For example:

```sh
python scripts/run_corpus.py --backend cbqn \
  --backend bqn-gpu-tinygrad --backend native-tinygrad --backend native-torch \
  --device CUDA --tag paired --size 1048576 --output results.json
```

For ordinary compiler iterations, `--profile development` runs a fixed two-size sentinel subset. `--profile certification` is its strict superset and is reserved for occasional tagged measurements. This keeps fast and long results directly comparable without spending most development time benchmarking.

cBQN is always a CPU reference in this runner. By default, cBQN arguments and tensor inputs are both constructed before timing, so `resident-compute` rows compare the same execution boundary. `--cbqn-timing-scope boundary` separately measures cBQN's current `HostValue` embedding copies and marks those rows `host-value-boundary`; they must not be used for resident GPU speedup claims. The bqn-gpu and native tinygrad implementations use reusable JIT-captured graphs; native Torch uses ordinary eager PyTorch. Every result records language, implementation kind, framework, physical device, timing scope, execution mode, source hash, cold time, raw warm repetitions, and aggregates. Required device synchronization is inside tensor timings, while host/device transfer is excluded.

Performance and capability history is published on the [bqn-gpu website](https://bqn-gpu-website.spencerhhubert.workers.dev) by the TypeScript Cloudflare Worker in [`site/`](site/). Its D1 schema keeps raw timings together with the tested commit, BQN source hashes, full input recipe, correctness result, and fingerprinted hardware/software environment. Tagged `site-v*` releases automatically deploy the dashboard and API; see [`site/README.md`](site/README.md) for the versioned ingestion contract and reproduction endpoints.

Schema-version 2 corpus reports capture the CPU topology, memory, accelerators, drivers, kernel, Python, cBQN, and backend versions without recording hostnames or infrastructure addresses. A clean report can be validated and published with:

```sh
python3 scripts/publish_results.py results.json \
  --suite paired-1m \
  --validation-manifest .build/validation/gpu-validation.json \
  --junit .build/validation/junit.xml \
  --dry-run --output-payload payload.json
```

Remove `--dry-run` and provide `BQN_GPU_RESULTS_TOKEN` to ingest it. The publisher rejects dirty benchmark reports by default, verifies program source hashes and commit agreement, and can attach the complete conformance manifest plus JUnit counts.

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
