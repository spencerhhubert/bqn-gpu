# Architecture

## Boundary

The project separates BQN semantics from device execution:

1. The source frontend accepts a `.bqn` file or BQN string and lowers the supported syntax to a small expression IR.
2. The semantic HIR preserves BQN primitives and modifiers before dispatch through a framework-independent backend protocol.
3. A backend-specific value remains resident on its execution device while a semantic adapter validates domains and translates BQN rules.
4. Differential tests evaluate the original BQN source through pinned cBQN and compare the complete value: atom/array kind, shape, and data.

The CLI, `compile_bqn`, and `execute` start at step 1. `TinygradBackend.call` remains the lower-level adapter boundary. The source subset is documented precisely; this does not yet replace cBQN's own primitive dispatch.

## Value model

BQN distinguishes numeric atoms from rank-0 arrays. Tensor libraries represent both naturally as zero-dimensional tensors, so `TinygradValue` retains a separate `atom` bit. Losing that distinction would make shape and type observations incorrect even when the numeric payload matched.

The supported value domain is currently dense, real, `float64` data. `TinygradValue` keeps it device-resident and lazy. `HostValue` is a backend-independent serialization used at tests and integration boundaries.

## Leading-axis agreement

BQN pairs a lower-rank array with the leading axes of a higher-rank array. Tensor libraries generally broadcast along trailing axes. For BQN shapes `(3,)` and `(3, 2)`, the adapter reshapes the lower-rank tensor to `(3, 1)` before performing the operation.

The adapter first proves that the lower-rank shape is a prefix of the higher-rank shape. It never accepts a pair merely because the underlying tensor library could broadcast it under different rules.

## cBQN oracle

`scripts/build_cbqn.sh` builds the revision in `deps/cbqn.rev` as a shared library. The test harness uses cBQN's public embedding interface to construct values, evaluate primitives, invoke them, and read the result's kind, shape, and numeric data.

The oracle never depends on formatted textual output. Random differential cases record their seed so a failure can be reproduced.

## Why tinygrad first

tinygrad supplies allocation, lazy graphs, fusion, scheduling, code generation, and CUDA execution behind a small readable implementation. Laziness is important to the project goal: a naively written BQN elementwise pipeline should be fused rather than forced into one GPU launch per source primitive.

The semantic layer does not expose tinygrad UOps directly. A future raw CUDA backend will implement the same backend protocol and reuse the BQN agreement rules, program corpus, cBQN oracle, and result comparison. The staged optimization contract is described in [compiler.md](compiler.md); the broader language milestone is defined in [dense-numeric-tier.md](dense-numeric-tier.md), and [generative-testing.md](generative-testing.md) defines typed discovery and equivalence mutation.

PyTorch is a second implementation of that protocol and a familiar performance baseline. The source compiler and corpus do not contain Torch-specific programs: the same BQN source IR dispatches to either tensor backend. This makes backend discrepancies visible and prevents benchmark definitions from drifting apart.

CUDA validation runs the tinygrad and PyTorch adapter suites in fresh processes. Each framework owns CUDA runtime and context state, and process isolation prevents one adapter's tests from leaving stale handles in the other while retaining the same commit, hardware, inputs, and semantic oracle.

## Growth path

1. Establish primitive semantics and conformance through tinygrad.
2. Build the actual-BQN program corpus, source frontend, and small composable IR.
3. Complete the coherent dense numeric algorithm tier while retaining high-level array operations in semantic HIR.
4. Add correct cBQN fallback for unsupported domains.
5. Add a cBQN integration point for transparent dispatch and persistent device values.
6. Lower semantic HIR through array/index IR, introduce a raw CUDA backend, and add tuned custom kernels behind the same semantic layer.

Every new conformance claim must identify its supported domain and differential tests in `conformance.json`.
