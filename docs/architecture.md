# Architecture

## Boundary

The project separates BQN semantics from device execution:

1. A frontend or caller identifies a BQN primitive and valence.
2. A backend-specific value remains resident on its execution device.
3. A semantic adapter validates the supported domain and translates BQN rules into backend operations.
4. Differential tests invoke the same primitive through a pinned cBQN shared library and compare the complete value: atom/array kind, shape, and data.

The current public API starts at step 1 with `TinygradBackend.call`. It does not yet parse arbitrary BQN source or replace cBQN's own primitive dispatch.

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

The semantic layer does not expose tinygrad UOps directly. A future raw CUDA backend will implement the same backend protocol and reuse the BQN agreement rules, program corpus, cBQN oracle, and result comparison.

## Growth path

1. Establish primitive semantics and conformance through tinygrad.
2. Build the cross-language program corpus and a small composable IR.
3. Expand the supported primitive surface and fuse naive compositions.
4. Add correct cBQN fallback for unsupported domains.
5. Add a cBQN integration point for transparent dispatch and persistent device values.
6. Introduce a raw CUDA backend and custom kernels behind the same semantic layer.

Every new conformance claim must identify its supported domain and differential tests in `conformance.json`.
