# Dense numeric algorithm tier

The next language milestone is not defined by an arbitrary primitive count. It is the largest coherent BQN subset whose values can remain dense real numeric atoms and arrays from source through GPU execution.

This tier is intended to express substantial array algorithms rather than isolated arithmetic pipelines. Its completion boundary is:

- all pervasive arithmetic, logic, comparison, match, dimension, and identity functions;
- dense restructuring: deshape/reshape, join-to, solo/couple, reverse/rotate, transpose/reorder-axes, in-bounds take/drop, windows, and dense shifts once fill metadata is represented;
- dense indexing and ordering: indices/replicate, grade/bins, first/select, first/pick, classify/index-of, occurrence/progressive-index-of, mark-firsts/member-of, deduplicate/find;
- Fold, Insert, and Scan for supported dyadic functions;
- the pure combinators Self/Swap, Atop/Over, and Before/After;
- Cells/Rank, Each, and Table when every produced cell has one uniform dense numeric shape; and
- numeric strands and list notation needed to state shapes, axes, indices, and constants.

The tier deliberately excludes behavior that inherently crosses the current value boundary:

- characters, strings, namespaces, and operation values as data;
- irregular nested arrays and mixed-depth results;
- Enclose, Merge, Pair, Prefixes/Suffixes, multidimensional Range, and Group outside dense special cases;
- fill-sensitive expansion until fill elements are carried explicitly by the value model; and
- recursion, mutation, exceptions, system values, and general dynamic control flow.

An excluded general case does not prevent a dense special case from being supported, but the domain must be stated in `conformance.json` and tested against cBQN. Unsupported cases must fall back or fail explicitly.

## Implementation order

1. Add source and semantic-IR support without erasing the original BQN operation.
2. Establish deterministic and randomized cBQN differential tests for the claimed dense domain.
3. Add a correct backend lowering, initially using framework operations where appropriate.
4. Lower structural operations to index maps and fuse them with neighboring elementwise work.
5. Add specialized GPU algorithms or custom kernels only when measurements identify a material gap.

Correctness iterations use the CPU oracle suite and a short CUDA semantic sweep. A small performance-sentinel set tracks compiler regressions during development. The full corpus across all benchmark sizes is reserved for tagged capability or optimizer milestones.
