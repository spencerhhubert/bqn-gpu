# Optimizing compiler contract

BQN's high-level array operations are optimization opportunities. The compiler must not irreversibly decompose them into a sequence of eager framework calls before it knows shapes, layouts, consumers, and the surrounding program.

The intended pipeline is:

```text
BQN source
  -> semantic HIR (BQN glyphs, modifiers, valence, atom/array distinction)
  -> shape/type specialization and legality checks
  -> array/index IR (iteration domains, index maps, reductions, scans, gathers)
  -> fusion, layout, placement, and cost decisions
  -> scheduled kernel IR
  -> tinygrad, generated CUDA, or a correct host fallback
```

## Semantic HIR

The high-level IR retains primitive identity and BQN evaluation rules. It is the common input to the reference interpreter and optimized compilation. It must preserve atom versus rank-0-array kind, leading-axis agreement, cell rank, fill dependencies, and exact structural domains.

Every lowering or optimization is checked by evaluating the original source with pinned cBQN. Optimized and unoptimized execution must also agree so that a compiler pass cannot silently redefine the language.

`bqn-gpu explain SOURCE --x JSON` exposes the semantic and optimized IR plus named rewrite events for a concrete argument signature. Compiler observability is a versioned machine-readable output, not only debug logging, so benchmark records can eventually retain the exact rewrite and lowering decisions that produced a kernel.

## Specialization

GPU programs are specialized by the facts that materially change generated code: argument kind, dtype, rank, shape or symbolic shape constraints, layout, device, and semantic options. Compilation artifacts are cached by this signature. Dynamic shapes remain possible, but a hot stable shape should reach a branch-free specialized kernel.

Shape-only expressions are evaluated outside kernels when possible. Structural values used as indices or axes are kept on-device when doing so avoids synchronization and enables a useful fused kernel.

## Array/index IR

Elementwise operations, broadcasts, reshapes, transposes, reverses, slices, rotates, windows, and regular selections should compose as index maps. A consumer can then read its producer through the composed map without materializing every intermediate array.

Materialization remains available when reuse, irregular access, register pressure, or launch occupancy makes it cheaper. The compiler records layouts and aliases explicitly rather than relying on incidental tensor-framework behavior.

Reductions, scans, sorting, searching, grouping, and irregular gathers are first-class operations. They are not encoded as opaque Python loops. Each may select among library lowering, generated multi-stage kernels, and tuned custom kernels.

## Optimization policy

The default objective is end-to-end resident execution time subject to exact BQN semantics. Important passes include:

- constant, shape, and index-expression folding;
- algebraic simplification guarded by floating-point legality;
- elementwise and producer/consumer fusion;
- fusion of structural index maps into elementwise consumers;
- reduction and scan fusion where ordering permits it;
- dead result and unused argument elimination;
- layout propagation and transpose avoidance;
- common-subexpression reuse when cheaper than recomputation;
- launch coalescing, vectorization, memory coalescing, and occupancy-aware scheduling; and
- CPU/GPU placement based on transfer, launch, and computation costs.

Autotuning decisions must be keyed to reproducible hardware and shape signatures. A tuned result is an optimization choice, never part of language semantics.

The initial dispatch cost policy avoids tinygrad JIT replay for expressions that
only construct layout views (reshape, solo, reverse, transpose, and identity).
A fused tensor consumer and static structural call still select JIT replay, and
rewrites that erase all data-dependent work select a direct optimized no-op.
The selected mode and reason are emitted by `bqn-gpu explain` and retained with
benchmark results. This conservative static policy is a starting point for
measured, hardware-keyed decisions rather than a permanent list of special
cases.

Dense mapping remains explicit as Cells/Rank/Each/Table IR until argument shapes
and frames are known. Pervasive Each calls and primitive Table calls lower to
one broadcast tensor expression. The common BQN matrix-vector form
`𝕨 +˝∘×⎉1 𝕩` is recognized after Rank planning and lowered to one broadcast
multiply/reduction graph rather than one host-constructed reduction per row.
Other uniform-result mapped functions currently use a correct cell planner and
are candidates for further batched lowering when measurements justify it.

Parenthesized function trains remain explicit in semantic IR. After argument
specialization they expand according to BQN two-train composition and
right-associated three-train fork rules. Nested trains and derived functions
expand recursively, producing the same ordinary tensor IR as an explicitly
written block and recording each train-inline decision in the explanation.

Valences (`⊘`) remains explicit in semantic IR until call valence is known.
It then selects its left operand for a monadic call or its right operand for a
dyadic call before ordinary optimization, so the unselected branch creates no
backend work.

Repeat with a literal natural count remains explicit in semantic IR and is
unrolled only after the complete source expression is known. Counts are capped
at 64, and the expanded expression is capped at 4,096 semantic IR nodes, so an
operand that duplicates its input cannot grow the compiler graph without bound. Zero
repetition specializes to the right argument; positive monadic or dyadic
repetition becomes one fixed graph, with the dyadic left argument reused on each
application as required by BQN.

Undo (`⁼`) also remains explicit until semantic optimization. The compiler
proves an inverse from the operand structure and replaces it with ordinary
tensor IR: required arithmetic inverses, logarithm, reverse/rotate, selected
Self cases, mapped inverses, Valences, constant-left Bind, Atop, two-function
trains, and double Undo are covered within the documented dense-real domain.
The lowering treats a dyadic left argument as fixed, following BQN's
right-inverse rule. Unsupported inverse proofs remain compile errors and can be
delegated to cBQN by the CLI; there is no runtime function dispatch in a proven
GPU graph.

Monadic Classify, Occurrence Count, Mark Firsts, and Deduplicate compare major
cells, flattening only each cell's trailing axes for equality while preserving
BQN's leading-axis semantics. The first three have fixed list output and remain
JIT-capturable. Deduplicate has a data-dependent leading-axis length and uses an
explicit eager boundary. The initial equality plan is quadratic in the number
of major cells; the recorded benchmark sentinel makes a future exact hash or
sort-based GPU kernel directly comparable.

Nudge and Shift (`»«`) lower to fixed-shape leading-axis slice and
concatenation graphs. Monadic numeric shifts synthesize one zero-fill major
cell; dyadic shifts accept one cell or an array of cells and retain exactly the
right argument's shape.

## Measurement discipline

Development uses a compact sentinel set covering elementwise fusion, structural index-map fusion, reductions, scans, selection, sorting/search, and mixed complex programs. Full multi-size measurements are recorded only for tagged milestones. Raw repetitions, exact commits, hardware, compiler choices, and correctness evidence remain reproducible through the results service.
