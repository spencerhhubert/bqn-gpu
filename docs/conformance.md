# Conformance

This file is generated from `conformance.json`. Do not edit it by hand.

Semantic oracle: **cBQN**, pinned by `deps/cbqn.rev`.

Execution backend: **tinygrad** at `2d48fe8b7bd9acfa00e91a7f89b28b3ded370c27`, using `float64` on `CPU`, `CUDA`.

Additional adapter: **PyTorch** `>=2.7`, using `float64` on `CPU`, `CUDA`; tested by `tests/test_torch_backend.py`.

| Primitive | Monad | Dyad | Tests |
|---|---|---|---|
| `+` Conjugate / Add | supported | supported | `tests/test_add_conformance.py`<br>`tests/test_corpus.py` |
| `-` Negate / Subtract | supported | supported | `tests/test_corpus.py` |
| `×` Sign / Multiply | supported | supported | `tests/test_corpus.py` |
| `÷` Reciprocal / Divide | supported | supported | `tests/test_corpus.py` |
| `⋆` Exponential / Power | supported | supported | `tests/test_corpus.py` |
| `√` Square Root / Root | supported | supported | `tests/test_corpus.py` |
| `⌊` Floor / Minimum | supported | supported | `tests/test_corpus.py` |
| `⌈` Ceiling / Maximum | supported | supported | `tests/test_corpus.py` |
| `|` Absolute Value / Modulus | supported | supported | `tests/test_corpus.py` |
| `=` Rank / Equals | supported | supported | `tests/test_corpus.py` |
| `≠` Length / Not Equals | supported | supported | `tests/test_corpus.py` |
| `<` Enclose / Less Than | unsupported | supported | `tests/test_corpus.py` |
| `>` Merge / Greater Than | unsupported | supported | `tests/test_corpus.py` |
| `≤` Less Than or Equal | unsupported | supported | `tests/test_corpus.py` |
| `≥` Greater Than or Equal | unsupported | supported | `tests/test_corpus.py` |
| `≢` Shape / Not Match | supported | supported | `tests/test_corpus.py`<br>`tests/test_dense_primitives.py` |
| `≡` Depth / Match | supported | supported | `tests/test_dense_primitives.py` |
| `∧` Sort Up / Logical And | supported | supported | `tests/test_dense_primitives.py` |
| `∨` Sort Down / Logical Or | supported | supported | `tests/test_dense_primitives.py` |
| `¬` Not / Span | supported | supported | `tests/test_dense_primitives.py`<br>`tests/test_source_frontend.py` |
| `⊣` Identity / Left | supported | supported | `tests/test_dense_primitives.py` |
| `⊢` Identity / Right | supported | supported | `tests/test_dense_primitives.py` |
| `⥊` Deshape / Reshape | supported | supported | `tests/test_dense_primitives.py`<br>`tests/test_source_frontend.py` |
| `∾` Join / Join To | unsupported | supported | `tests/test_dense_primitives.py` |
| `≍` Solo / Couple | supported | supported | `tests/test_dense_primitives.py`<br>`tests/test_source_frontend.py` |
| `⋈` Enlist / Pair | supported | supported | `tests/test_dense_primitives.py` |
| `↑` Prefixes / Take | unsupported | supported | `tests/test_dense_primitives.py` |
| `↓` Suffixes / Drop | unsupported | supported | `tests/test_dense_primitives.py` |
| `⌽` Reverse / Rotate | supported | supported | `tests/test_dense_primitives.py` |
| `⍉` Transpose / Reorder Axes | supported | supported | `tests/test_dense_primitives.py` |
| `/` Indices / Replicate | supported | supported | `tests/test_dense_primitives.py` |
| `⍋` Grade Up / Bins Up | supported | supported | `tests/test_dense_primitives.py` |
| `⍒` Grade Down / Bins Down | supported | supported | `tests/test_dense_primitives.py` |
| `⊏` First Cell / Select | supported | supported | `tests/test_dense_primitives.py` |
| `⊑` First / Pick | supported | supported | `tests/test_dense_primitives.py` |
| `⊐` Classify / Index Of | supported | supported | `tests/test_dense_primitives.py` |
| `⊒` Occurrence Count / Progressive Index Of | supported | supported | `tests/test_dense_primitives.py` |
| `∊` Mark Firsts / Member Of | supported | supported | `tests/test_dense_primitives.py` |
| `⍷` Deduplicate / Find | supported | supported | `tests/test_dense_primitives.py` |
| `↕` Range / Windows | supported | supported | `tests/test_corpus.py`<br>`tests/test_dense_primitives.py` |

## `+` — Conjugate / Add

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense real numeric arrays of any rank and shape.

Identity for the current real-only domain.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise addition with atom extension and explicit leading-axis agreement.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `-` — Negate / Subtract

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise negation.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise subtraction with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `×` — Sign / Multiply

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise sign.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise multiplication with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `÷` — Reciprocal / Divide

### Monadic

Status: **supported**

Domain: Nonzero real numeric atoms and dense arrays in the tested corpus.

Element-wise reciprocal.

### Dyadic

Status: **supported**

Domain: Real numerator and nonzero real denominator satisfying BQN leading-axis agreement.

Element-wise division with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⋆` — Exponential / Power

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise natural exponential.

### Dyadic

Status: **supported**

Domain: Positive real base and real exponent satisfying BQN leading-axis agreement in the tested corpus.

Element-wise Power with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `√` — Square Root / Root

### Monadic

Status: **supported**

Domain: Positive real numeric atoms and dense arrays in the tested corpus.

Element-wise square root.

### Dyadic

Status: **supported**

Domain: Positive real degree and radicand satisfying BQN leading-axis agreement in the tested corpus.

Element-wise Root, raising the right argument to the reciprocal of the left.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⌊` — Floor / Minimum

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise floor.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise minimum with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⌈` — Ceiling / Maximum

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise ceiling.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise maximum with atom extension.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `|` — Absolute Value / Modulus

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise absolute value.

### Dyadic

Status: **supported**

Domain: Nonzero real left argument and real right argument satisfying BQN leading-axis agreement.

BQN modulus, implemented as x minus w times floor(x divided by w).

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `=` — Rank / Equals

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays of any tested shape, including rank-0 and empty arrays.

Returns the number of axes as a numeric atom.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise atomic equality, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≠` — Length / Not Equals

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays of any tested shape, including rank-0 and empty arrays.

Returns the first-axis length, or one for an atom or rank-0 array.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise atomic inequality, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `<` — Enclose / Less Than

### Monadic

Status: **unsupported**

Domain: Nested results are outside the current value boundary.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise numeric less-than, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `>` — Merge / Greater Than

### Monadic

Status: **unsupported**

Domain: No monadic Merge domain is claimed yet.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise numeric greater-than, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≤` — Less Than or Equal

### Monadic

Status: **unsupported**

Domain: This glyph has no monadic primitive meaning.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise numeric less-than-or-equal, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≥` — Greater Than or Equal

### Monadic

Status: **unsupported**

Domain: This glyph has no monadic primitive meaning.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Element-wise numeric greater-than-or-equal, producing numeric zero or one.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≢` — Shape / Not Match

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays of any tested shape, including rank-0 and empty arrays.

Returns the shape as a numeric list; units produce an empty list.

### Dyadic

Status: **supported**

Domain: Real numeric atoms or dense arrays; arguments may differ in kind, shape, or rank.

Returns numeric one unless atom/array kind, shape, and every element match.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≡` — Depth / Match

### Monadic

Status: **supported**

Domain: Real numeric atoms and non-nested dense real arrays.

Returns zero for an atom and one for a dense array.

### Dyadic

Status: **supported**

Domain: Real numeric atoms or dense arrays; arguments may differ in kind, shape, or rank.

Returns numeric one exactly when atom/array kind, shape, and every element match.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `∧` — Sort Up / Logical And

### Monadic

Status: **supported**

Domain: Real numeric lists.

Sorts the list in ascending numeric order.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Computes BQN numeric logical And as multiplication.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `∨` — Sort Down / Logical Or

### Monadic

Status: **supported**

Domain: Real numeric lists.

Sorts the list in descending numeric order.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Computes BQN numeric logical Or as w+x-w×x.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `¬` — Not / Span

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Computes one minus the argument element-wise.

### Dyadic

Status: **supported**

Domain: Real arguments satisfying BQN leading-axis agreement.

Computes one plus the left argument minus the right argument element-wise.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊣` — Identity / Left

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Returns the argument without changing its kind or shape.

### Dyadic

Status: **supported**

Domain: Any two values inside the dense-real boundary.

Returns the left argument.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊢` — Identity / Right

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Returns the argument without changing its kind or shape.

### Dyadic

Status: **supported**

Domain: Any two values inside the dense-real boundary.

Returns the right argument.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⥊` — Deshape / Reshape

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Returns a flat numeric list of the argument elements.

### Dyadic

Status: **supported**

Domain: A natural-number atom or list left argument and a dense right argument; a nonempty result requires a nonempty source.

Cycles the ravel of the right argument into the requested dense shape.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `∾` — Join / Join To

### Monadic

Status: **unsupported**

Domain: General Join consumes nested cells, which are outside the current value boundary.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Dense real arguments with matching trailing cell shapes and ranks differing by no more than one.

Joins the arguments along their leading axis, unitifying an argument when required.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `≍` — Solo / Couple

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Adds a leading unit axis.

### Dyadic

Status: **supported**

Domain: Two dense-real values with the same atom/array kind and shape.

Stacks the arguments along a new leading axis.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⋈` — Enlist / Pair

### Monadic

Status: **supported**

Domain: Real numeric atoms only.

Returns a one-element numeric list; general nested Enlist is outside the value boundary.

### Dyadic

Status: **supported**

Domain: Two real numeric atoms only.

Returns the two atoms as a numeric list; general nested Pair is outside the value boundary.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `↑` — Prefixes / Take

### Monadic

Status: **unsupported**

Domain: Prefixes generally produces nested results.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Whole-number atom/list counts over existing axes where no fill expansion is required.

Takes leading or trailing cells along the specified axes.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `↓` — Suffixes / Drop

### Monadic

Status: **unsupported**

Domain: Suffixes generally produces nested results.

Delegated to cBQN by the CLI or rejected in strict mode.

### Dyadic

Status: **supported**

Domain: Whole-number atom/list counts over existing axes of a dense real value.

Drops leading or trailing cells along the specified axes.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⌽` — Reverse / Rotate

### Monadic

Status: **supported**

Domain: Dense real arrays with at least one axis.

Reverses the leading axis.

### Dyadic

Status: **supported**

Domain: Whole-number atom/list rotations naming no more axes than the dense right argument has.

Rotates the named leading axes.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⍉` — Transpose / Reorder Axes

### Monadic

Status: **supported**

Domain: Dense real numeric values.

Moves the leading axis to the end; atoms become rank-zero arrays.

### Dyadic

Status: **supported**

Domain: A natural-number atom/list that completes to a permutation of the right argument axes.

Reorders dense axes; diagonalizing repeated destinations is not yet supported.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `/` — Indices / Replicate

### Monadic

Status: **supported**

Domain: Natural-number atoms or lists.

Returns each index repeated by its corresponding count.

### Dyadic

Status: **supported**

Domain: Natural-number atom/list counts agreeing with the leading axis of a dense right argument.

Replicates major cells according to the counts.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⍋` — Grade Up / Bins Up

### Monadic

Status: **supported**

Domain: Real numeric lists.

Returns the ascending grade permutation.

### Dyadic

Status: **supported**

Domain: An ascending real numeric list left argument and dense real queries.

Returns the ascending insertion bin for each query.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⍒` — Grade Down / Bins Down

### Monadic

Status: **supported**

Domain: Real numeric lists.

Returns the descending grade permutation.

### Dyadic

Status: **supported**

Domain: A descending real numeric list left argument and dense real queries.

Returns the descending insertion bin for each query.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊏` — First Cell / Select

### Monadic

Status: **supported**

Domain: Nonempty dense real arrays with a leading axis.

Returns the first major cell as an array.

### Dyadic

Status: **supported**

Domain: An in-bounds natural-number atom or list selecting the leading axis of a dense array.

Selects major cells while preserving dense shape.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊑` — First / Pick

### Monadic

Status: **supported**

Domain: Nonempty dense real values.

Returns the first element as a numeric atom.

### Dyadic

Status: **supported**

Domain: An in-bounds natural-number atom or coordinate list and a dense real array.

Indexes successive axes; a full coordinate returns an atom.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊐` — Classify / Index Of

### Monadic

Status: **supported**

Domain: Real numeric lists.

Returns the index of each element's first occurrence.

### Dyadic

Status: **supported**

Domain: A real numeric list principal argument and dense real queries.

Returns each query's first index, or the principal length when absent.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⊒` — Occurrence Count / Progressive Index Of

### Monadic

Status: **supported**

Domain: Real numeric lists.

Counts earlier occurrences of each element.

### Dyadic

Status: **supported**

Domain: A real numeric list principal argument and dense real queries.

Finds successive occurrences for repeated queries, returning the principal length when exhausted.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `∊` — Mark Firsts / Member Of

### Monadic

Status: **supported**

Domain: Real numeric lists.

Marks each element whose occurrence count is zero.

### Dyadic

Status: **supported**

Domain: Dense real queries on the left and a real numeric list principal argument on the right.

Returns numeric membership booleans with the shape of the left argument.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `⍷` — Deduplicate / Find

### Monadic

Status: **supported**

Domain: Real numeric lists.

Keeps the first occurrence of each distinct value.

### Dyadic

Status: **supported**

Domain: Real numeric list pattern and searched list.

Marks every complete occurrence of the left pattern in the right list.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## `↕` — Range / Windows

### Monadic

Status: **supported**

Domain: Natural-number atoms only.

Returns the float64 list of natural numbers below the argument. List Range, whose result is nested, is outside the current value boundary.

### Dyadic

Status: **supported**

Domain: Positive natural-number atom/list window sizes naming no more axes than the dense right argument has, with at least one complete window.

Returns the uniform dense array of sliding windows.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- Comparison results are BQN numeric booleans represented as float64 zero or one on the backend.
- The CLI delegates unsupported programs to cBQN only when the result fits the current dense-real numeric boundary; `--fallback error` requires acceleration instead.

## Fold

| Function operand | Status | Domain |
|---|---|---|
| `+´` | supported | Nonempty real numeric lists; sum cases are differentially tested. |
| `×´` | supported | Nonempty real numeric lists. |
| `∧´` | supported | Nonempty real numeric lists. |
| `∨´` | supported | Nonempty real numeric lists. |
| `⌊´` | supported | Nonempty real numeric lists. |
| `⌈´` | supported | Nonempty real numeric lists. |

## Insert

| Function operand | Status | Domain |
|---|---|---|
| `+˝` | supported | Dense real arrays with at least one axis; reduction is over major cells. |
| `×˝` | supported | Dense real arrays with at least one axis; reduction is over major cells. |
| `∧˝` | supported | Dense real arrays with at least one axis; numeric logical And is reduced over major cells. |
| `∨˝` | supported | Dense real arrays with at least one axis; numeric logical Or is reduced over major cells. |
| `⌊˝` | supported | Dense real arrays with at least one axis; minimum is reduced over major cells. |
| `⌈˝` | supported | Dense real arrays with at least one axis; maximum is reduced over major cells. |

## Scan

| Function operand | Status | Domain |
|---|---|---|
| `+`` | supported | Dense real arrays with at least one axis; inclusive scan is over major cells. |
| `×`` | supported | Dense real arrays with at least one axis; inclusive scan is over major cells. |
| `∧`` | supported | Dense real arrays with at least one axis; inclusive numeric logical-And scan is over major cells. |
| `∨`` | supported | Dense real arrays with at least one axis; inclusive numeric logical-Or scan is over major cells. |
| `⌊`` | supported | Dense real arrays with at least one axis; inclusive minimum scan is over major cells. |
| `⌈`` | supported | Dense real arrays with at least one axis; inclusive maximum scan is over major cells. |

## Pure combinators

| Modifier | Status | Domain | Behavior |
|---|---|---|---|
| `˜` Self / Swap | supported | Supported primitive or reduction operands whose monadic Self or dyadic Swap expansion remains within the dense-real backend domain. | A monadic call duplicates its right argument; a dyadic call exchanges the left and right arguments. |
| `∘` Atop | supported | Supported primitive or reduction operands, including unparenthesized combinator chains, whose expansion remains within the dense-real backend domain. | The right operand receives the original argument or arguments and its result is passed monadically to the left operand. |
| `○` Over | supported | Supported primitive operands whose expansion remains within the dense-real backend domain. | The right operand is applied monadically to each argument before the left operand is applied. |
| `⊸` Before / Bind Left | supported | Supported primitive operands or a numeric literal left operand, with a dense-real result. | The left operand transforms the left argument (or the duplicated right argument monadically) before the right operand receives it and the original right argument. |
| `⟜` After / Bind Right | supported | Supported primitive operands or a numeric literal right operand, with a dense-real result. | The right operand transforms the right argument before the left operand receives the original left argument and transformed right argument. |

## Function trains

| Form | Status | Domain | Behavior |
|---|---|---|---|
| `2-train` Composition train | supported | Parenthesized supported function components whose expansion remains within the dense-real backend domain. | Calls the right component at the original valence and passes its result monadically to the left component. |
| `3-train` Fork train | supported | Parenthesized supported function components, with numeric subjects allowed in argument positions, whose expansion remains within the dense-real backend domain. | Calls the outer components at the original valence and passes their results dyadically to the middle component. |
| `long train` Right-associated train | supported | Four or more supported components following BQN train role rules, including reduction and combinator-derived functions. | Associates from the right into two- and three-trains according to BQN train semantics, then specializes the expanded tensor expression. |
| `nested train` Nested train | supported | Parenthesized trains used as components of another parenthesized train, with a final dense-real result. | Retains each nested train in semantic IR and recursively inlines it before backend execution planning. |

## Bounded iteration modifiers

| Modifier | Status | Domain | Behavior |
|---|---|---|---|
| `⍟` Repeat | supported | A supported function operand and one literal natural-number count from zero through 64 whose unrolled expression contains at most 4,096 semantic IR nodes, with monadic or dyadic dense-real arguments. | Zero returns the right argument. Positive counts apply the operand that many times; a dyadic call reuses the original left argument each time. The compiler unrolls the bounded repetition before execution planning. |

## Dense mapping modifiers

| Modifier | Status | Domain | Behavior |
|---|---|---|---|
| `˘` Cells | supported | Dense real arguments with mapped frames that have no zero-length axes, and supported operands that return one uniform dense shape. Pervasive numeric operands also support empty arrays. | Applies the operand to major cells, using leading-axis frame agreement for two arguments, and combines uniform results as major cells. |
| `⎉` Rank | supported | Literal numeric rank atoms or one-to-three-item strands, dense real arguments with mapped frames that have no zero-length axes, and supported operands that return one uniform dense shape. Pervasive numeric operands also support empty arrays. | Natural ranks select trailing cells, negative ranks select a frame-axis count, ranks clamp to the argument rank, positive infinity selects the entire argument, negative infinity selects atoms, and dyadic frames use leading-axis agreement. |
| `¨` Each | supported | Dense real atoms and arrays with supported operands that return one uniform dense shape; general non-pervasive operands require mapped frames with no zero-length axes. Pervasive numeric operands support empty arrays. | Applies the operand to elements with leading-axis agreement for two arguments and always returns an array, including a rank-0 array for atom input. |
| `⌜` Table | supported | Dense real atoms and arrays with supported operands that return one uniform dense shape; general non-pervasive operands require argument frames with no zero-length axes. | Monadic Table maps over elements; dyadic Table applies the operand to every element pair and concatenates the two argument shapes as the result frame. |

## BQN source frontend

Both `.bqn` files and BQN strings compile to the same backend-neutral expression IR, then execute on the selected device backend.

| Construct | Status | Constraint |
|---|---|---|
| Function blocks `{...}` | supported | Headerless blocks using `𝕨` and/or `𝕩`, or immediate numeric blocks. |
| Bare program bodies | supported | Accepted as a runner convenience with the same expression subset. |
| Right-to-left function application | supported | Supported primitives, numeric constants, arguments, names, and parentheses. |
| Local assignment `←` | supported | Subject-valued local names assigned in statement order. |
| Separators and comments | supported | Newline, `⋄`, comma, and `#` line comments. |
| Numeric strands | supported | Literal numeric strands separated by `‿`, used for shapes, coordinates, counts, and axes. |
| Fold `´`, Insert `˝`, and Scan `` ` `` | supported | Prefix use with the supported dyadic function operands and no dyadic initial value. |
| Self/Swap `˜`, Atop `∘`, Over `○`, Before `⊸`, and After `⟜` | supported | Primitive and reduction operands plus numeric literal Bind operands in unparenthesized derived-function chains. Combinators remain explicit in semantic IR and are inlined only after specialization. |
| Cells `˘`, Rank `⎉`, Each `¨`, and Table `⌜` | supported | Dense uniform-result mapping. Rank accepts literal numeric atoms or strands; general computed rank operands and empty generic frames are not compiled yet. |
| Parenthesized function trains | supported | Two-, three-, long-, and nested trains over supported primitive or derived functions, including numeric subjects in train argument positions. The train must be applied within the accepted program body. |
| Statically bounded Repeat `⍟` | supported | A literal natural count from zero through 64, a supported operand, and an unrolled expression of at most 4,096 semantic IR nodes. Dynamic, negative, infinite, and array-valued repetition counts are not compiled yet. |
| General BQN syntax | unsupported | General array notation, standalone function-valued programs, headers, computed or named function values, remaining modifiers, namespaces, strings, nested values, and control flow are not compiled yet. The CLI can delegate numeric-boundary programs to cBQN. |

Source frontend tests:

- `tests/test_source_frontend.py`
- `tests/test_dense_primitives.py`
- `tests/test_corpus.py`

## Meaning of status

- **supported**: the stated domain is covered by automated differential tests against the pinned cBQN revision.
- **fallback**: the backend deliberately delegates to a semantically correct non-GPU implementation.
- **unsupported**: the backend rejects the operation or domain explicitly.
