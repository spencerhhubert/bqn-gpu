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
| `=` Rank / Equals | unsupported | supported | `tests/test_corpus.py` |
| `≠` Length / Not Equals | unsupported | supported | `tests/test_corpus.py` |
| `<` Enclose / Less Than | unsupported | supported | `tests/test_corpus.py` |
| `>` Merge / Greater Than | unsupported | supported | `tests/test_corpus.py` |
| `≤` Mark Firsts / Less Than or Equal | unsupported | supported | `tests/test_corpus.py` |
| `≥` Occurrence Count / Greater Than or Equal | unsupported | supported | `tests/test_corpus.py` |

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

Status: **unsupported**

Domain: No monadic Rank domain is claimed yet.

Delegated to cBQN by the CLI or rejected in strict mode.

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

Status: **unsupported**

Domain: No monadic Length domain is claimed yet.

Delegated to cBQN by the CLI or rejected in strict mode.

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

## `≤` — Mark Firsts / Less Than or Equal

### Monadic

Status: **unsupported**

Domain: No monadic Mark Firsts domain is claimed yet.

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

## `≥` — Occurrence Count / Greater Than or Equal

### Monadic

Status: **unsupported**

Domain: No monadic Occurrence Count domain is claimed yet.

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

## Fold

| Function operand | Status | Domain |
|---|---|---|
| `+´` | supported | Nonempty real numeric lists; sum cases are differentially tested. |
| `×´` | supported | Nonempty real numeric lists. |
| `⌊´` | supported | Nonempty real numeric lists. |
| `⌈´` | supported | Nonempty real numeric lists. |

## BQN source frontend

Both `.bqn` files and BQN strings compile to the same backend-neutral expression IR, then execute on the selected device backend.

| Construct | Status | Constraint |
|---|---|---|
| Function blocks `{...}` | supported | Headerless blocks using `𝕨` and/or `𝕩`, or immediate numeric blocks. |
| Bare program bodies | supported | Accepted as a runner convenience with the same expression subset. |
| Right-to-left function application | supported | Supported primitives, numeric constants, arguments, names, and parentheses. |
| Local assignment `←` | supported | Subject-valued local names assigned in statement order. |
| Separators and comments | supported | Newline, `⋄`, comma, and `#` line comments. |
| General BQN syntax | unsupported | Arrays in source, trains, headers, modifiers other than Fold, namespaces, strings, nested values, and control flow are not compiled yet. The CLI can delegate numeric-boundary programs to cBQN. |

Source frontend tests:

- `tests/test_source_frontend.py`
- `tests/test_corpus.py`

## Meaning of status

- **supported**: the stated domain is covered by automated differential tests against the pinned cBQN revision.
- **fallback**: the backend deliberately delegates to a semantically correct non-GPU implementation.
- **unsupported**: the backend rejects the operation or domain explicitly.
