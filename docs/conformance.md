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
| `⋆` Exponential / Power | supported | unsupported | `tests/test_corpus.py` |
| `√` Square Root / Root | supported | unsupported | `tests/test_corpus.py` |
| `⌊` Floor / Minimum | supported | supported | `tests/test_corpus.py` |
| `⌈` Ceiling / Maximum | supported | supported | `tests/test_corpus.py` |
| `|` Absolute Value / Modulus | supported | supported | `tests/test_corpus.py` |

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

## `⋆` — Exponential / Power

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense arrays.

Element-wise natural exponential.

### Dyadic

Status: **unsupported**

Domain: No dyadic domain is claimed yet.

Rejected explicitly.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

## `√` — Square Root / Root

### Monadic

Status: **supported**

Domain: Positive real numeric atoms and dense arrays in the tested corpus.

Element-wise square root.

### Dyadic

Status: **unsupported**

Domain: No dyadic domain is claimed yet.

Rejected explicitly.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with BQN portability guidance.
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
- There is not yet a cBQN fallback for source outside the supported subset. It is rejected explicitly.

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
| General BQN syntax | unsupported | Arrays in source, trains, headers, modifiers other than Fold, namespaces, strings, nested values, and control flow are not compiled yet. |

Source frontend tests:

- `tests/test_source_frontend.py`
- `tests/test_corpus.py`

## Meaning of status

- **supported**: the stated domain is covered by automated differential tests against the pinned cBQN revision.
- **fallback**: the backend deliberately delegates to a semantically correct non-GPU implementation.
- **unsupported**: the backend rejects the operation or domain explicitly.
