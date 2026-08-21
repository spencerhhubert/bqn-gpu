# Conformance

This file is generated from `conformance.json`. Do not edit it by hand.

Semantic oracle: **cBQN**, pinned by `deps/cbqn.rev`.

Execution backend: **tinygrad** at `2d48fe8b7bd9acfa00e91a7f89b28b3ded370c27`, using `float64` on `CPU`, `CUDA`.

| Primitive | Monad | Dyad | Tests |
|---|---|---|---|
| `+` Conjugate / Add | supported | supported | `tests/test_add_conformance.py`<br>`tests/test_tinygrad_backend.py` |

## `+` — Conjugate / Add

### Monadic

Status: **supported**

Domain: Real numeric atoms and dense real numeric arrays of any rank and shape.

Identity, matching current BQN Conjugate behavior for real numbers.

### Dyadic

Status: **supported**

Domain: Pairs of real numeric atoms or dense real numeric arrays whose shapes satisfy BQN leading-axis agreement.

Element-wise float64 addition with atom extension and leading-axis agreement.

### Limitations

- Nested arrays and characters are not supported.
- Values are represented as float64; preservation of cBQN's packed integer storage is not claimed.
- Signed-zero preservation is not claimed, consistent with the BQN documentation's portability guidance.
- This primitive API is not yet a parser for arbitrary BQN source and is not transparently installed into cBQN.

## Meaning of status

- **supported**: the stated domain is covered by automated differential tests against the pinned cBQN revision.
- **fallback**: the backend deliberately delegates to a semantically correct non-GPU implementation.
- **unsupported**: the backend rejects the operation or domain explicitly.
