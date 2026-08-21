# bqn-gpu agent instructions

## Project intent

This repository is private during early development but is intended to become public. Treat every tracked file, commit message, issue, test artifact, and piece of documentation as public material.

Do not commit credentials, tokens, private URLs, machine-specific paths, internal infrastructure details, personal data, or anything else specific to a contributor's environment. Put local-only instructions in `AGENTS.local.md` and longer-lived local context in `agents-notes-local/`; both are intentionally ignored by Git.

We are building a GPU/CUDA backend for the BQN programming language, specifically in relation to cBQN. The backend should be:

- semantically correct according to cBQN;
- extremely minimal in architecture and dependency surface;
- fast in the circumstances it claims to accelerate;
- thoroughly tested, including edge cases and randomized differential tests; and
- explicit about which parts of BQN it supports.

## Correctness and conformance

cBQN is the semantic reference. Accelerated execution must produce the same observable result as cBQN. If an operation, type, shape, rank, or edge case is not supported on the GPU, it must take a correct fallback path rather than silently changing semantics.

CI must automatically detect semantic differences between the GPU backend and cBQN. Prefer small deterministic tests, differential tests against the CPU/reference implementation, and reproducible randomized tests with recorded seeds. Test both accelerated paths and their boundaries or fallbacks.

Maintain precise user-facing documentation of the supported language surface. The conformance documentation should identify, at minimum:

- the BQN primitive or construct;
- supported argument types, ranks, shapes, and other constraints;
- whether execution is accelerated, falls back, or is unsupported;
- known semantic limitations; and
- the tests that establish the claim.

Do not describe a feature as supported until its behavior is covered by automated correctness tests.

## Engineering principles

- Start with the smallest end-to-end implementation that can be tested against cBQN.
- Keep the CUDA kernel surface small. Add custom kernels only with a correctness test and a demonstrated reason they are needed.
- Separate semantic decisions from device-specific execution details.
- Prefer obvious ownership, synchronization, and error handling over clever abstractions.
- Make failures deterministic and actionable whenever possible.
- Benchmark optimized paths, but never trade away language correctness for speed.
- Keep changes narrow, reviewable, documented, and covered by tests.

## Local instructions

Read `AGENTS.local.md` and `agents-notes-local/` when they exist. They may contain machine-specific instructions, research, decisions, or project context and must remain untracked.
