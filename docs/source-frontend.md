# BQN source frontend

`bqn-gpu run FILE.bqn` and `bqn-gpu eval SOURCE` use the same parser, compiler, and execution path. `compile_bqn(source)` lowers supported BQN source to a small backend-neutral expression tree; `CompiledProgram.execute` supplies values and dispatches that tree to a selected backend.

The accepted subset is deliberately narrower than BQN. It includes:

- headerless `{...}` function blocks using `𝕨` and `𝕩`;
- immediate blocks and bare numeric program bodies;
- numeric constants, including BQN `¯`, decimal exponents, `π`, and `∞` spellings;
- parentheses and right-to-left calls of the primitives in the conformance table;
- numeric strands for shapes, axes, counts, and other literal vectors;
- Fold `´`, Insert `˝`, and Scan `` ` `` for the functions in their conformance tables;
- Self/Swap `˜`, Atop `∘`, Over `○`, Before `⊸`, and After `⟜` over supported primitive or reduction operands, including numeric literal Bind operands;
- dense uniform-result Cells `˘`, Rank `⎉`, Each `¨`, and Table `⌜`, with
  literal numeric Rank specifications;
- parenthesized two-, three-, long-, and nested function trains over supported
  primitive or derived functions, including numeric subjects in argument slots;
- local subject assignment with `←`;
- newline, `⋄`, and comma statement separators; and
- `#` line comments.

The frontend does not yet compile general array notation, standalone
function-valued programs, block headers, computed or named function values, the
remaining modifiers, strings or characters, namespaces, system values,
recursion, mutation, or control flow.
An unsupported token or construct has a line-and-column diagnostic. By default
the CLI reports that diagnostic on stderr and delegates to cBQN when its shared
library is available and the result fits the current dense-real numeric
boundary. `--fallback error` rejects instead. There is no silent
reinterpretation.

## JSON values

Command-line arguments use JSON:

| JSON | BQN boundary value |
|---|---|
| `3.5` | numeric atom |
| `[1,2,3]` | numeric list with shape `⟨3⟩` |
| `[[1,2],[3,4]]` | dense numeric array with shape `⟨2,2⟩` |
| `{"shape":[],"data":[3]}` | rank-0 numeric array, distinct from an atom |
| `{"shape":[0,3],"data":[]}` | empty rank-2 numeric array |

Nested JSON input must be rectangular. The explicit form is authoritative when shape cannot be inferred. Only real numeric data is currently accepted.
