# Generative program discovery

The tracked corpus is curated: every durable program has a stable identity, independent BQN/tinygrad/Torch definitions, deterministic inputs, and a reason to remain. Curated examples alone tend to revisit operations that already fit the compiler well, so the project also generates candidates from a typed grammar.

`scripts/discover_programs.py` has five deterministic strategies:

- `grammar` builds valid dense-list programs from semantic operations while tracking their result kind and rank. It does not sample BQN text, so malformed syntax is not treated as language coverage.
- `mutation` starts with a generated typed expression and adds a known equivalence such as double Reverse or cancelling Rotates. The unmodified BQN is retained beside the candidate, making redundant work and missing simplifications directly measurable.
- `combinator` wraps a generated dense expression in Self, Atop, Over,
  Before/Bind, After/Bind, or an Atop chain. It records the fully expanded BQN
  beside the compact form, so both language correctness and missed inlining or
  fusion opportunities are directly testable.
- `train` wraps a generated expression in two-, three-, long-, nested-,
  subject-, or derived-function trains. It retains an explicit equivalent
  block, testing parser roles, right association, semantic inlining, and fusion
  without trusting source-string generation for correctness.
- `repeat` chooses a natural count and a pervasive, structural, bound, or train
  operand, then retains its fully unrolled equivalent. Zero counts and repeated
  multi-stage expressions exercise both semantic identity and graph-growth
  boundaries.

Every candidate records its generator seed, construction depth, semantic features, BQN source, IR, and equivalent source when one exists. The discovery command executes the candidate through bqn-gpu and pinned cBQN before saving it:

```sh
python scripts/discover_programs.py --seed 20260821 --count 100 \
  --min-steps 8 --max-steps 32
```

Generated output under `.build/` is diagnostic rather than benchmark history. A candidate becomes durable only when it adds a semantic interaction, compiler path, scaling behavior, correctness failure, or optimization opportunity. Promotion means adding a stable case to `scripts/generate_corpus.py`, generating independent native sources, and then letting the ordinary corpus and benchmark profiles test it. This keeps random volume from diluting the aggregate dashboard while preserving every useful discovery.

The intended search score combines novelty and consequence: new glyph/shape interactions, a previously unseen IR or materialization path, a cBQN disagreement, a large native-framework gap, poor scaling, or an equivalence mutation that the optimizer fails to erase. Future feedback-guided generation can select parents from these signals; the deterministic grammar remains the source of syntax and type correctness.
