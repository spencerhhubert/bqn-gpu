# Program corpus

`corpus/programs.json` is a generated, tracked corpus of actual BQN sources. The initial floor is 100 programs, not a target or limit. New semantic bugs, optimization boundaries, real workloads, useful phrases, and backend discrepancies should add durable cases with stable IDs.

The current 200-case seed has five layers:

- individual glyphs over monadic and dyadic valence, atom extension, equal shapes, and leading-axis agreement;
- short elementwise phrases;
- naive and idiomatic paired formulations of the same computation;
- monadic and dyadic reductions; and
- long elementwise pipelines ending in reductions.

Each entry records its BQN source, arity, deterministic input mode and domains, comparison tolerance, category, variant, and tags. Input size is a runner parameter rather than part of the source, so the same program can be exercised from tiny correctness shapes through benchmark-scale arrays.

The correctness test compiles every entry through the public BQN source frontend, executes it through the selected backend, evaluates the original source with pinned cBQN, and compares atom/array identity, complete shape, and all data. The source—not a Python reimplementation—is the corpus's semantic identity.

`scripts/run_corpus.py` selects by stable ID glob and tags, scales deterministic inputs, dispatches the compiled source to cBQN, tinygrad, and/or PyTorch, checks correctness, and records cold and repeated warm timings as JSON. Tensor-backend input transfer occurs before timing and output transfer after it. tinygrad's execution mode is a reusable JIT-captured graph after two warmups; PyTorch currently uses eager dispatch. cBQN timings include its embedding-boundary value conversion and CPU execution. Each result records its execution mode, so consumers should retain those labels rather than treating all timing scopes as identical.

Run `python scripts/generate_corpus.py --check` to verify the tracked manifest or omit `--check` to regenerate it. The generator enforces unique stable IDs and the initial floor; CI checks that the tracked file is current.
