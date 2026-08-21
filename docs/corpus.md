# Program corpus

`corpus/programs.json` is a generated, tracked corpus of actual BQN sources. The initial floor is 100 programs, not a target or limit. New semantic bugs, optimization boundaries, real workloads, useful phrases, and backend discrepancies should add durable cases with stable IDs.

The current 200-case seed has five layers:

- individual glyphs over monadic and dyadic valence, atom extension, equal shapes, and leading-axis agreement;
- short elementwise phrases;
- naive and idiomatic paired formulations of the same computation;
- monadic and dyadic reductions; and
- long elementwise pipelines ending in reductions.

Each entry records its BQN source, independent direct tinygrad and direct PyTorch sources, a backend-neutral workload expression, arity, deterministic input mode and domains, comparison tolerance, category, variant, and tags. Input size is a runner parameter rather than part of the source, so the same program can be exercised from tiny correctness shapes through benchmark-scale arrays.

The correctness suite evaluates the original source with pinned cBQN, compiles it through the public bqn-gpu source frontend, and independently executes the direct tinygrad and direct PyTorch programs. It compares atom/array identity, complete shape, and all data. The BQN source remains the semantic identity; native sources are independent performance/correctness baselines and never define BQN behavior.

`scripts/run_corpus.py` selects by stable ID glob and tags, scales deterministic inputs, checks every requested implementation against cBQN, and records cold and repeated warm timings as JSON. Its unambiguous implementation names are `cbqn`, `bqn-gpu-tinygrad`, `bqn-gpu-torch`, `native-tinygrad`, and `native-torch`. The default comparison is cBQN CPU, bqn-gpu on tinygrad, direct tinygrad, and direct PyTorch. Tensor input transfer occurs before timing and output transfer after it. bqn-gpu/tinygrad and native tinygrad use reusable JIT-captured graphs; native PyTorch is eager. Each result records its language, implementation kind, framework, device, execution mode, and timing scope.

Run `python scripts/generate_corpus.py --check` to verify the tracked manifest or omit `--check` to regenerate it. The generator enforces unique stable IDs and the initial floor; CI checks that the tracked file is current.
