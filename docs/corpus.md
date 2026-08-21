# Program corpus

`corpus/programs.json` is a generated, tracked corpus of actual BQN sources. The initial floor is 100 programs, not a target or limit. New semantic bugs, optimization boundaries, real workloads, useful phrases, and backend discrepancies should add durable cases with stable IDs.

The current 276-case seed has ten layers:

- individual glyphs over monadic and dyadic valence, atom extension, equal shapes, and leading-axis agreement;
- short elementwise phrases;
- naive and idiomatic paired formulations of the same computation;
- monadic and dyadic reductions; and
- long elementwise pipelines ending in reductions;
- dense structural primitives such as reverse, transpose, take/drop, windows, selection, and reshaping;
- ordering, grading, match, logic, and identity functions;
- Fold, Insert, and Scan families;
- structural and modifier phrases; and
- naive/idiomatic structural pairs whose results agree but whose compiler costs currently differ.

Each entry records its BQN source, independent direct tinygrad and direct PyTorch sources, a backend-neutral workload expression, arity, deterministic input mode and domains, comparison tolerance, category, variant, and tags. Input size is a runner parameter rather than part of the source, so the same program can be exercised from tiny correctness shapes through benchmark-scale arrays.

The correctness suite evaluates the original source with pinned cBQN, compiles it through the public bqn-gpu source frontend, and independently executes the direct tinygrad and direct PyTorch programs. It compares atom/array identity, complete shape, and all data. The BQN source remains the semantic identity; native sources are independent performance/correctness baselines and never define BQN behavior.

`scripts/run_corpus.py` selects by stable ID glob and tags, scales deterministic inputs, checks every requested implementation against cBQN, and records cold and repeated warm timings as JSON. Its unambiguous implementation names are `cbqn`, `bqn-gpu-tinygrad`, `bqn-gpu-torch`, `native-tinygrad`, and `native-torch`. The default comparison is cBQN CPU, bqn-gpu on tinygrad, direct tinygrad, and direct PyTorch. Tensor input transfer occurs before timing and output transfer after it. bqn-gpu/tinygrad and native tinygrad use reusable JIT-captured graphs; native PyTorch is eager. Each result records its language, implementation kind, framework, device, execution mode, and timing scope.

`corpus/benchmark-profiles.json` aligns frequent and occasional measurements. The `development` profile is a fixed program-and-size subset of `certification`, so both measurements describe the same program IDs, deterministic seeds, input recipes, timing boundary, and backend identities at a commit. Development runs use fewer warmups and repetitions; certification adds the rest of the corpus and larger scale. Explicit `--size`, `--warmup`, or `--repeat` arguments override a profile for diagnosis while remaining recorded in the report.

Curated construction is complemented by deterministic typed-grammar generation and equivalence mutation. Discovery candidates are checked against cBQN before they are written, but remain local diagnostic artifacts until a correctness failure, new compiler path, scaling gap, or missed simplification earns a stable corpus entry. See [generative-testing.md](generative-testing.md).

```sh
python scripts/run_corpus.py --profile development --device CUDA \
  --backend cbqn --backend bqn-gpu-tinygrad \
  --backend native-tinygrad --backend native-torch \
  --output development-results.json
```

Run `python scripts/generate_corpus.py --check` to verify the tracked manifest or omit `--check` to regenerate it. The generator enforces unique stable IDs and the initial floor; CI checks that the tracked file is current.
