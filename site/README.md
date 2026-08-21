# bqn-gpu website

This directory contains the public benchmark and capability history website. It is a TypeScript Cloudflare Worker with a D1 database and a static Tailwind frontend.

## Data contract

`POST /api/v1/ingest` accepts a schema-versioned experiment bundle containing:

- the exact repository commit and ref;
- a fingerprinted hardware/software environment, including CPU topology, accelerators, drivers, runtimes, cBQN, and framework revisions;
- the suite, command, timing scope, dtype, seed, input profile, warmup/repetition counts, and artifact URL;
- stable BQN program IDs, sources, hashes, input generators, comparison policies, and tags;
- correctness outcomes, explicit timing scope, raw nanosecond timings, aggregates, and output hashes for each backend; and
- an optional capability snapshot with its full manifest and feature-level evidence.

Runs are immutable. Retrying an identical run ID is idempotent; attempting to reuse it with different content fails. A stable program ID cannot be reused for different source. Per-result timing scopes prevent resident compute and host-boundary measurements from being treated as comparable. Result identity includes input size, so aligned profiles can record several scales in one reproducible run. Extra metadata objects are retained so new hardware and measurement fields can be added without discarding old records.

Public, read-only endpoints are listed at `/api/v1/schema`. `/api/v1/runs/{run_id}` returns a reproduction bundle with the commit, environment, invocation, exact BQN sources and input policies, raw results, and capability evidence for a recorded experiment. Ingestion requires the `BENCHMARK_INGEST_TOKEN` Worker secret.

## Local development

```sh
cd site
npm ci
npm run migrate:local
npm run dev
npm run check
```

Put a development-only `BENCHMARK_INGEST_TOKEN` in `site/.dev.vars`. The file is ignored. Never put secrets in `wrangler.jsonc` or tracked source.

## Deployment

Pushes and pull requests verify the site. Tags matching `site-v*` apply pending D1 migrations and deploy the Worker, static assets, and required ingestion secret with GitHub Actions. For example:

```sh
git tag -a site-v0.1.0 -m "Deploy website v0.1.0"
git push origin site-v0.1.0
```

The GitHub environment contains an account-scoped Cloudflare token limited to Workers Scripts Write, D1 Write, and Account Settings Read. The benchmark ingestion token is separate and has no Cloudflare API permissions.

Raw benchmark reports and generated ingestion payloads can be attached to a GPU-validation GitHub Release. The manual `Publish benchmark results` workflow downloads matching payload assets and sends them to the API using the repository ingestion secret. Reusing an already-published run ID is safe when the payload is identical.
