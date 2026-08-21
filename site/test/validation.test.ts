import { describe, expect, it } from "vitest";
import { validatePayload, ValidationError } from "../src/validation";

const SHA = "a".repeat(64);
const COMMIT = "b".repeat(40);

function payload() {
  return {
    schema_version: 1,
    commit: {
      sha: COMMIT,
      repository: "https://github.com/spencerhhubert/bqn-gpu",
      ref: "refs/tags/gpu-validation-test",
    },
    environment: {
      fingerprint: "c".repeat(64),
      captured_at: "2026-08-21T12:00:00Z",
      architecture: "x86_64",
      cpu: { model: "Example CPU", cores: 8, threads: 16 },
      accelerators: [{ kind: "gpu", vendor: "NVIDIA", model: "A10", count: 1 }],
      software: { python: "3.12.0", cbqn: COMMIT },
      metadata: { future_field: { remains: true } },
    },
    run: {
      id: "gpu-validation-test.1",
      suite: "paired-1m",
      started_at: "2026-08-21T12:00:00Z",
      finished_at: "2026-08-21T12:01:00Z",
      status: "pass",
      device: "CUDA",
      dtype: "float64",
      timing_scope: "resident-compute",
      input_profile: { size: 1_048_576, generator: "deterministic-v1" },
      warmups: 2,
      repetitions: 10,
    },
    programs: [{
      id: "pair.pipeline.01.naive",
      source: "{𝕩+1}",
      source_sha256: SHA,
      category: "paired",
      variant: "naive",
      tags: ["paired"],
      input_generator: { name: "positive" },
      comparison_policy: { relative_tolerance: 1e-12 },
    }],
    results: [{
      program_id: "pair.pipeline.01.naive",
      backend: "tinygrad",
      backend_version: "test",
      execution_mode: "jit-captured",
      correct: true,
      input_size: 1_048_576,
      cold_ns: 2_000_000,
      median_ns: 250_000,
      min_ns: 240_000,
      max_ns: 280_000,
      p95_ns: 275_000,
      timings_ns: [240_000, 250_000, 280_000],
    }],
    capability: {
      backend: "tinygrad",
      manifest_sha256: "d".repeat(64),
      corpus_programs: 200,
      glyphs_total: 17,
      monadic_supported: 13,
      dyadic_supported: 15,
      folds_supported: 4,
      tests_passed: 247,
      tests_failed: 0,
      tests_skipped: 1,
      manifest: { schema_version: 1 },
      features: [{
        id: "add.dyadic",
        glyph: "+",
        name: "Add",
        valence: "dyadic",
        status: "accelerated",
        evidence: ["tests/test_add_conformance.py"],
      }],
    },
  };
}

describe("ingestion payload validation", () => {
  it("accepts a complete, extensible run record", () => {
    const result = validatePayload(payload());
    expect(result.run.id).toBe("gpu-validation-test.1");
    expect(result.environment.metadata).toEqual({ future_field: { remains: true } });
    expect(result.capability?.features?.[0].status).toBe("accelerated");
  });

  it("rejects an unknown schema version", () => {
    const value = payload();
    value.schema_version = 2;
    expect(() => validatePayload(value)).toThrowError(ValidationError);
  });

  it("rejects results whose program is absent", () => {
    const value = payload();
    value.results[0].program_id = "missing.program";
    expect(() => validatePayload(value)).toThrow("result references missing program");
  });

  it("rejects reuse of a program ID inside one payload", () => {
    const value = payload();
    value.programs.push({ ...value.programs[0] });
    expect(() => validatePayload(value)).toThrow("program IDs must be unique");
  });

  it("requires full hashes and non-negative timings", () => {
    const badHash = payload();
    badHash.programs[0].source_sha256 = "abc";
    expect(() => validatePayload(badHash)).toThrow("lowercase SHA-256");

    const badTiming = payload();
    badTiming.results[0].median_ns = -1;
    expect(() => validatePayload(badTiming)).toThrow("must be a safe integer >= 0");
  });
});
