import type {
  CapabilityRecord,
  CommitRecord,
  EnvironmentRecord,
  IngestPayload,
  JsonObject,
  ProgramRecord,
  ResultRecord,
  RunRecord,
} from "./model";

export class ValidationError extends Error {}

const SHA256 = /^[0-9a-f]{64}$/;
const GIT_SHA = /^[0-9a-f]{40,64}$/;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,199}$/;

export function validatePayload(value: unknown): IngestPayload {
  const root = object(value, "payload");
  exact(root.schema_version, 1, "schema_version");
  const payload: IngestPayload = {
    schema_version: 1,
    commit: validateCommit(root.commit),
    environment: validateEnvironment(root.environment),
    run: validateRun(root.run),
    programs: array(root.programs, "programs").map((item, index) =>
      validateProgram(item, `programs[${index}]`),
    ),
    results: array(root.results, "results").map((item, index) =>
      validateResult(item, `results[${index}]`),
    ),
  };
  if (root.capability !== undefined) payload.capability = validateCapability(root.capability);
  if (payload.programs.length === 0) fail("programs must not be empty");
  if (payload.results.length === 0) fail("results must not be empty");
  if (new Set(payload.programs.map((program) => program.id)).size !== payload.programs.length) {
    fail("program IDs must be unique");
  }
  const programIds = new Set(payload.programs.map((program) => program.id));
  for (const result of payload.results) {
    if (!programIds.has(result.program_id)) {
      fail(`result references missing program ${result.program_id}`);
    }
  }
  return payload;
}

function validateCommit(value: unknown): CommitRecord {
  const item = object(value, "commit");
  const sha = text(item.sha, "commit.sha", 64);
  if (!GIT_SHA.test(sha)) fail("commit.sha must be a full lowercase Git SHA");
  return {
    sha,
    repository: text(item.repository, "commit.repository", 300),
    ...optionalTextFields(item, ["ref", "committed_at", "url"]),
    metadata: optionalObject(item.metadata, "commit.metadata"),
  };
}

function validateEnvironment(value: unknown): EnvironmentRecord {
  const item = object(value, "environment");
  const fingerprint = text(item.fingerprint, "environment.fingerprint", 64);
  if (!SHA256.test(fingerprint)) fail("environment.fingerprint must be a lowercase SHA-256");
  const accelerators = array(item.accelerators, "environment.accelerators").map((value, index) => {
    const accelerator = object(value, `environment.accelerators[${index}]`);
    return {
      kind: text(accelerator.kind, `environment.accelerators[${index}].kind`, 40),
      model: text(accelerator.model, `environment.accelerators[${index}].model`, 200),
      ...optionalTextFields(accelerator, ["vendor", "compute_capability", "driver", "runtime"]),
      count: optionalInteger(accelerator.count, `environment.accelerators[${index}].count`, 1),
      memory_bytes: optionalInteger(accelerator.memory_bytes, `environment.accelerators[${index}].memory_bytes`, 0),
      metadata: optionalObject(accelerator.metadata, `environment.accelerators[${index}].metadata`),
    };
  });
  const softwareValue = object(item.software, "environment.software");
  const software: Record<string, string> = {};
  for (const [key, value] of Object.entries(softwareValue)) software[key] = text(value, `environment.software.${key}`, 300);
  const cpuValue = item.cpu === undefined ? undefined : object(item.cpu, "environment.cpu");
  return compact({
    fingerprint,
    captured_at: text(item.captured_at, "environment.captured_at", 80),
    ...optionalTextFields(item, ["label", "architecture", "operating_system", "kernel"]),
    cpu: cpuValue
      ? compact({
          model: optionalText(cpuValue.model, "environment.cpu.model", 300),
          sockets: optionalInteger(cpuValue.sockets, "environment.cpu.sockets", 0),
          cores: optionalInteger(cpuValue.cores, "environment.cpu.cores", 0),
          threads: optionalInteger(cpuValue.threads, "environment.cpu.threads", 0),
        })
      : undefined,
    memory_bytes: optionalInteger(item.memory_bytes, "environment.memory_bytes", 0),
    accelerators,
    software,
    metadata: optionalObject(item.metadata, "environment.metadata"),
  }) as unknown as EnvironmentRecord;
}

function validateRun(value: unknown): RunRecord {
  const item = object(value, "run");
  const status = text(item.status, "run.status", 20);
  if (!(["pass", "fail", "partial"] as const).includes(status as RunRecord["status"])) fail("run.status is invalid");
  return compact({
    id: identifier(item.id, "run.id"),
    suite: identifier(item.suite, "run.suite"),
    started_at: text(item.started_at, "run.started_at", 80),
    finished_at: optionalText(item.finished_at, "run.finished_at", 80),
    status: status as RunRecord["status"],
    device: text(item.device, "run.device", 100),
    dtype: optionalText(item.dtype, "run.dtype", 100),
    timing_scope: identifier(item.timing_scope, "run.timing_scope"),
    input_profile: jsonValue(item.input_profile, "run.input_profile"),
    warmups: integer(item.warmups, "run.warmups", 0),
    repetitions: integer(item.repetitions, "run.repetitions", 1),
    seed: optionalInteger(item.seed, "run.seed"),
    command: optionalText(item.command, "run.command", 2000),
    artifact_url: optionalText(item.artifact_url, "run.artifact_url", 1000),
    runner_version: optionalText(item.runner_version, "run.runner_version", 200),
    metadata: optionalObject(item.metadata, "run.metadata"),
  }) as unknown as RunRecord;
}

function validateProgram(value: unknown, path: string): ProgramRecord {
  const item = object(value, path);
  const sourceSha = text(item.source_sha256, `${path}.source_sha256`, 64);
  if (!SHA256.test(sourceSha)) fail(`${path}.source_sha256 must be a lowercase SHA-256`);
  const tags = item.tags === undefined ? undefined : array(item.tags, `${path}.tags`).map((tag, index) => text(tag, `${path}.tags[${index}]`, 100));
  return compact({
    id: identifier(item.id, `${path}.id`),
    source: text(item.source, `${path}.source`, 100_000),
    source_sha256: sourceSha,
    category: identifier(item.category, `${path}.category`),
    variant: optionalText(item.variant, `${path}.variant`, 100),
    tags,
    input_generator: optionalJsonValue(item.input_generator, `${path}.input_generator`),
    comparison_policy: optionalJsonValue(item.comparison_policy, `${path}.comparison_policy`),
    metadata: optionalObject(item.metadata, `${path}.metadata`),
  }) as unknown as ProgramRecord;
}

function validateResult(value: unknown, path: string): ResultRecord {
  const item = object(value, path);
  const timings = item.timings_ns === undefined ? undefined : array(item.timings_ns, `${path}.timings_ns`).map((timing, index) => integer(timing, `${path}.timings_ns[${index}]`, 0));
  const outputSha = optionalText(item.output_sha256, `${path}.output_sha256`, 64);
  if (outputSha !== undefined && !SHA256.test(outputSha)) fail(`${path}.output_sha256 must be a lowercase SHA-256`);
  return compact({
    program_id: identifier(item.program_id, `${path}.program_id`),
    backend: identifier(item.backend, `${path}.backend`),
    backend_version: optionalText(item.backend_version, `${path}.backend_version`, 300),
    execution_mode: identifier(item.execution_mode, `${path}.execution_mode`),
    correct: boolean(item.correct, `${path}.correct`),
    skipped: optionalBoolean(item.skipped, `${path}.skipped`),
    error: optionalText(item.error, `${path}.error`, 4000),
    input_size: optionalInteger(item.input_size, `${path}.input_size`, 0),
    cold_ns: optionalInteger(item.cold_ns, `${path}.cold_ns`, 0),
    median_ns: optionalInteger(item.median_ns, `${path}.median_ns`, 0),
    min_ns: optionalInteger(item.min_ns, `${path}.min_ns`, 0),
    max_ns: optionalInteger(item.max_ns, `${path}.max_ns`, 0),
    p95_ns: optionalInteger(item.p95_ns, `${path}.p95_ns`, 0),
    timings_ns: timings,
    output_sha256: outputSha,
    metadata: optionalObject(item.metadata, `${path}.metadata`),
  }) as unknown as ResultRecord;
}

function validateCapability(value: unknown): CapabilityRecord {
  const item = object(value, "capability");
  const manifestSha = text(item.manifest_sha256, "capability.manifest_sha256", 64);
  if (!SHA256.test(manifestSha)) fail("capability.manifest_sha256 must be a lowercase SHA-256");
  const features = item.features === undefined ? undefined : array(item.features, "capability.features").map((value, index) => {
    const path = `capability.features[${index}]`;
    const feature = object(value, path);
    const status = text(feature.status, `${path}.status`, 30);
    if (!(["accelerated", "fallback", "unsupported", "supported"] as const).includes(status as never)) fail(`${path}.status is invalid`);
    return compact({
      id: identifier(feature.id, `${path}.id`),
      glyph: optionalText(feature.glyph, `${path}.glyph`, 30),
      name: text(feature.name, `${path}.name`, 200),
      valence: identifier(feature.valence, `${path}.valence`),
      status: status as CapabilityRecord["features"] extends Array<infer U> ? U extends { status: infer S } ? S : never : never,
      domain: optionalText(feature.domain, `${path}.domain`, 4000),
      behavior: optionalText(feature.behavior, `${path}.behavior`, 4000),
      evidence: optionalJsonValue(feature.evidence, `${path}.evidence`),
      metadata: optionalObject(feature.metadata, `${path}.metadata`),
    });
  });
  return compact({
    backend: identifier(item.backend, "capability.backend"),
    manifest_sha256: manifestSha,
    corpus_programs: integer(item.corpus_programs, "capability.corpus_programs", 0),
    glyphs_total: integer(item.glyphs_total, "capability.glyphs_total", 0),
    monadic_supported: integer(item.monadic_supported, "capability.monadic_supported", 0),
    dyadic_supported: integer(item.dyadic_supported, "capability.dyadic_supported", 0),
    folds_supported: integer(item.folds_supported, "capability.folds_supported", 0),
    tests_passed: integer(item.tests_passed, "capability.tests_passed", 0),
    tests_failed: integer(item.tests_failed, "capability.tests_failed", 0),
    tests_skipped: integer(item.tests_skipped, "capability.tests_skipped", 0),
    value_domain: optionalText(item.value_domain, "capability.value_domain", 2000),
    manifest: jsonValue(item.manifest, "capability.manifest"),
    features,
    metadata: optionalObject(item.metadata, "capability.metadata"),
  }) as unknown as CapabilityRecord;
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(`${path} must be an object`);
  return value as Record<string, unknown>;
}
function array(value: unknown, path: string): unknown[] { if (!Array.isArray(value)) fail(`${path} must be an array`); return value; }
function text(value: unknown, path: string, max = 200): string { if (typeof value !== "string" || value.length === 0 || value.length > max) fail(`${path} must be a non-empty string no longer than ${max}`); return value; }
function optionalText(value: unknown, path: string, max = 200): string | undefined { return value === undefined ? undefined : text(value, path, max); }
function identifier(value: unknown, path: string): string { const result = text(value, path, 200); if (!IDENTIFIER.test(result)) fail(`${path} contains unsupported characters`); return result; }
function integer(value: unknown, path: string, minimum?: number): number { if (typeof value !== "number" || !Number.isSafeInteger(value) || (minimum !== undefined && value < minimum)) fail(`${path} must be a safe integer${minimum === undefined ? "" : ` >= ${minimum}`}`); return value; }
function optionalInteger(value: unknown, path: string, minimum?: number): number | undefined { return value === undefined ? undefined : integer(value, path, minimum); }
function boolean(value: unknown, path: string): boolean { if (typeof value !== "boolean") fail(`${path} must be a boolean`); return value; }
function optionalBoolean(value: unknown, path: string): boolean | undefined { return value === undefined ? undefined : boolean(value, path); }
function exact(value: unknown, expected: unknown, path: string): void { if (value !== expected) fail(`${path} must be ${String(expected)}`); }
function optionalObject(value: unknown, path: string): JsonObject | undefined { return value === undefined ? undefined : object(value, path) as JsonObject; }
function optionalJsonValue(value: unknown, path: string) { return value === undefined ? undefined : jsonValue(value, path); }
function jsonValue(value: unknown, path: string): any { try { JSON.stringify(value); } catch { fail(`${path} must be JSON-serializable`); } if (value === undefined || typeof value === "bigint" || typeof value === "function" || typeof value === "symbol") fail(`${path} must be JSON-serializable`); return value; }
function optionalTextFields(item: Record<string, unknown>, keys: string[]) { const result: Record<string, string> = {}; for (const key of keys) { const value = optionalText(item[key], key, 1000); if (value !== undefined) result[key] = value; } return result; }
function compact<T extends Record<string, unknown>>(value: T): Partial<T> { return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as Partial<T>; }
function fail(message: string): never { throw new ValidationError(message); }
