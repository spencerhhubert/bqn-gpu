import type { Env, IngestPayload, JsonValue } from "./model";

const JSON_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Cache-Control": "public, max-age=15, stale-while-revalidate=60",
  "Content-Type": "application/json; charset=utf-8",
};

export async function routeApi(request: Request, env: Env, url: URL): Promise<Response> {
  const path = url.pathname.replace(/\/+$/, "") || "/";
  if (request.method === "GET" && path === "/api/v1/health") return health(env);
  if (request.method === "GET" && path === "/api/v1/schema") return schema();
  if (request.method === "GET" && path === "/api/v1/summary") return summary(env);
  if (request.method === "GET" && path === "/api/v1/runs") return runs(env, url);
  if (request.method === "GET" && path.startsWith("/api/v1/runs/")) return runBundle(env, decodeURIComponent(path.slice(13)));
  if (request.method === "GET" && path === "/api/v1/performance") return performance(env, url);
  if (request.method === "GET" && path === "/api/v1/capability") return capability(env, url);
  if (request.method === "GET" && path === "/api/v1/programs") return programs(env, url);
  if (request.method === "GET" && path === "/api/v1/environments") return environments(env, url);
  return error(404, "not_found", "No API route matches this request");
}

export async function ingest(env: Env, payload: IngestPayload, payloadSha256: string): Promise<Response> {
  const existing = await env.DB.prepare("SELECT payload_sha256 FROM runs WHERE id = ?")
    .bind(payload.run.id)
    .first<{ payload_sha256: string }>();
  if (existing) {
    if (existing.payload_sha256 === payloadSha256) return json({ ok: true, run_id: payload.run.id, idempotent: true }, 200, false);
    return error(409, "run_conflict", "That run ID already exists with different content");
  }

  const ids = payload.programs.map((program) => program.id);
  const known = ids.length
    ? await env.DB.prepare(
        "SELECT id, source_sha256 FROM programs WHERE id IN (SELECT value FROM json_each(?))",
      ).bind(JSON.stringify(ids)).all<{ id: string; source_sha256: string }>()
    : { results: [] };
  const requestedHashes = new Map(payload.programs.map((program) => [program.id, program.source_sha256]));
  const mismatch = known.results.find((program: { id: string; source_sha256: string }) => requestedHashes.get(program.id) !== program.source_sha256);
  if (mismatch) return error(409, "program_conflict", `Program ${mismatch.id} already has a different source hash`);

  const acceleratorModels = payload.environment.accelerators
    .map((accelerator) => `${accelerator.count ?? 1}× ${accelerator.model}`)
    .join(", ");
  const statements: D1PreparedStatement[] = [
    env.DB.prepare(`
      INSERT INTO commits (sha, repository, ref, committed_at, url, metadata_json)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(sha) DO UPDATE SET
        repository = excluded.repository,
        ref = COALESCE(excluded.ref, commits.ref),
        committed_at = COALESCE(excluded.committed_at, commits.committed_at),
        url = COALESCE(excluded.url, commits.url),
        metadata_json = excluded.metadata_json
    `).bind(
      payload.commit.sha,
      payload.commit.repository,
      payload.commit.ref ?? null,
      payload.commit.committed_at ?? null,
      payload.commit.url ?? null,
      stringify(payload.commit.metadata ?? {}),
    ),
    env.DB.prepare(`
      INSERT INTO environments (
        fingerprint, label, captured_at, architecture, operating_system, kernel,
        cpu_model, cpu_sockets, cpu_cores, cpu_threads, memory_bytes,
        accelerator_count, accelerator_models, profile_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(fingerprint) DO UPDATE SET
        label = COALESCE(excluded.label, environments.label),
        captured_at = excluded.captured_at,
        profile_json = excluded.profile_json
    `).bind(
      payload.environment.fingerprint,
      payload.environment.label ?? null,
      payload.environment.captured_at,
      payload.environment.architecture ?? null,
      payload.environment.operating_system ?? null,
      payload.environment.kernel ?? null,
      payload.environment.cpu?.model ?? null,
      payload.environment.cpu?.sockets ?? null,
      payload.environment.cpu?.cores ?? null,
      payload.environment.cpu?.threads ?? null,
      payload.environment.memory_bytes ?? null,
      payload.environment.accelerators.reduce((sum, accelerator) => sum + (accelerator.count ?? 1), 0),
      acceleratorModels,
      stringify(payload.environment),
    ),
    env.DB.prepare(`
      INSERT INTO programs (
        id, source, source_sha256, category, variant, tags_json,
        input_generator_json, comparison_policy_json, metadata_json,
        first_seen_commit, last_seen_commit
      )
      SELECT
        json_extract(value, '$.id'), json_extract(value, '$.source'),
        json_extract(value, '$.source_sha256'), json_extract(value, '$.category'),
        json_extract(value, '$.variant'), COALESCE(json_extract(value, '$.tags_json'), '[]'),
        COALESCE(json_extract(value, '$.input_generator_json'), '{}'),
        COALESCE(json_extract(value, '$.comparison_policy_json'), '{}'),
        COALESCE(json_extract(value, '$.metadata_json'), '{}'), ?, ?
      FROM json_each(?) WHERE true
      ON CONFLICT(id) DO UPDATE SET
        last_seen_commit = excluded.last_seen_commit,
        category = excluded.category,
        variant = excluded.variant,
        tags_json = excluded.tags_json,
        input_generator_json = excluded.input_generator_json,
        comparison_policy_json = excluded.comparison_policy_json,
        metadata_json = excluded.metadata_json
    `).bind(payload.commit.sha, payload.commit.sha, stringify(payload.programs.map(programRow))),
    env.DB.prepare(`
      INSERT INTO runs (
        id, schema_version, project_commit, environment_fingerprint, suite,
        started_at, finished_at, status, device, dtype, timing_scope,
        input_profile_json, warmups, repetitions, seed, command, artifact_url,
        runner_version, payload_sha256, metadata_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      payload.run.id,
      payload.schema_version,
      payload.commit.sha,
      payload.environment.fingerprint,
      payload.run.suite,
      payload.run.started_at,
      payload.run.finished_at ?? null,
      payload.run.status,
      payload.run.device,
      payload.run.dtype ?? null,
      payload.run.timing_scope,
      stringify(payload.run.input_profile),
      payload.run.warmups,
      payload.run.repetitions,
      payload.run.seed ?? null,
      payload.run.command ?? null,
      payload.run.artifact_url ?? null,
      payload.run.runner_version ?? null,
      payloadSha256,
      stringify(payload.run.metadata ?? {}),
    ),
    env.DB.prepare(`
      INSERT INTO results (
        run_id, program_id, backend, backend_version, execution_mode, timing_scope, correct,
        skipped, error, input_size, cold_ns, median_ns, min_ns, max_ns, p95_ns,
        timings_json, output_sha256, metadata_json
      )
      SELECT ?,
        json_extract(value, '$.program_id'), json_extract(value, '$.backend'),
        json_extract(value, '$.backend_version'), json_extract(value, '$.execution_mode'),
        json_extract(value, '$.timing_scope'), json_extract(value, '$.correct'), json_extract(value, '$.skipped'),
        json_extract(value, '$.error'), json_extract(value, '$.input_size'),
        json_extract(value, '$.cold_ns'), json_extract(value, '$.median_ns'),
        json_extract(value, '$.min_ns'), json_extract(value, '$.max_ns'),
        json_extract(value, '$.p95_ns'), COALESCE(json_extract(value, '$.timings_json'), '[]'),
        json_extract(value, '$.output_sha256'), COALESCE(json_extract(value, '$.metadata_json'), '{}')
      FROM json_each(?)
    `).bind(payload.run.id, stringify(payload.results.map(resultRow))),
  ];

  if (payload.capability) {
    const value = payload.capability;
    statements.push(
      env.DB.prepare(`
        INSERT INTO capability_snapshots (
          run_id, backend, manifest_sha256, corpus_programs, glyphs_total,
          monadic_supported, dyadic_supported, folds_supported, tests_passed,
          tests_failed, tests_skipped, value_domain, manifest_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        payload.run.id, value.backend, value.manifest_sha256, value.corpus_programs,
        value.glyphs_total, value.monadic_supported, value.dyadic_supported,
        value.folds_supported, value.tests_passed, value.tests_failed, value.tests_skipped,
        value.value_domain ?? null, stringify(value.manifest), stringify(value.metadata ?? {}),
      ),
    );
    if (value.features?.length) {
      statements.push(
        env.DB.prepare(`
          INSERT INTO capability_features (
            run_id, backend, feature_id, glyph, name, valence, status,
            domain, behavior, evidence_json, metadata_json
          )
          SELECT ?, ?, json_extract(value, '$.id'), json_extract(value, '$.glyph'),
            json_extract(value, '$.name'), json_extract(value, '$.valence'),
            json_extract(value, '$.status'), json_extract(value, '$.domain'),
            json_extract(value, '$.behavior'), COALESCE(json_extract(value, '$.evidence_json'), '[]'),
            COALESCE(json_extract(value, '$.metadata_json'), '{}')
          FROM json_each(?)
        `).bind(payload.run.id, value.backend, stringify(value.features.map(featureRow))),
      );
    }
  }

  await env.DB.batch(statements);
  return json({
    ok: true,
    run_id: payload.run.id,
    commit: payload.commit.sha,
    programs: payload.programs.length,
    results: payload.results.length,
    capability: Boolean(payload.capability),
    idempotent: false,
  }, 201, false);
}

async function health(env: Env): Promise<Response> {
  const row = await env.DB.prepare("SELECT count(*) AS runs FROM runs").first<{ runs: number }>();
  return json({ ok: true, service: "bqn-gpu-website", schema_version: 1, runs: row?.runs ?? 0 });
}

async function summary(env: Env): Promise<Response> {
  const [counts, latest, capabilityRow, backends] = await env.DB.batch([
    env.DB.prepare(`SELECT
      (SELECT count(*) FROM runs) AS runs,
      (SELECT count(*) FROM commits) AS commits,
      (SELECT count(*) FROM programs) AS programs,
      (SELECT count(*) FROM results) AS results,
      (SELECT count(*) FROM results WHERE correct = 1 AND skipped = 0) AS correct_results`),
    env.DB.prepare(`SELECT r.id, r.project_commit, r.started_at, r.status, r.device,
      r.suite, r.timing_scope, e.cpu_model, e.accelerator_models
      FROM runs r JOIN environments e ON e.fingerprint = r.environment_fingerprint
      ORDER BY r.started_at DESC LIMIT 1`),
    env.DB.prepare(`SELECT c.run_id, c.backend, c.manifest_sha256,
      c.corpus_programs, c.glyphs_total, c.monadic_supported,
      c.dyadic_supported, c.folds_supported, c.tests_passed,
      c.tests_failed, c.tests_skipped, c.value_domain,
      r.project_commit, r.started_at
      FROM capability_snapshots c JOIN runs r ON r.id = c.run_id
      ORDER BY r.started_at DESC LIMIT 1`),
    env.DB.prepare(`SELECT backend, count(*) AS results, count(DISTINCT run_id) AS runs
      FROM results GROUP BY backend ORDER BY backend`),
  ]);
  return json({
    counts: counts.results[0] ?? { runs: 0, commits: 0, programs: 0, results: 0, correct_results: 0 },
    latest_run: latest.results[0] ?? null,
    latest_capability: capabilityRow.results[0] ?? null,
    backends: backends.results,
  });
}

async function runs(env: Env, url: URL): Promise<Response> {
  const max = limit(url, 100, 500);
  const commit = url.searchParams.get("commit");
  const query = `SELECT r.*, e.label AS environment_label, e.cpu_model, e.accelerator_models,
    (SELECT count(*) FROM results x WHERE x.run_id = r.id) AS result_count,
    (SELECT count(*) FROM results x WHERE x.run_id = r.id AND x.correct = 0) AS incorrect_count
    FROM runs r JOIN environments e ON e.fingerprint = r.environment_fingerprint
    ${commit ? "WHERE r.project_commit = ?" : ""}
    ORDER BY r.started_at DESC LIMIT ?`;
  const statement = commit ? env.DB.prepare(query).bind(commit, max) : env.DB.prepare(query).bind(max);
  const result = await statement.all();
  return json({ runs: result.results.map(parseRunRow) });
}

async function runBundle(env: Env, id: string): Promise<Response> {
  if (!id || id.length > 200) return error(400, "invalid_run", "Invalid run ID");
  const run = await env.DB.prepare(`SELECT r.*, c.repository, c.ref, c.committed_at, c.url AS commit_url,
    c.metadata_json AS commit_metadata_json, e.profile_json
    FROM runs r JOIN commits c ON c.sha = r.project_commit
    JOIN environments e ON e.fingerprint = r.environment_fingerprint WHERE r.id = ?`).bind(id).first();
  if (!run) return error(404, "run_not_found", "Run not found");
  const [results, programRows, capabilityRows, features] = await env.DB.batch([
    env.DB.prepare("SELECT * FROM results WHERE run_id = ? ORDER BY program_id, backend").bind(id),
    env.DB.prepare(`SELECT DISTINCT p.* FROM programs p
      JOIN results x ON x.program_id = p.id
      WHERE x.run_id = ? ORDER BY p.id`).bind(id),
    env.DB.prepare("SELECT * FROM capability_snapshots WHERE run_id = ? ORDER BY backend").bind(id),
    env.DB.prepare("SELECT * FROM capability_features WHERE run_id = ? ORDER BY backend, feature_id").bind(id),
  ]);
  return json({
    run: parseRunRow(run as Record<string, unknown>),
    programs: programRows.results.map((row) => parseProgramRow(row as Record<string, unknown>)),
    results: results.results.map((row) => parseResultRow(row as Record<string, unknown>)),
    capabilities: capabilityRows.results.map((row) => parseCapabilityRow(row as Record<string, unknown>)),
    capability_features: features.results.map((row) => parseFeatureRow(row as Record<string, unknown>)),
  });
}

async function performance(env: Env, url: URL): Promise<Response> {
  const max = limit(url, 1000, 5000);
  const conditions = ["x.skipped = 0", "x.median_ns IS NOT NULL"];
  const values: unknown[] = [];
  for (const [parameter, column] of [["program_id", "x.program_id"], ["backend", "x.backend"], ["commit", "r.project_commit"]] as const) {
    const value = url.searchParams.get(parameter);
    if (value) { conditions.push(`${column} = ?`); values.push(value); }
  }
  const result = await env.DB.prepare(`SELECT * FROM (SELECT x.*, r.project_commit, r.started_at, r.device,
    r.dtype, r.input_profile_json, e.cpu_model, e.accelerator_models
    FROM results x JOIN runs r ON r.id = x.run_id
    JOIN environments e ON e.fingerprint = r.environment_fingerprint
    WHERE ${conditions.join(" AND ")} ORDER BY r.started_at DESC LIMIT ?) AS recent
    ORDER BY recent.started_at ASC`).bind(...values, max).all();
  return json({ results: result.results.map((row: Record<string, unknown>) => parseResultRow(row)) });
}

async function capability(env: Env, url: URL): Promise<Response> {
  const max = limit(url, 200, 1000);
  const backend = url.searchParams.get("backend");
  const result = await (backend
    ? env.DB.prepare(`SELECT * FROM (SELECT c.*, r.project_commit, r.started_at, r.device, e.accelerator_models
        FROM capability_snapshots c JOIN runs r ON r.id = c.run_id
        JOIN environments e ON e.fingerprint = r.environment_fingerprint
        WHERE c.backend = ? ORDER BY r.started_at DESC LIMIT ?) AS recent
        ORDER BY recent.started_at ASC`).bind(backend, max)
    : env.DB.prepare(`SELECT * FROM (SELECT c.*, r.project_commit, r.started_at, r.device, e.accelerator_models
        FROM capability_snapshots c JOIN runs r ON r.id = c.run_id
        JOIN environments e ON e.fingerprint = r.environment_fingerprint
        ORDER BY r.started_at DESC LIMIT ?) AS recent
        ORDER BY recent.started_at ASC`).bind(max)).all();
  return json({ snapshots: result.results.map((row: Record<string, unknown>) => parseCapabilityRow(row)) });
}

async function programs(env: Env, url: URL): Promise<Response> {
  const max = limit(url, 250, 1000);
  const result = await env.DB.prepare(`SELECT id, source, source_sha256, category, variant, tags_json,
    input_generator_json, comparison_policy_json, metadata_json,
    first_seen_commit, last_seen_commit FROM programs ORDER BY id LIMIT ?`).bind(max).all();
  return json({ programs: result.results.map((row) => parseProgramRow(row as Record<string, unknown>)) });
}

async function environments(env: Env, url: URL): Promise<Response> {
  const max = limit(url, 100, 500);
  const result = await env.DB.prepare(`SELECT e.*,
    (SELECT count(*) FROM runs r WHERE r.environment_fingerprint = e.fingerprint) AS run_count
    FROM environments e ORDER BY e.captured_at DESC LIMIT ?`).bind(max).all();
  return json({ environments: result.results.map((row) => parseJsonFields(row as Record<string, unknown>, ["profile_json"])) });
}

function schema(): Response {
  return json({
    schema_version: 1,
    ingestion: {
      endpoint: "/api/v1/ingest",
      authentication: "Authorization: Bearer <ingestion token>",
      immutable_identity: ["run.id", "program.id + program.source_sha256"],
      required_sections: ["commit", "environment", "run", "programs", "results"],
      optional_sections: ["capability"],
    },
    reproduction_bundle: "/api/v1/runs/{run_id}",
    timing_unit: "nanoseconds",
    notes: [
      "Raw repetition timings are retained alongside aggregates.",
      "Environment profiles include hardware, drivers, runtimes, and framework revisions.",
      "Unknown metadata keys are preserved as JSON for forward-compatible extensions.",
    ],
  });
}

function programRow(value: IngestPayload["programs"][number]) {
  return {
    ...value,
    tags_json: stringify(value.tags ?? []),
    input_generator_json: stringify(value.input_generator ?? {}),
    comparison_policy_json: stringify(value.comparison_policy ?? {}),
    metadata_json: stringify(value.metadata ?? {}),
  };
}

function resultRow(value: IngestPayload["results"][number]) {
  return {
    ...value,
    correct: value.correct ? 1 : 0,
    skipped: value.skipped ? 1 : 0,
    timings_json: stringify(value.timings_ns ?? []),
    metadata_json: stringify(value.metadata ?? {}),
  };
}

type CapabilityFeature = NonNullable<NonNullable<IngestPayload["capability"]>["features"]>[number];

function featureRow(value: CapabilityFeature) {
  return { ...value, evidence_json: stringify(value.evidence ?? []), metadata_json: stringify(value.metadata ?? {}) };
}

function parseRunRow(row: Record<string, unknown>) { return parseJsonFields(row, ["input_profile_json", "metadata_json", "commit_metadata_json", "profile_json"]); }
function parseProgramRow(row: Record<string, unknown>) { return parseJsonFields(row, ["tags_json", "input_generator_json", "comparison_policy_json", "metadata_json"]); }
function parseResultRow(row: Record<string, unknown>) { return parseJsonFields(row, ["timings_json", "metadata_json", "input_profile_json"]); }
function parseCapabilityRow(row: Record<string, unknown>) { return parseJsonFields(row, ["manifest_json", "metadata_json"]); }
function parseFeatureRow(row: Record<string, unknown>) { return parseJsonFields(row, ["evidence_json", "metadata_json"]); }

function parseJsonFields(row: Record<string, unknown>, fields: string[]) {
  const result = { ...row };
  for (const field of fields) {
    if (typeof result[field] === "string") {
      const outputName = field.replace(/_json$/, "");
      try { result[outputName] = JSON.parse(result[field] as string) as JsonValue; } catch { result[outputName] = null; }
      delete result[field];
    }
  }
  return result;
}

function limit(url: URL, fallback: number, maximum: number): number {
  const value = Number.parseInt(url.searchParams.get("limit") ?? "", 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, maximum) : fallback;
}

function stringify(value: unknown): string { return JSON.stringify(value); }

export function json(value: unknown, status = 200, cache = true): Response {
  const headers = { ...JSON_HEADERS };
  if (!cache) headers["Cache-Control"] = "no-store";
  return new Response(JSON.stringify(value), { status, headers });
}

export function error(status: number, code: string, message: string): Response {
  return json({ error: { code, message } }, status, false);
}
