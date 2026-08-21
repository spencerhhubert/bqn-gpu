export interface Env {
  DB: D1Database;
  ASSETS: Fetcher;
  BENCHMARK_INGEST_TOKEN?: string;
}

export interface IngestPayload {
  schema_version: 1;
  commit: CommitRecord;
  environment: EnvironmentRecord;
  run: RunRecord;
  programs: ProgramRecord[];
  results: ResultRecord[];
  capability?: CapabilityRecord;
}

export interface CommitRecord {
  sha: string;
  repository: string;
  ref?: string;
  committed_at?: string;
  url?: string;
  metadata?: JsonObject;
}

export interface EnvironmentRecord {
  fingerprint: string;
  label?: string;
  captured_at: string;
  architecture?: string;
  operating_system?: string;
  kernel?: string;
  cpu?: {
    model?: string;
    sockets?: number;
    cores?: number;
    threads?: number;
  };
  memory_bytes?: number;
  accelerators: Array<{
    kind: string;
    vendor?: string;
    model: string;
    count?: number;
    memory_bytes?: number;
    compute_capability?: string;
    driver?: string;
    runtime?: string;
    metadata?: JsonObject;
  }>;
  software: Record<string, string>;
  metadata?: JsonObject;
}

export interface RunRecord {
  id: string;
  suite: string;
  started_at: string;
  finished_at?: string;
  status: "pass" | "fail" | "partial";
  device: string;
  dtype?: string;
  timing_scope: string;
  input_profile: JsonValue;
  warmups: number;
  repetitions: number;
  seed?: number;
  command?: string;
  artifact_url?: string;
  runner_version?: string;
  metadata?: JsonObject;
}

export interface ProgramRecord {
  id: string;
  source: string;
  source_sha256: string;
  category: string;
  variant?: string;
  tags?: string[];
  input_generator?: JsonValue;
  comparison_policy?: JsonValue;
  metadata?: JsonObject;
}

export interface ResultRecord {
  program_id: string;
  backend: string;
  backend_version?: string;
  execution_mode: string;
  timing_scope: string;
  correct: boolean;
  skipped?: boolean;
  error?: string;
  input_size?: number;
  cold_ns?: number;
  median_ns?: number;
  min_ns?: number;
  max_ns?: number;
  p95_ns?: number;
  timings_ns?: number[];
  output_sha256?: string;
  metadata?: JsonObject;
}

export interface CapabilityRecord {
  backend: string;
  manifest_sha256: string;
  corpus_programs: number;
  glyphs_total: number;
  monadic_supported: number;
  dyadic_supported: number;
  folds_supported: number;
  tests_passed: number;
  tests_failed: number;
  tests_skipped: number;
  value_domain?: string;
  manifest: JsonValue;
  features?: Array<{
    id: string;
    glyph?: string;
    name: string;
    valence: string;
    status: "accelerated" | "fallback" | "unsupported" | "supported";
    domain?: string;
    behavior?: string;
    evidence?: JsonValue;
    metadata?: JsonObject;
  }>;
  metadata?: JsonObject;
}

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject { [key: string]: JsonValue }
