PRAGMA foreign_keys = ON;

CREATE TABLE commits (
  sha TEXT PRIMARY KEY CHECK (length(sha) BETWEEN 40 AND 64),
  repository TEXT NOT NULL,
  ref TEXT,
  committed_at TEXT,
  url TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE environments (
  fingerprint TEXT PRIMARY KEY CHECK (length(fingerprint) = 64),
  label TEXT,
  captured_at TEXT NOT NULL,
  architecture TEXT,
  operating_system TEXT,
  kernel TEXT,
  cpu_model TEXT,
  cpu_sockets INTEGER,
  cpu_cores INTEGER,
  cpu_threads INTEGER,
  memory_bytes INTEGER,
  accelerator_count INTEGER NOT NULL DEFAULT 0,
  accelerator_models TEXT NOT NULL DEFAULT '',
  profile_json TEXT NOT NULL,
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE programs (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
  category TEXT NOT NULL,
  variant TEXT,
  tags_json TEXT NOT NULL DEFAULT '[]',
  input_generator_json TEXT NOT NULL DEFAULT '{}',
  comparison_policy_json TEXT NOT NULL DEFAULT '{}',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  first_seen_commit TEXT NOT NULL REFERENCES commits(sha),
  last_seen_commit TEXT NOT NULL REFERENCES commits(sha),
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  project_commit TEXT NOT NULL REFERENCES commits(sha),
  environment_fingerprint TEXT NOT NULL REFERENCES environments(fingerprint),
  suite TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('pass', 'fail', 'partial')),
  device TEXT NOT NULL,
  dtype TEXT,
  timing_scope TEXT NOT NULL,
  input_profile_json TEXT NOT NULL DEFAULT '{}',
  warmups INTEGER NOT NULL CHECK (warmups >= 0),
  repetitions INTEGER NOT NULL CHECK (repetitions > 0),
  seed INTEGER,
  command TEXT,
  artifact_url TEXT,
  runner_version TEXT,
  payload_sha256 TEXT NOT NULL CHECK (length(payload_sha256) = 64),
  metadata_json TEXT NOT NULL DEFAULT '{}',
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE results (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  program_id TEXT NOT NULL REFERENCES programs(id),
  backend TEXT NOT NULL,
  backend_version TEXT,
  execution_mode TEXT NOT NULL,
  correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
  skipped INTEGER NOT NULL DEFAULT 0 CHECK (skipped IN (0, 1)),
  error TEXT,
  input_size INTEGER,
  cold_ns INTEGER,
  median_ns INTEGER,
  min_ns INTEGER,
  max_ns INTEGER,
  p95_ns INTEGER,
  timings_json TEXT NOT NULL DEFAULT '[]',
  output_sha256 TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, program_id, backend, execution_mode)
);

CREATE TABLE capability_snapshots (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  backend TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64),
  corpus_programs INTEGER NOT NULL,
  glyphs_total INTEGER NOT NULL,
  monadic_supported INTEGER NOT NULL,
  dyadic_supported INTEGER NOT NULL,
  folds_supported INTEGER NOT NULL,
  tests_passed INTEGER NOT NULL,
  tests_failed INTEGER NOT NULL,
  tests_skipped INTEGER NOT NULL,
  value_domain TEXT,
  manifest_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, backend)
);

CREATE TABLE capability_features (
  run_id TEXT NOT NULL,
  backend TEXT NOT NULL,
  feature_id TEXT NOT NULL,
  glyph TEXT,
  name TEXT NOT NULL,
  valence TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('accelerated', 'fallback', 'unsupported', 'supported')),
  domain TEXT,
  behavior TEXT,
  evidence_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, backend, feature_id),
  FOREIGN KEY (run_id, backend) REFERENCES capability_snapshots(run_id, backend) ON DELETE CASCADE
);

CREATE INDEX runs_commit_idx ON runs(project_commit);
CREATE INDEX runs_started_idx ON runs(started_at DESC);
CREATE INDEX results_program_idx ON results(program_id, backend);
CREATE INDEX capability_started_idx ON capability_snapshots(backend, run_id);
