PRAGMA foreign_keys = OFF;

CREATE TABLE results_multi_size (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  program_id TEXT NOT NULL REFERENCES programs(id),
  backend TEXT NOT NULL,
  backend_version TEXT,
  execution_mode TEXT NOT NULL,
  timing_scope TEXT NOT NULL DEFAULT 'resident-compute',
  correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
  skipped INTEGER NOT NULL DEFAULT 0 CHECK (skipped IN (0, 1)),
  error TEXT,
  input_size INTEGER NOT NULL,
  cold_ns INTEGER,
  median_ns INTEGER,
  min_ns INTEGER,
  max_ns INTEGER,
  p95_ns INTEGER,
  timings_json TEXT NOT NULL DEFAULT '[]',
  output_sha256 TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, program_id, backend, execution_mode, input_size)
);

INSERT INTO results_multi_size (
  run_id, program_id, backend, backend_version, execution_mode, timing_scope,
  correct, skipped, error, input_size, cold_ns, median_ns, min_ns, max_ns,
  p95_ns, timings_json, output_sha256, metadata_json
)
SELECT
  run_id, program_id, backend, backend_version, execution_mode, timing_scope,
  correct, skipped, error, COALESCE(input_size, 0), cold_ns, median_ns, min_ns,
  max_ns, p95_ns, timings_json, output_sha256, metadata_json
FROM results;

DROP TABLE results;
ALTER TABLE results_multi_size RENAME TO results;

CREATE INDEX results_program_idx ON results(program_id, backend);

PRAGMA foreign_keys = ON;
