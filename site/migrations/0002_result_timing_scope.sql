ALTER TABLE results ADD COLUMN timing_scope TEXT NOT NULL DEFAULT 'resident-compute';

CREATE INDEX results_scope_idx ON results(timing_scope, backend, program_id);
