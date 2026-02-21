-- L2-01: episodes table
-- Structured log of past task completions.
-- Each row captures the full execution history of one agent task so the
-- agent can learn from prior attempts when faced with a similar goal.

CREATE TABLE IF NOT EXISTS episodes (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    task_goal   TEXT        NOT NULL,
    -- JSON array of { action, input, success, error? } objects — one per step
    steps       JSONB       NOT NULL DEFAULT '[]',
    -- 'success' or 'failure'
    outcome     TEXT        NOT NULL,
    -- NULL when outcome = 'success'; populated on failure
    error       TEXT,
    duration_ms BIGINT      NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Most queries will want the most recent episodes first
CREATE INDEX IF NOT EXISTS episodes_created_at_idx
    ON episodes (created_at DESC);
