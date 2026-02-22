//! Episodic log — structured record of completed agent tasks.
//!
//! Every time the agent finishes a task (success or failure), a single row is
//! appended to the `episodes` table. The log lets the planner (L3) look back at
//! similar past goals and avoid repeating mistakes.
//!
//! # Write path
//! `PlanExecutor` → `EpisodicLog::insert` (called at the end of every task)
//!
//! # Read path
//! `Planner::plan` → `EpisodicLog::recent` / `EpisodicLog::find_similar_goals`

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

/// The outcome of an agent task execution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Failure,
}

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Failure => "failure",
        }
    }
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Outcome {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "success" => Ok(Outcome::Success),
            "failure" => Ok(Outcome::Failure),
            other => Err(format!("unknown outcome: '{other}'")),
        }
    }
}

/// A single step taken during a task execution.
///
/// Stored as elements of the `steps JSONB []` array column in `episodes`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskStep {
    /// Tool/action name (e.g. `"url_fetch"`, `"code_exec"`, `"plan"`)
    pub action: String,
    /// The raw input passed to the action (arbitrary JSON)
    pub input: Value,
    /// Whether this individual step succeeded
    pub success: bool,
    /// Error message if `success == false`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl TaskStep {
    pub fn success(action: impl Into<String>, input: Value) -> Self {
        Self {
            action: action.into(),
            input,
            success: true,
            error: None,
        }
    }

    pub fn failure(action: impl Into<String>, input: Value, error: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            input,
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Data needed to record one completed task.
///
/// Passed to [`EpisodicLog::insert`].
///
/// # Invariant
/// `error` must be `Some` when `outcome == Outcome::Failure` and `None` when
/// `outcome == Outcome::Success`, matching the `0002_create_episodes_table.sql`
/// migration comment. [`EpisodicLog::insert`] enforces this with `debug_assert!`.
pub struct EpisodeInput<'a> {
    pub task_goal: &'a str,
    pub steps: &'a [TaskStep],
    pub outcome: Outcome,
    /// `Some(msg)` when `outcome == Outcome::Failure`; `None` when success.
    pub error: Option<&'a str>,
    /// Wall-clock duration of the full task in milliseconds.
    pub duration_ms: i64,
}

/// A single episode record as read back from the `episodes` table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EpisodeRecord {
    pub id: Uuid,
    pub task_goal: String,
    /// The raw JSONB steps array, deserialized to `Vec<TaskStep>`.
    pub steps: Vec<TaskStep>,
    pub outcome: Outcome,
    pub error: Option<String>,
    pub duration_ms: i64,
    pub created_at: DateTime<Utc>,
}

/// Errors from episodic log operations.
#[derive(Debug, thiserror::Error)]
pub enum EpisodeError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("invalid outcome value in database: '{0}'")]
    InvalidOutcome(String),

    #[error("failed to deserialize steps: {0}")]
    Deserialization(#[from] serde_json::Error),
}

/// Append-only log of completed agent tasks.
///
/// Wraps a `PgPool` and operates on the `episodes` table created in migration
/// `0002_create_episodes_table.sql`. All operations are async.
#[derive(Clone, Debug)]
pub struct EpisodicLog {
    pool: PgPool,
}

impl EpisodicLog {
    /// Create a new `EpisodicLog` backed by the given connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Return the underlying pool (for sharing with other subsystems).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Append a completed task record.
    ///
    /// Returns the newly assigned episode `Uuid`.
    pub async fn insert(&self, ep: EpisodeInput<'_>) -> Result<Uuid, EpisodeError> {
        // Enforce the migration invariant: error is Some iff outcome is Failure.
        debug_assert!(
            matches!(ep.outcome, Outcome::Failure) == ep.error.is_some(),
            "EpisodeInput invariant violated: error must be Some on Failure and None on Success"
        );

        let steps_json = serde_json::to_value(ep.steps)?;
        let outcome_str = ep.outcome.as_str();

        let row = sqlx::query!(
            r#"
            INSERT INTO episodes (task_goal, steps, outcome, error, duration_ms)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            "#,
            ep.task_goal,
            steps_json,
            outcome_str,
            ep.error,
            ep.duration_ms,
        )
        .fetch_one(&self.pool)
        .await?;

        tracing::debug!(
            id = %row.id,
            outcome = outcome_str,
            duration_ms = ep.duration_ms,
            "episodic_log: inserted episode"
        );
        Ok(row.id)
    }

    /// Return the `limit` most recent episodes, newest first.
    pub async fn recent(&self, limit: usize) -> Result<Vec<EpisodeRecord>, EpisodeError> {
        debug_assert!(limit > 0, "limit must be > 0");
        let limit = limit as i64;

        let rows = sqlx::query!(
            r#"
            SELECT id, task_goal, steps, outcome, error, duration_ms, created_at
            FROM episodes
            ORDER BY created_at DESC
            LIMIT $1
            "#,
            limit,
        )
        .fetch_all(&self.pool)
        .await?;

        let records = rows
            .into_iter()
            .map(|row| {
                map_episode_row(RawEpisodeRow {
                    id: row.id,
                    task_goal: row.task_goal,
                    steps: row.steps,
                    outcome: row.outcome,
                    error: row.error,
                    duration_ms: row.duration_ms,
                    created_at: row.created_at,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        tracing::debug!(
            count = records.len(),
            "episodic_log: fetched recent episodes"
        );
        Ok(records)
    }

    /// Search for past episodes whose `task_goal` contains `keyword` (case-insensitive,
    /// **textual** substring match via `ILIKE`).
    ///
    /// Returns up to `limit` matches, newest first. Used by the planner to retrieve
    /// prior attempts at textually similar goals.
    ///
    /// > **Note:** This is a substring search, not semantic search. A future
    /// > enhancement could embed `task_goal` with pgvector and use cosine distance
    /// > here instead.
    ///
    /// ILIKE wildcard characters (`%`, `_`, `\`) in `keyword` are escaped so they
    /// match literally rather than acting as patterns.
    pub async fn find_similar_goals(
        &self,
        keyword: &str,
        limit: usize,
    ) -> Result<Vec<EpisodeRecord>, EpisodeError> {
        debug_assert!(limit > 0, "limit must be > 0");
        // Escape ILIKE special characters so the keyword is treated as a literal
        // substring. PostgreSQL's default ILIKE escape char is `\`.
        let escaped = keyword
            .replace('\\', r"\\")
            .replace('%', r"\%")
            .replace('_', r"\_");
        let pattern = format!("%{escaped}%");
        let limit = limit as i64;

        let rows = sqlx::query!(
            r#"
            SELECT id, task_goal, steps, outcome, error, duration_ms, created_at
            FROM episodes
            WHERE task_goal ILIKE $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
            pattern,
            limit,
        )
        .fetch_all(&self.pool)
        .await?;

        let records = rows
            .into_iter()
            .map(|row| {
                map_episode_row(RawEpisodeRow {
                    id: row.id,
                    task_goal: row.task_goal,
                    steps: row.steps,
                    outcome: row.outcome,
                    error: row.error,
                    duration_ms: row.duration_ms,
                    created_at: row.created_at,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        tracing::debug!(
            keyword,
            count = records.len(),
            "episodic_log: find_similar_goals"
        );
        Ok(records)
    }

    /// Delete an episode by ID.
    ///
    /// Returns `Ok(true)` if deleted, `Ok(false)` if not found.
    pub async fn delete(&self, id: Uuid) -> Result<bool, EpisodeError> {
        let result = sqlx::query!("DELETE FROM episodes WHERE id = $1", id)
            .execute(&self.pool)
            .await?;
        let deleted = result.rows_affected() > 0;
        tracing::debug!(%id, deleted, "episodic_log: deleted episode");
        Ok(deleted)
    }
}

// -------------------------------------------------------------------------
// Private helpers
// -------------------------------------------------------------------------

/// Raw columns returned by SELECT queries, bundled to avoid long arg lists.
struct RawEpisodeRow {
    id: Uuid,
    task_goal: String,
    steps: Value,
    outcome: String,
    error: Option<String>,
    duration_ms: i64,
    created_at: DateTime<Utc>,
}

fn map_episode_row(row: RawEpisodeRow) -> Result<EpisodeRecord, EpisodeError> {
    let outcome = row
        .outcome
        .parse::<Outcome>()
        .map_err(|_| EpisodeError::InvalidOutcome(row.outcome.clone()))?;
    let steps: Vec<TaskStep> = serde_json::from_value(row.steps)?;
    Ok(EpisodeRecord {
        id: row.id,
        task_goal: row.task_goal,
        steps,
        outcome,
        error: row.error,
        duration_ms: row.duration_ms,
        created_at: row.created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::connect_db;
    use serde_json::json;

    // -------------------------------------------------------------------------
    // Unit tests — no database required
    // -------------------------------------------------------------------------

    #[test]
    fn outcome_roundtrip_success() {
        let o = Outcome::Success;
        assert_eq!(o.as_str(), "success");
        assert_eq!("success".parse::<Outcome>().unwrap(), Outcome::Success);
        assert_eq!(o.to_string(), "success");
    }

    #[test]
    fn outcome_roundtrip_failure() {
        let o = Outcome::Failure;
        assert_eq!(o.as_str(), "failure");
        assert_eq!("failure".parse::<Outcome>().unwrap(), Outcome::Failure);
    }

    #[test]
    fn outcome_unknown_errors() {
        assert!("unknown".parse::<Outcome>().is_err());
    }

    #[test]
    fn task_step_success_ctor() {
        let step = TaskStep::success("url_fetch", json!({"url": "https://example.com"}));
        assert!(step.success);
        assert!(step.error.is_none());
        assert_eq!(step.action, "url_fetch");
    }

    #[test]
    fn task_step_failure_ctor() {
        let step = TaskStep::failure("code_exec", json!({"code": "bad"}), "exit 1");
        assert!(!step.success);
        assert_eq!(step.error.as_deref(), Some("exit 1"));
    }

    #[test]
    fn task_step_serializes_without_error_field_when_none() {
        let step = TaskStep::success("tool", json!(null));
        let s = serde_json::to_string(&step).unwrap();
        assert!(
            !s.contains("error"),
            "error field should be skipped when None"
        );
    }

    // -------------------------------------------------------------------------
    // Integration tests — require a running Postgres
    // Run with: cargo test episodic -- --ignored
    // -------------------------------------------------------------------------

    fn sample_steps() -> Vec<TaskStep> {
        vec![
            TaskStep::success("url_fetch", json!({"url": "https://example.com"})),
            TaskStep::failure("code_exec", json!({"code": "broken"}), "exit code 1"),
        ]
    }

    async fn cleanup(log: &EpisodicLog, id: Uuid) {
        log.delete(id).await.expect("cleanup delete failed");
    }

    #[tokio::test]
    #[ignore]
    async fn insert_and_recent_returns_episode() {
        let log = EpisodicLog::new(connect_db().await);

        let id = log
            .insert(EpisodeInput {
                task_goal: "fetch and summarise the Rust blog",
                steps: &sample_steps(),
                outcome: Outcome::Success,
                error: None,
                duration_ms: 1234,
            })
            .await
            .expect("insert failed");

        let recent = log.recent(5).await.expect("recent failed");
        let ep = recent
            .iter()
            .find(|e| e.id == id)
            .expect("episode not found in recent");

        assert_eq!(ep.task_goal, "fetch and summarise the Rust blog");
        assert_eq!(ep.outcome, Outcome::Success);
        assert_eq!(ep.duration_ms, 1234);
        assert_eq!(ep.steps.len(), 2);
        assert!(ep.error.is_none());

        cleanup(&log, id).await;
    }

    #[tokio::test]
    #[ignore]
    async fn insert_failure_stores_error_message() {
        let log = EpisodicLog::new(connect_db().await);

        let id = log
            .insert(EpisodeInput {
                task_goal: "run a broken script",
                steps: &sample_steps(),
                outcome: Outcome::Failure,
                error: Some("docker exec timed out"),
                duration_ms: 5000,
            })
            .await
            .expect("insert failed");

        let recent = log.recent(5).await.expect("recent failed");
        let ep = recent.iter().find(|e| e.id == id).unwrap();

        assert_eq!(ep.outcome, Outcome::Failure);
        assert_eq!(ep.error.as_deref(), Some("docker exec timed out"));

        cleanup(&log, id).await;
    }

    #[tokio::test]
    #[ignore]
    async fn find_similar_goals_returns_matching_episodes() {
        let log = EpisodicLog::new(connect_db().await);

        let id = log
            .insert(EpisodeInput {
                task_goal: "summarise the weekly engineering report",
                steps: &[],
                outcome: Outcome::Success,
                error: None,
                duration_ms: 800,
            })
            .await
            .expect("insert failed");

        let matches = log
            .find_similar_goals("engineering", 10)
            .await
            .expect("find_similar_goals failed");

        assert!(
            matches.iter().any(|e| e.id == id),
            "should find the inserted episode"
        );

        cleanup(&log, id).await;
    }

    #[tokio::test]
    #[ignore]
    async fn find_similar_goals_case_insensitive() {
        let log = EpisodicLog::new(connect_db().await);

        let id = log
            .insert(EpisodeInput {
                task_goal: "Deploy to Production",
                steps: &[],
                outcome: Outcome::Success,
                error: None,
                duration_ms: 200,
            })
            .await
            .expect("insert failed");

        // Search in different case
        let matches = log
            .find_similar_goals("deploy to production", 10)
            .await
            .expect("find_similar_goals failed");

        assert!(matches.iter().any(|e| e.id == id));

        cleanup(&log, id).await;
    }

    #[tokio::test]
    #[ignore]
    async fn steps_roundtrip_through_jsonb() {
        let log = EpisodicLog::new(connect_db().await);
        let steps = sample_steps();

        let id = log
            .insert(EpisodeInput {
                task_goal: "steps roundtrip test",
                steps: &steps,
                outcome: Outcome::Success,
                error: None,
                duration_ms: 1,
            })
            .await
            .expect("insert failed");

        let recent = log.recent(5).await.expect("recent failed");
        let ep = recent.iter().find(|e| e.id == id).unwrap();

        assert_eq!(ep.steps.len(), 2);
        assert_eq!(ep.steps[0].action, "url_fetch");
        assert!(ep.steps[0].success);
        assert_eq!(ep.steps[1].action, "code_exec");
        assert!(!ep.steps[1].success);
        assert_eq!(ep.steps[1].error.as_deref(), Some("exit code 1"));

        cleanup(&log, id).await;
    }

    #[tokio::test]
    #[ignore]
    async fn delete_nonexistent_returns_false() {
        let log = EpisodicLog::new(connect_db().await);
        let deleted = log.delete(Uuid::new_v4()).await.expect("delete failed");
        assert!(!deleted);
    }

    #[tokio::test]
    #[ignore]
    async fn recent_is_ordered_newest_first() {
        let log = EpisodicLog::new(connect_db().await);

        // Use a unique UUID marker so we can isolate our rows from any other
        // concurrent tests that are also writing to the episodes table.
        let marker = format!("ordering-test:{}", Uuid::new_v4());

        let id1 = log
            .insert(EpisodeInput {
                task_goal: &format!("{marker}:first"),
                steps: &[],
                outcome: Outcome::Success,
                error: None,
                duration_ms: 1,
            })
            .await
            .expect("insert 1 failed");

        let id2 = log
            .insert(EpisodeInput {
                task_goal: &format!("{marker}:second"),
                steps: &[],
                outcome: Outcome::Success,
                error: None,
                duration_ms: 1,
            })
            .await
            .expect("insert 2 failed");

        // find_similar_goals also orders by created_at DESC; using our unique
        // marker guarantees only our two rows appear in the result.
        let matches = log
            .find_similar_goals(&marker, 10)
            .await
            .expect("find_similar_goals failed");

        assert_eq!(matches.len(), 2, "expected exactly our 2 test rows");
        // id2 was inserted last, so it must appear first (newest-first order).
        assert_eq!(matches[0].id, id2, "newer episode should be first");
        assert_eq!(matches[1].id, id1, "older episode should be second");

        log.delete(id1).await.expect("cleanup id1 failed");
        log.delete(id2).await.expect("cleanup id2 failed");
    }
}
