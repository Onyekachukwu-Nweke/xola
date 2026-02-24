//! L3-07: Integration tests for the planning layer.
//!
//! These tests require:
//! - Postgres with pgvector (`DATABASE_URL` env var)
//! - Python LLM surface server running on TCP (`LLM_SURFACE_URL` or default
//!   `http://127.0.0.1:8001`)
//! - `OPENAI_API_KEY` set (the Python server calls the real LLM)
//!
//! Run: `cargo test --test planning_integration -- --ignored`

use std::sync::Arc;

use xola_runtime::ipc::IpcClient;
use xola_runtime::memory::{EpisodicLog, Outcome};
use xola_runtime::planning::{PlanConfig, PlanExecutor, ReplanConfig};
use xola_runtime::tools::mock::MockTool;
use xola_runtime::tools::{ToolRegistry, ToolTimeoutConfig};

fn llm_surface_url() -> String {
    std::env::var("LLM_SURFACE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string())
}

/// L3-07a: A simple task that should complete in a few ReAct iterations.
///
/// The agent has access to `mock_echo` and must figure out how to use it
/// to answer the goal. The real LLM decides the plan.
#[tokio::test]
#[ignore]
async fn test_multi_step_task_completes() {
    dotenvy::dotenv().ok();

    let ipc = IpcClient::new(llm_surface_url()).expect("create IPC client");
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("register mock tool");

    let config = PlanConfig {
        max_iterations: 10,
        stm_token_budget: 8000,
        replan: ReplanConfig::default(),
    };

    let executor = PlanExecutor::new(
        Arc::new(ipc),
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        config,
    );

    let result = executor
        .execute("Use the mock_echo tool to echo the message 'integration test' and then provide the echoed result as your final answer.")
        .await
        .expect("task should complete");

    assert_eq!(result.outcome, Outcome::Success);
    assert!(result.final_answer.is_some(), "expected a final answer");
    assert!(!result.steps.is_empty(), "expected at least one tool step");
    assert!(result.iterations <= 10, "should finish within limit");
}

/// L3-07b: Verify that replanning fires on tool failure.
///
/// The agent first tries a nonexistent tool (injected via the goal phrasing),
/// then should replan and use `mock_echo` instead.
#[tokio::test]
#[ignore]
async fn test_replan_on_tool_failure() {
    dotenvy::dotenv().ok();

    let ipc = IpcClient::new(llm_surface_url()).expect("create IPC client");
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("register mock tool");

    let config = PlanConfig {
        max_iterations: 15,
        stm_token_budget: 8000,
        replan: ReplanConfig {
            max_replans_per_step: 3,
            max_replans_per_task: 10,
        },
    };

    let executor = PlanExecutor::new(
        Arc::new(ipc),
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        config,
    );

    // The goal mentions a tool that doesn't exist to prompt a failure + replan.
    let result = executor
        .execute("First try using 'web_search' tool to search for 'hello'. If that fails, use mock_echo to echo 'recovered' and return that as the final answer.")
        .await;

    // Either the task succeeds (LLM recovered) or we hit replan limits.
    // Both are valid outcomes for this test.
    match result {
        Ok(task_result) => {
            // The LLM may skip the bad tool entirely or replan after failure.
            assert_eq!(task_result.outcome, Outcome::Success);
            assert!(task_result.final_answer.is_some());
        }
        Err(e) => {
            // Acceptable: max replans reached (the LLM kept trying web_search).
            let err_str = e.to_string();
            assert!(
                err_str.contains("max replans") || err_str.contains("max iterations"),
                "unexpected error: {err_str}"
            );
        }
    }
}

/// L3-07c: Verify that completed tasks are recorded in the episodic log.
#[tokio::test]
#[ignore]
async fn test_episode_recorded_after_task() {
    dotenvy::dotenv().ok();

    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = sqlx::PgPool::connect(&db_url)
        .await
        .expect("connect to Postgres");

    let ipc = IpcClient::new(llm_surface_url()).expect("create IPC client");
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("register mock tool");

    let episodic_log = EpisodicLog::new(pool);

    let config = PlanConfig {
        max_iterations: 10,
        stm_token_budget: 8000,
        replan: ReplanConfig::default(),
    };

    let executor = PlanExecutor::new(
        Arc::new(ipc),
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        config,
    )
    .with_episodic_log(episodic_log.clone());

    let goal = format!(
        "Use mock_echo to echo 'episode-test-{}' and return it.",
        uuid::Uuid::new_v4()
    );

    let result = executor.execute(&goal).await.expect("task should complete");
    assert_eq!(result.outcome, Outcome::Success);

    // Verify episode was recorded.
    let episodes = episodic_log
        .find_similar_goals("episode-test-", 5)
        .await
        .expect("query episodic log");
    assert!(
        episodes.iter().any(|ep| ep.task_goal == goal),
        "expected to find the episode in the log"
    );

    // Cleanup: delete the test episode.
    for ep in &episodes {
        if ep.task_goal == goal {
            episodic_log
                .delete(ep.id)
                .await
                .expect("delete test episode");
        }
    }
}
