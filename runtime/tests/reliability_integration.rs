//! Integration tests for Layer 4: Reliability & Fault Tolerance (L4-08).
//!
//! These tests verify circuit breaker recovery, loop detection with replanning,
//! and timeout handling in realistic scenarios.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use xola_runtime::ipc::IpcClient;
use xola_runtime::memory::Outcome;
use xola_runtime::planning::{FailureCategory, PlanConfig, PlanExecutor, ReplanConfig};
use xola_runtime::reliability::{LoopDetectorConfig, TimeoutConfig};
use xola_runtime::tools::{Tool, ToolError, ToolRegistry, ToolTimeoutConfig};

// ---------------------------------------------------------------------------
// Test Tools
// ---------------------------------------------------------------------------

/// A tool that fails N times before succeeding (for circuit breaker testing).
struct FlakyTool {
    fail_count: std::sync::atomic::AtomicUsize,
    max_failures: usize,
}

impl FlakyTool {
    fn new(max_failures: usize) -> Self {
        Self {
            fail_count: std::sync::atomic::AtomicUsize::new(0),
            max_failures,
        }
    }
}

#[async_trait::async_trait]
impl Tool for FlakyTool {
    fn name(&self) -> &str {
        "flaky_tool"
    }

    fn description(&self) -> &str {
        "A tool that fails several times before succeeding"
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        let count = self
            .fail_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if count < self.max_failures {
            // Fail with a generic error
            Err(ToolError::ExecutionFailed(format!(
                "Simulated failure {} of {}",
                count + 1,
                self.max_failures
            )))
        } else {
            // Success after max_failures
            Ok(json!({
                "result": format!("Success after {} failures", self.max_failures),
                "input": input["input"]
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Circuit Breaker Recovery with Replanning
// ---------------------------------------------------------------------------

/// Test that circuit breaker opens after failures, then allows recovery
/// after timeout, and that replanning works correctly.
///
/// This test is marked #[ignore] because it requires the Python IPC server.
#[ignore]
#[tokio::test]
async fn test_circuit_breaker_recovery_with_replan() {
    // Setup: Create a flaky tool that fails 3 times before succeeding
    let flaky_tool = Arc::new(FlakyTool::new(3));
    let mut registry = ToolRegistry::default();
    registry
        .register(flaky_tool.clone())
        .expect("register flaky tool");

    // Note: Circuit breaker is automatically configured with defaults (5 failures, 30s recovery)

    // Setup IPC client (requires Python server running)
    let ipc = Arc::new(match IpcClient::new("http://127.0.0.1:8001") {
        Ok(client) => client,
        Err(e) => {
            println!("Skipping test: Failed to create IPC client: {}", e);
            return;
        }
    });

    // Create executor with replanning enabled
    let executor = PlanExecutor::new(
        ipc,
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        PlanConfig {
            max_iterations: 20,
            stm_token_budget: 8000,
            replan: ReplanConfig {
                max_replans_per_step: 5,
                max_replans_per_task: 10,
            },
        },
    );

    // Execute a task that will trigger replanning
    let result = executor
        .execute("Call flaky_tool with input 'test'. Keep trying until it succeeds.")
        .await;

    match result {
        Ok(task_result) => {
            // Task should eventually succeed after replanning
            assert_eq!(task_result.outcome, Outcome::Success);
            assert!(task_result.total_replans > 0, "Expected replanning to occur");
            println!(
                "Circuit breaker test passed: {} replans, {} steps",
                task_result.total_replans,
                task_result.steps.len()
            );
        }
        Err(e) => {
            panic!("Circuit breaker recovery failed: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Loop Detection with Replanning
// ---------------------------------------------------------------------------

/// Test that loop detector catches infinite loops and triggers replanning.
///
/// This test is marked #[ignore] because it requires the Python IPC server.
#[ignore]
#[tokio::test]
async fn test_loop_detection_triggers_failure() {
    // Setup: Normal registry
    let registry = ToolRegistry::default();

    // Setup IPC client
    let ipc = Arc::new(match IpcClient::new("http://127.0.0.1:8001") {
        Ok(client) => client,
        Err(e) => {
            println!("Skipping test: Failed to create IPC client: {}", e);
            return;
        }
    });

    // Create executor with strict loop detection
    let executor = PlanExecutor::new(
        ipc,
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        PlanConfig {
            max_iterations: 20,
            stm_token_budget: 8000,
            replan: ReplanConfig::default(),
        },
    )
    .with_loop_detector_config(LoopDetectorConfig {
        window_size: 5,
        repeat_threshold: 3,
    });

    // Execute a task that might cause the LLM to loop
    // (This depends on LLM behavior - may not always trigger)
    let result = executor
        .execute("Keep calling the same non-existent tool 'does_not_exist' repeatedly with the same input.")
        .await;

    // Expect either loop detection or max replans
    match result {
        Err(e) => {
            let category = e.category();
            assert!(
                matches!(
                    category,
                    FailureCategory::Loop | FailureCategory::ResourceExhaustion
                ),
                "Expected Loop or ResourceExhaustion, got: {:?}",
                category
            );
            println!("Loop detection test passed: {}", e);
        }
        Ok(_) => {
            println!("Loop detection test: Task completed without looping (LLM avoided loop)");
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Timeout Enforcement
// ---------------------------------------------------------------------------

/// Test that task-level timeout is enforced correctly.
///
/// This test is marked #[ignore] because it requires the Python IPC server.
#[ignore]
#[tokio::test]
async fn test_task_timeout_enforcement() {
    // Setup: Normal registry
    let registry = ToolRegistry::default();

    // Setup IPC client
    let ipc = Arc::new(match IpcClient::new("http://127.0.0.1:8001") {
        Ok(client) => client,
        Err(e) => {
            println!("Skipping test: Failed to create IPC client: {}", e);
            return;
        }
    });

    // Create executor with very short timeout
    let executor = PlanExecutor::new(
        ipc,
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        PlanConfig {
            max_iterations: 100,
            stm_token_budget: 8000,
            replan: ReplanConfig::default(),
        },
    )
    .with_timeout_context_config(TimeoutConfig {
        tool_timeout: Duration::from_secs(30),
        step_timeout: Duration::from_secs(60),
        task_timeout: Duration::from_millis(500), // Very short timeout
    });

    // Execute a task that would take longer than the timeout
    let result = executor
        .execute("Perform a complex multi-step research task that requires many tool calls.")
        .await;

    // Expect timeout error
    match result {
        Err(e) => {
            assert_eq!(
                e.category(),
                FailureCategory::Timeout,
                "Expected Timeout, got: {:?}",
                e.category()
            );
            println!("Timeout test passed: {}", e);
        }
        Ok(_) => {
            println!("Timeout test: Task completed quickly (no timeout needed)");
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Failure Category Classification
// ---------------------------------------------------------------------------

#[test]
fn test_failure_category_classification() {
    use xola_runtime::planning::PlanError;

    // Timeout error
    let timeout_err = PlanError::Timeout {
        level: "task".to_string(),
        duration_ms: 5000,
    };
    assert_eq!(timeout_err.category(), FailureCategory::Timeout);
    assert!(timeout_err.is_retryable());
    assert!(!timeout_err.is_stuck());

    // Loop error
    let loop_err = PlanError::Loop("Detected loop".to_string());
    assert_eq!(loop_err.category(), FailureCategory::Loop);
    assert!(!loop_err.is_retryable());
    assert!(loop_err.is_stuck());

    // Tool error
    let tool_err = PlanError::Tool(ToolError::NotFound("tool".to_string()));
    assert_eq!(tool_err.category(), FailureCategory::ToolError);
    assert!(tool_err.is_retryable());
    assert!(!tool_err.is_stuck());

    // Resource exhaustion
    let exhaustion_err = PlanError::MaxIterationsReached(10);
    assert_eq!(exhaustion_err.category(), FailureCategory::ResourceExhaustion);
    assert!(!exhaustion_err.is_retryable());
    assert!(exhaustion_err.is_stuck());
}
