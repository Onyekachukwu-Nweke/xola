//! Hierarchical timeout management for agent execution.
//!
//! Every execution level has a timeout, enforced through a `CancellationToken` tree:
//!
//! ```text
//! Task Timeout (e.g., 5 minutes)
//! ├── Step Timeout (e.g., 60 seconds)
//! │   ├── Tool Call Timeout (e.g., 30 seconds)
//! │   ├── Tool Call Timeout
//! │   └── ...
//! ├── Step Timeout
//! │   ├── Tool Call Timeout
//! │   └── ...
//! └── ...
//! ```
//!
//! When a parent timeout fires, all children are automatically cancelled.
//!
//! Related task: L4-06 in AGENTS.md

use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Configuration for hierarchical timeouts.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Timeout for individual tool execution.
    pub tool_timeout: Duration,
    /// Timeout for a single planning step (reason + tool call + observe).
    pub step_timeout: Duration,
    /// Timeout for the entire task from start to final answer.
    pub task_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            tool_timeout: Duration::from_secs(30),
            step_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Hierarchical timeout context for agent execution.
///
/// Maintains a tree of cancellation tokens where parent cancellation
/// automatically cancels all children.
///
/// # Thread Safety
///
/// All CancellationTokens are thread-safe and can be shared across tasks.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::reliability::{TimeoutConfig, TimeoutContext};
/// use std::time::Duration;
///
/// #[tokio::main]
/// async fn main() {
///     let config = TimeoutConfig::default();
///     let ctx = TimeoutContext::new_task(&config);
///
///     // Execute with task-level timeout
///     let task_deadline = tokio::time::sleep(config.task_timeout);
///
///     tokio::select! {
///         result = execute_plan(&ctx) => {
///             println!("Plan completed: {:?}", result);
///         }
///         _ = task_deadline => {
///             ctx.cancel_task();
///             println!("Task timeout exceeded");
///         }
///     }
/// }
/// ```
pub struct TimeoutContext {
    /// Root cancellation token for the entire task
    task_token: CancellationToken,
    /// Timeout configuration
    config: TimeoutConfig,
}

impl TimeoutContext {
    /// Creates a new timeout context for a task.
    ///
    /// # Parameters
    ///
    /// * `config` - Timeout configuration
    ///
    /// # Returns
    ///
    /// A new timeout context with a root task token
    pub fn new_task(config: TimeoutConfig) -> Self {
        Self {
            task_token: CancellationToken::new(),
            config,
        }
    }

    /// Returns the root task cancellation token.
    ///
    /// This token should be monitored at the task level. When cancelled,
    /// all child tokens (steps, tools) are automatically cancelled.
    pub fn task_token(&self) -> &CancellationToken {
        &self.task_token
    }

    /// Creates a child token for a planning step.
    ///
    /// The step token is a child of the task token. If the task is cancelled,
    /// the step is automatically cancelled.
    ///
    /// # Returns
    ///
    /// A cancellation token for this step
    pub fn new_step_token(&self) -> CancellationToken {
        self.task_token.child_token()
    }

    /// Creates a child token for a tool call within a step.
    ///
    /// The tool token is a child of the step token. If either the task or
    /// step is cancelled, the tool call is automatically cancelled.
    ///
    /// # Parameters
    ///
    /// * `step_token` - The step's cancellation token
    ///
    /// # Returns
    ///
    /// A cancellation token for this tool call
    pub fn new_tool_token(&self, step_token: &CancellationToken) -> CancellationToken {
        step_token.child_token()
    }

    /// Cancels the entire task, which cascades to all steps and tools.
    pub fn cancel_task(&self) {
        self.task_token.cancel();
    }

    /// Returns the configured tool timeout.
    pub fn tool_timeout(&self) -> Duration {
        self.config.tool_timeout
    }

    /// Returns the configured step timeout.
    pub fn step_timeout(&self) -> Duration {
        self.config.step_timeout
    }

    /// Returns the configured task timeout.
    pub fn task_timeout(&self) -> Duration {
        self.config.task_timeout
    }

    /// Returns a reference to the timeout configuration.
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = TimeoutConfig::default();
        assert_eq!(config.tool_timeout, Duration::from_secs(30));
        assert_eq!(config.step_timeout, Duration::from_secs(60));
        assert_eq!(config.task_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_new_task_context() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        assert!(!ctx.task_token().is_cancelled());
        assert_eq!(ctx.tool_timeout(), Duration::from_secs(30));
        assert_eq!(ctx.step_timeout(), Duration::from_secs(60));
        assert_eq!(ctx.task_timeout(), Duration::from_secs(300));
    }

    #[test]
    fn test_task_cancellation_propagates() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        let step_token = ctx.new_step_token();
        let tool_token = ctx.new_tool_token(&step_token);

        // Initially none are cancelled
        assert!(!ctx.task_token().is_cancelled());
        assert!(!step_token.is_cancelled());
        assert!(!tool_token.is_cancelled());

        // Cancel task - should cascade
        ctx.cancel_task();

        assert!(ctx.task_token().is_cancelled());
        assert!(step_token.is_cancelled());
        assert!(tool_token.is_cancelled());
    }

    #[test]
    fn test_step_cancellation_propagates_to_tool_only() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        let step_token = ctx.new_step_token();
        let tool_token = ctx.new_tool_token(&step_token);

        // Cancel step - should only affect tool, not task
        step_token.cancel();

        assert!(!ctx.task_token().is_cancelled());
        assert!(step_token.is_cancelled());
        assert!(tool_token.is_cancelled());
    }

    #[test]
    fn test_tool_cancellation_doesnt_affect_parents() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        let step_token = ctx.new_step_token();
        let tool_token = ctx.new_tool_token(&step_token);

        // Cancel tool - should only affect tool
        tool_token.cancel();

        assert!(!ctx.task_token().is_cancelled());
        assert!(!step_token.is_cancelled());
        assert!(tool_token.is_cancelled());
    }

    #[test]
    fn test_multiple_steps_independent() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        let step1_token = ctx.new_step_token();
        let step2_token = ctx.new_step_token();

        // Cancel step1 - should not affect step2
        step1_token.cancel();

        assert!(!ctx.task_token().is_cancelled());
        assert!(step1_token.is_cancelled());
        assert!(!step2_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_timeout_with_select() {
        let config = TimeoutConfig {
            tool_timeout: Duration::from_millis(50),
            step_timeout: Duration::from_millis(100),
            task_timeout: Duration::from_millis(200),
        };
        let step_timeout = config.step_timeout;
        let ctx = TimeoutContext::new_task(config);

        let step_token = ctx.new_step_token();

        // Simulate a long-running operation
        let work = async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            "completed"
        };

        // Use select to race work against timeout
        let result = tokio::select! {
            output = work => Ok(output),
            _ = tokio::time::sleep(step_timeout) => {
                step_token.cancel();
                Err("timeout")
            }
        };

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "timeout");
        assert!(step_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancellation_token_in_select() {
        let config = TimeoutConfig::default();
        let ctx = TimeoutContext::new_task(config);

        let step_token = ctx.new_step_token();
        let step_token_clone = step_token.clone();

        // Spawn a task that will cancel the step
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            step_token_clone.cancel();
        });

        // Wait for cancellation
        let result = tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(10)) => "timeout",
            _ = step_token.cancelled() => "cancelled",
        };

        assert_eq!(result, "cancelled");
    }

    #[test]
    fn test_custom_config() {
        let config = TimeoutConfig {
            tool_timeout: Duration::from_secs(10),
            step_timeout: Duration::from_secs(30),
            task_timeout: Duration::from_secs(120),
        };

        let ctx = TimeoutContext::new_task(config);

        assert_eq!(ctx.tool_timeout(), Duration::from_secs(10));
        assert_eq!(ctx.step_timeout(), Duration::from_secs(30));
        assert_eq!(ctx.task_timeout(), Duration::from_secs(120));
    }
}
