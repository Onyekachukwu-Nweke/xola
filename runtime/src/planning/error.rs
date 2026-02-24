//! Error types for the planning and orchestration layer.

use crate::ipc::IpcError;
use crate::tools::ToolError;
use thiserror::Error;

/// Failure taxonomy for structured error classification (L4-07).
///
/// This enum categorizes all planning errors into distinct failure modes,
/// enabling better observability, error handling, and recovery strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureCategory {
    /// Error related to timeouts at any level (task/step/tool).
    Timeout,
    /// Error from LLM output parsing or validation.
    ParseError,
    /// Error during tool execution (not found, validation, circuit open, etc.).
    ToolError,
    /// Agent stuck in an infinite loop.
    Loop,
    /// Maximum iteration or replan limits reached.
    ResourceExhaustion,
    /// Communication failure with the Python IPC server.
    IpcFailure,
    /// Internal error (task panic, join error).
    InternalError,
}

/// Errors that can occur during plan execution.
#[derive(Error, Debug)]
pub enum PlanError {
    /// The planning loop hit the maximum iteration count without completing.
    #[error("max iterations reached ({0}) without completing the task")]
    MaxIterationsReached(usize),

    /// Replanning attempts exhausted for a single step.
    #[error("max replans per step reached ({0}) for action '{1}'")]
    MaxReplansPerStep(usize, String),

    /// Replanning attempts exhausted for the entire task.
    #[error("max replans per task reached ({0})")]
    MaxReplansPerTask(usize),

    /// IPC call to the Python server failed.
    #[error("IPC error: {0}")]
    Ipc(#[from] IpcError),

    /// Tool execution failed and replanning was not attempted or exhausted.
    #[error("tool error: {0}")]
    Tool(#[from] ToolError),

    /// A parallel branch failed.
    #[error("parallel branch failed: {0}")]
    ParallelBranchFailed(String),

    /// A tokio JoinHandle error (task panicked).
    #[error("task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    /// Agent is stuck in a loop, calling the same tool repeatedly.
    #[error("loop detected: {0}")]
    Loop(String),

    /// Execution timeout at task, step, or tool level.
    #[error("{level} timeout after {duration_ms}ms")]
    Timeout {
        /// Level at which timeout occurred (task/step/tool)
        level: String,
        /// Duration in milliseconds before timeout
        duration_ms: u64,
    },
}

impl PlanError {
    /// Classifies this error into a failure category (L4-07).
    ///
    /// This enables structured error handling, observability, and recovery logic
    /// based on the failure mode rather than specific error types.
    ///
    /// # Returns
    ///
    /// The [`FailureCategory`] that best describes this error.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match error.category() {
    ///     FailureCategory::Timeout => {
    ///         // Log timeout metrics, potentially retry with longer timeout
    ///     }
    ///     FailureCategory::Loop => {
    ///         // Clear loop detector state, try different approach
    ///     }
    ///     FailureCategory::ToolError => {
    ///         // Circuit breaker may be open, wait for recovery
    ///     }
    ///     _ => {
    ///         // Generic error handling
    ///     }
    /// }
    /// ```
    pub fn category(&self) -> FailureCategory {
        match self {
            PlanError::Timeout { .. } => FailureCategory::Timeout,
            PlanError::Loop(_) => FailureCategory::Loop,
            PlanError::Tool(_) => FailureCategory::ToolError,
            PlanError::Ipc(_) => FailureCategory::IpcFailure,
            PlanError::MaxIterationsReached(_)
            | PlanError::MaxReplansPerStep(_, _)
            | PlanError::MaxReplansPerTask(_) => FailureCategory::ResourceExhaustion,
            PlanError::JoinError(_) | PlanError::ParallelBranchFailed(_) => {
                FailureCategory::InternalError
            }
        }
    }

    /// Returns true if this error is retryable.
    ///
    /// Retryable errors are transient failures that may succeed on retry,
    /// such as timeouts or temporary tool failures. Non-retryable errors
    /// indicate fundamental issues like loops or resource exhaustion.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.category(),
            FailureCategory::Timeout | FailureCategory::ToolError | FailureCategory::IpcFailure
        )
    }

    /// Returns true if this error indicates the agent is stuck.
    ///
    /// Stuck errors require external intervention (clearing state, changing
    /// approach) rather than simple retry.
    pub fn is_stuck(&self) -> bool {
        matches!(
            self.category(),
            FailureCategory::Loop | FailureCategory::ResourceExhaustion
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_category() {
        let error = PlanError::Timeout {
            level: "task".to_string(),
            duration_ms: 5000,
        };
        assert_eq!(error.category(), FailureCategory::Timeout);
        assert!(error.is_retryable());
        assert!(!error.is_stuck());
    }

    #[test]
    fn test_loop_category() {
        let error = PlanError::Loop("Detected loop after 3 calls".to_string());
        assert_eq!(error.category(), FailureCategory::Loop);
        assert!(!error.is_retryable());
        assert!(error.is_stuck());
    }

    #[test]
    fn test_tool_error_category() {
        let error = PlanError::Tool(ToolError::NotFound("missing_tool".to_string()));
        assert_eq!(error.category(), FailureCategory::ToolError);
        assert!(error.is_retryable());
        assert!(!error.is_stuck());
    }

    #[test]
    fn test_ipc_failure_category() {
        let error = PlanError::Ipc(IpcError::ServerError {
            status: 500,
            body: "Server error".to_string(),
        });
        assert_eq!(error.category(), FailureCategory::IpcFailure);
        assert!(error.is_retryable());
        assert!(!error.is_stuck());
    }

    #[test]
    fn test_resource_exhaustion_category() {
        let error = PlanError::MaxIterationsReached(10);
        assert_eq!(error.category(), FailureCategory::ResourceExhaustion);
        assert!(!error.is_retryable());
        assert!(error.is_stuck());

        let error = PlanError::MaxReplansPerStep(3, "tool".to_string());
        assert_eq!(error.category(), FailureCategory::ResourceExhaustion);
        assert!(!error.is_retryable());
        assert!(error.is_stuck());

        let error = PlanError::MaxReplansPerTask(5);
        assert_eq!(error.category(), FailureCategory::ResourceExhaustion);
        assert!(!error.is_retryable());
        assert!(error.is_stuck());
    }

    #[test]
    fn test_internal_error_category() {
        let error = PlanError::ParallelBranchFailed("Branch failed".to_string());
        assert_eq!(error.category(), FailureCategory::InternalError);
        assert!(!error.is_retryable());
        assert!(!error.is_stuck());
    }
}
