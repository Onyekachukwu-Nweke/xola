//! Error types for the planning and orchestration layer.

use crate::ipc::IpcError;
use crate::tools::ToolError;
use thiserror::Error;

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
}
