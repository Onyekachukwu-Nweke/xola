//! Tool trait and error types for the Xola agent runtime.
//!
//! This module defines the core abstraction for tools - capabilities the agent
//! can invoke to interact with external systems, execute code, or retrieve data.
//!
//! Every tool must implement the `Tool` trait, which provides:
//! - A unique name and description for LLM-facing selection
//! - A JSON Schema defining expected input format
//! - An async execute function that performs the work
//!
//! Related tasks: L1-01 through L1-09 in AGENTS.md

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

pub mod mock;
pub mod registry;
pub mod validator;

// Re-export registry types for convenient access
#[allow(unused_imports)] // RegistryError not yet used - will be used in L1-04+
pub use registry::{RegistryError, ToolRegistry};
#[allow(unused_imports)] // InputValidator not yet used directly - used via ToolRegistry
pub use validator::InputValidator;

/// Errors that can occur during tool execution.
///
/// This enum covers validation failures, execution errors, and timeouts.
/// Every variant includes context to help the LLM understand what went wrong
/// and potentially adjust its next action.
#[derive(Error, Debug)]
#[allow(dead_code)] // Some variants not yet used - will be used in L1-02+
pub enum ToolError {
    /// Input failed JSON Schema validation.
    ///
    /// Contains the validation error message so the LLM can correct its input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Tool execution failed.
    ///
    /// This is the catch-all for tool-specific errors: network failures,
    /// API errors, Docker container failures, etc.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Tool execution exceeded its timeout.
    ///
    /// The timeout duration is set per-tool in the dispatcher (L1-04).
    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// JSON serialization/deserialization error.
    ///
    /// Occurs when tool output cannot be serialized to JSON.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Tool not found in registry.
    ///
    /// The LLM requested a tool that doesn't exist.
    #[error("Tool '{0}' not found in registry")]
    NotFound(String),
}

/// A tool the agent can execute.
///
/// Tools are the agent's capabilities - functions it can call to interact
/// with external systems. Each tool has a name, description, input schema,
/// and an async execute function.
///
/// # Thread Safety
///
/// Tools must be `Send + Sync` because they're stored in `Arc<dyn Tool>`
/// and dispatched across tokio tasks.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::{Tool, ToolError};
/// use serde_json::{json, Value};
///
/// struct WebSearchTool;
///
/// impl Tool for WebSearchTool {
///     fn name(&self) -> &str {
///         "web_search"
///     }
///
///     fn description(&self) -> &str {
///         "Search the web for information"
///     }
///
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "query": { "type": "string" },
///                 "num_results": { "type": "integer", "default": 5 }
///             },
///             "required": ["query"]
///         })
///     }
///
///     async fn execute(&self, input: Value) -> Result<Value, ToolError> {
///         // Implementation here
///         todo!()
///     }
/// }
/// ```
#[async_trait]
#[allow(dead_code)] // input_schema not yet used - will be used in L1-03 (validation)
pub trait Tool: Send + Sync {
    /// Unique identifier for this tool.
    ///
    /// Used by the LLM to select which tool to call. Must be lowercase,
    /// snake_case, and unique within the registry.
    ///
    /// Examples: `"web_search"`, `"url_fetch"`, `"code_exec"`
    fn name(&self) -> &str;

    /// Human-readable description of what this tool does.
    ///
    /// Included in the LLM's prompt to help it select the right tool.
    /// Should be concise (1-2 sentences) and focus on the tool's purpose.
    ///
    /// Example: "Fetches the text content of a URL"
    fn description(&self) -> &str;

    /// JSON Schema defining the expected input format.
    ///
    /// This schema is validated before execution (L1-03). The LLM sees this
    /// schema in its prompt and attempts to produce matching JSON.
    ///
    /// Should be a valid JSON Schema object with `type: "object"`,
    /// `properties`, and optionally `required`.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "url": { "type": "string" }
    ///   },
    ///   "required": ["url"]
    /// }
    /// ```
    fn input_schema(&self) -> Value;

    /// Execute the tool with validated input.
    ///
    /// This is called after input validation succeeds. The implementer can
    /// assume the input matches the schema.
    ///
    /// # Errors
    ///
    /// Return `ToolError::ExecutionFailed` for any tool-specific failures.
    /// The error message should be clear enough for the LLM to understand
    /// what went wrong.
    ///
    /// # Timeout
    ///
    /// Implementations should not handle timeouts themselves - the dispatcher
    /// (L1-04) wraps all tool calls with `tokio::time::timeout`.
    ///
    /// # Return Value
    ///
    /// Must return a JSON-serializable value. Typically a JSON object with
    /// multiple fields describing the result.
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_error_display() {
        let err = ToolError::InvalidInput("missing required field 'query'".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid input: missing required field 'query'"
        );

        let err = ToolError::Timeout { timeout_ms: 5000 };
        assert_eq!(err.to_string(), "Timeout after 5000ms");

        let err = ToolError::NotFound("nonexistent_tool".to_string());
        assert_eq!(
            err.to_string(),
            "Tool 'nonexistent_tool' not found in registry"
        );
    }

    #[test]
    fn test_tool_error_from_json_error() {
        let json_err = serde_json::from_str::<Value>("{invalid json}").unwrap_err();
        let tool_err: ToolError = json_err.into();
        assert!(matches!(tool_err, ToolError::JsonError(_)));
    }
}
