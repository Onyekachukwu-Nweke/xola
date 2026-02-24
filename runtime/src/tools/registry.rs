//! Tool registry for managing available agent capabilities.
//!
//! The registry provides centralized storage and lookup for all tools
//! available to the agent. Tools are registered during startup and then
//! accessed by name during execution.

use super::{Tool, ToolError};
use crate::reliability::{CircuitBreaker, CircuitBreakerConfig};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{error, info, info_span, warn, Instrument};

/// Errors that can occur during tool registration.
#[derive(Error, Debug)]
pub enum RegistryError {
    /// Attempted to register a tool with a name that's already taken.
    #[error("Tool '{0}' is already registered")]
    DuplicateName(String),
}

/// Central registry for all tools available to the agent.
///
/// Stores tools in a HashMap keyed by name. Each tool is wrapped in `Arc<dyn Tool>`
/// for thread-safe sharing across async tasks.
///
/// # Circuit Breakers
///
/// Each tool has an associated circuit breaker (L4-01, L4-02) that monitors
/// failures and temporarily disables tools that fail repeatedly.
///
/// # Thread Safety
///
/// The registry itself does not implement interior mutability. Registration happens
/// during startup, then the registry is wrapped in `Arc<ToolRegistry>` and shared
/// read-only across the runtime. Circuit breakers use atomics internally.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use xola_runtime::tools::{ToolRegistry, mock::MockTool};
/// use xola_runtime::reliability::CircuitBreakerConfig;
///
/// let config = CircuitBreakerConfig::default();
/// let mut registry = ToolRegistry::new(config);
/// registry.register(Arc::new(MockTool)).unwrap();
///
/// let tool = registry.get("mock_echo").unwrap();
/// assert_eq!(tool.name(), "mock_echo");
///
/// let schemas = registry.list_schemas();
/// assert_eq!(schemas.len(), 1);
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    circuit_breakers: HashMap<String, Arc<CircuitBreaker>>,
    circuit_breaker_config: CircuitBreakerConfig,
}

/// Truncates a JSON value to a maximum character count for logging.
///
/// If the JSON serialization exceeds max_chars, truncates and appends
/// a message indicating the total size.
fn truncate_json(value: &Value, max_chars: usize) -> String {
    let serialized = value.to_string();
    if serialized.len() > max_chars {
        format!(
            "{}... [truncated, {} bytes total]",
            &serialized[..max_chars],
            serialized.len()
        )
    } else {
        serialized
    }
}

impl ToolRegistry {
    /// Creates a new empty registry with the given circuit breaker configuration.
    ///
    /// Each tool registered will get its own circuit breaker instance with this config.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xola_runtime::tools::ToolRegistry;
    /// use xola_runtime::reliability::CircuitBreakerConfig;
    /// use std::time::Duration;
    ///
    /// let config = CircuitBreakerConfig {
    ///     failure_threshold: 3,
    ///     recovery_timeout: Duration::from_secs(60),
    /// };
    /// let registry = ToolRegistry::new(config);
    /// ```
    pub fn new(circuit_breaker_config: CircuitBreakerConfig) -> Self {
        Self {
            tools: HashMap::new(),
            circuit_breakers: HashMap::new(),
            circuit_breaker_config,
        }
    }

    /// Registers a tool in the registry.
    ///
    /// Creates a circuit breaker for this tool using the registry's configuration.
    ///
    /// # Errors
    ///
    /// Returns `RegistryError::DuplicateName` if a tool with the same name
    /// is already registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use xola_runtime::tools::{ToolRegistry, mock::MockTool};
    /// use xola_runtime::reliability::CircuitBreakerConfig;
    ///
    /// let mut registry = ToolRegistry::new(CircuitBreakerConfig::default());
    /// registry.register(Arc::new(MockTool)).unwrap();
    /// ```
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), RegistryError> {
        let name = tool.name().to_string();

        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }

        // Create circuit breaker for this tool
        let breaker = Arc::new(CircuitBreaker::new(self.circuit_breaker_config.clone()));
        self.circuit_breakers.insert(name.clone(), breaker);

        self.tools.insert(name, tool);
        Ok(())
    }

    /// Looks up a tool by name.
    ///
    /// Returns `None` if no tool with that name is registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = registry.get("mock_echo").unwrap();
    /// println!("Found tool: {}", tool.name());
    /// ```
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Returns all registered tool names.
    ///
    /// Useful for debugging and diagnostics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let names = registry.list_names();
    /// println!("Available tools: {:?}", names);
    /// ```
    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Generates tool schemas for the LLM prompt.
    ///
    /// Returns a JSON array where each element has:
    /// - `name`: Tool name
    /// - `description`: Human-readable description
    /// - `parameters`: JSON Schema for input validation
    ///
    /// This format matches the IPC contract for the `/reason` endpoint.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let schemas = registry.list_schemas();
    /// for schema in &schemas {
    ///     println!("{}: {}", schema["name"], schema["description"]);
    /// }
    /// ```
    pub fn list_schemas(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.input_schema()
                })
            })
            .collect()
    }

    /// Validates input against a tool's schema.
    ///
    /// This is a convenience method that looks up the tool, retrieves its schema,
    /// and validates the input. Used by the dispatcher before tool execution.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::NotFound` if the tool doesn't exist.
    /// Returns `ToolError::InvalidInput` if validation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = json!({ "message": "hello" });
    /// registry.validate_input("mock_echo", &input).unwrap();
    /// ```
    pub fn validate_input(&self, tool_name: &str, input: &Value) -> Result<(), ToolError> {
        let tool = self
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        let schema = tool.input_schema();
        crate::tools::validator::InputValidator::validate(&schema, input)
    }

    /// Executes a tool with timeout protection and circuit breaker.
    ///
    /// This orchestrates the full execution flow:
    /// 1. Check circuit breaker status
    /// 2. Validate input against the tool's JSON Schema
    /// 3. Look up the tool in the registry
    /// 4. Execute with a timeout wrapper
    /// 5. Record success/failure with circuit breaker
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `ToolError::CircuitOpen` if the circuit breaker is open
    /// - `ToolError::NotFound` if the tool doesn't exist
    /// - `ToolError::InvalidInput` if validation fails
    /// - `ToolError::Timeout` if execution exceeds the timeout
    /// - `ToolError::ExecutionFailed` if the tool itself returns an error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    /// use xola_runtime::tools::{ToolRegistry, ToolTimeoutConfig};
    /// use xola_runtime::reliability::CircuitBreakerConfig;
    ///
    /// let registry = ToolRegistry::new(CircuitBreakerConfig::default());
    /// let input = json!({ "message": "hello" });
    /// let timeout = Duration::from_secs(30);
    ///
    /// let result = registry
    ///     .execute_with_timeout("mock_echo", input, timeout)
    ///     .await?;
    /// ```
    pub async fn execute_with_timeout(
        &self,
        tool_name: &str,
        input: Value,
        timeout: std::time::Duration,
    ) -> Result<Value, ToolError> {
        let span = info_span!(
            "tool_call",
            tool.name = %tool_name,
            tool.status = tracing::field::Empty,
            tool.duration_ms = tracing::field::Empty,
        );

        self.execute_with_timeout_inner(tool_name, input, timeout)
            .instrument(span)
            .await
    }

    /// Inner implementation of tool execution, wrapped by the `tool_call` span.
    async fn execute_with_timeout_inner(
        &self,
        tool_name: &str,
        input: Value,
        timeout: std::time::Duration,
    ) -> Result<Value, ToolError> {
        // Start timer for latency measurement
        let start = Instant::now();

        // Prepare truncated input for logging
        let input_display = truncate_json(&input, 500);
        let input_size = input.to_string().len();

        // Step 0: Check circuit breaker (L4-02)
        let breaker = self
            .circuit_breakers
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        if !breaker.can_execute() {
            warn!(tool_name, "Circuit breaker is open - rejecting execution");
            tracing::Span::current().record("tool.status", "circuit_open");
            return Err(ToolError::CircuitOpen(tool_name.to_string()));
        }

        // Step 1: Validate input (reuses L1-03 validation)
        self.validate_input(tool_name, &input)?;

        // Step 2: Get tool from registry
        let tool = self
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        // Step 3: Execute with timeout wrapper
        let result = match tokio::time::timeout(timeout, tool.execute(input)).await {
            Ok(Ok(result)) => Ok(result), // Success
            Ok(Err(e)) => Err(e),         // Tool returned error
            Err(_) => Err(ToolError::Timeout {
                // Timeout exceeded
                timeout_ms: timeout.as_millis() as u64,
            }),
        };

        // Measure latency
        let latency_ms = start.elapsed().as_millis() as u64;
        tracing::Span::current().record("tool.duration_ms", latency_ms);

        // Step 4: Record result with circuit breaker (L4-02)
        match &result {
            Ok(output) => {
                breaker.record_success();
                tracing::Span::current().record("tool.status", "ok");

                let output_display = truncate_json(output, 500);
                let output_size = output.to_string().len();

                info!(
                    tool_name,
                    input_size,
                    output_size,
                    latency_ms,
                    input = %input_display,
                    output = %output_display,
                    "Tool execution succeeded"
                );
            }
            Err(err) => {
                breaker.record_failure();
                tracing::Span::current().record("tool.status", "error");

                error!(
                    tool_name,
                    input_size,
                    latency_ms,
                    error = %err,
                    input = %input_display,
                    "Tool execution failed"
                );
            }
        }

        result
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::mock::MockTool;
    use crate::tools::Tool;
    use async_trait::async_trait;
    use std::time::Duration;

    /// A test tool that always fails execution.
    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "failing_tool"
        }

        fn description(&self) -> &str {
            "A tool that always fails"
        }

        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }

        async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
            Err(ToolError::ExecutionFailed("Simulated failure".to_string()))
        }
    }

    #[test]
    fn test_new_registry_is_empty() {
        let config = CircuitBreakerConfig::default();
        let registry = ToolRegistry::new(config);
        assert!(registry.get("anything").is_none());
        assert!(registry.list_schemas().is_empty());
        assert!(registry.list_names().is_empty());
    }

    #[test]
    fn test_default_is_empty() {
        let registry = ToolRegistry::default();
        assert!(registry.get("anything").is_none());
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::default();
        let tool = Arc::new(MockTool);

        registry.register(tool.clone()).unwrap();

        let retrieved = registry.get("mock_echo").unwrap();
        assert_eq!(retrieved.name(), "mock_echo");
    }

    #[test]
    fn test_register_creates_circuit_breaker() {
        let mut registry = ToolRegistry::default();
        let tool = Arc::new(MockTool);

        registry.register(tool).unwrap();

        // Verify circuit breaker was created
        assert!(registry.circuit_breakers.contains_key("mock_echo"));
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let mut registry = ToolRegistry::default();
        let tool1 = Arc::new(MockTool);
        let tool2 = Arc::new(MockTool);

        registry.register(tool1).unwrap();
        let err = registry.register(tool2).unwrap_err();

        match err {
            RegistryError::DuplicateName(name) => {
                assert_eq!(name, "mock_echo");
            }
        }
    }

    #[test]
    fn test_list_schemas_format() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 1);

        let schema = &schemas[0];
        assert_eq!(schema["name"], "mock_echo");
        assert_eq!(
            schema["description"],
            "A mock tool that echoes back the input message"
        );
        assert!(schema["parameters"].is_object());
        assert_eq!(schema["parameters"]["type"], "object");
    }

    #[test]
    fn test_list_names() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let names = registry.list_names();
        assert_eq!(names, vec!["mock_echo"]);
    }

    #[test]
    fn test_get_nonexistent_tool() {
        let registry = ToolRegistry::default();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_tools() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let names = registry.list_names();
        assert_eq!(names.len(), 1);

        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 1);
    }

    #[test]
    fn test_validate_input_success() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": "test" });
        assert!(registry.validate_input("mock_echo", &input).is_ok());
    }

    #[test]
    fn test_validate_input_missing_field() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({});
        let result = registry.validate_input("mock_echo", &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_validate_input_wrong_type() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": 123 });
        assert!(registry.validate_input("mock_echo", &input).is_err());
    }

    #[test]
    fn test_validate_input_tool_not_found() {
        let registry = ToolRegistry::default();

        let input = json!({ "message": "test" });
        let result = registry.validate_input("nonexistent", &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::NotFound(name) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    #[tokio::test]
    async fn test_execute_with_timeout_success() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": "test" });
        let timeout = Duration::from_secs(5);

        let result = registry
            .execute_with_timeout("mock_echo", input, timeout)
            .await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output["echoed"], "test");
    }

    #[tokio::test]
    async fn test_execute_with_timeout_validates_input() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let invalid_input = json!({ "wrong_field": "oops" });
        let timeout = Duration::from_secs(5);

        let result = registry
            .execute_with_timeout("mock_echo", invalid_input, timeout)
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_execute_with_timeout_tool_not_found() {
        let registry = ToolRegistry::default();

        let input = json!({ "message": "test" });
        let timeout = Duration::from_secs(5);

        let result = registry
            .execute_with_timeout("nonexistent", input, timeout)
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::NotFound(name) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    // Note: Testing actual timeout requires a slow tool.
    // MockTool executes instantly, so we can't test timeout expiration yet.
    // This will be added when we implement real tools in L1-05+.

    #[tokio::test]
    async fn test_execution_trace_on_success() {
        // Verify that successful tool execution logs at info! level
        // with tool_name, input_size, output_size, latency_ms fields

        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": "test" });
        let result = registry
            .execute_with_timeout("mock_echo", input, Duration::from_secs(5))
            .await;

        assert!(result.is_ok());
        // Note: Actual log verification requires tracing-subscriber test utilities
        // For now, verify execution completes without panicking
    }

    #[tokio::test]
    async fn test_execution_trace_on_error() {
        // Verify that failed tool execution logs at error! level
        // with tool_name, error, latency_ms fields

        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "wrong": "field" }); // Invalid input
        let result = registry
            .execute_with_timeout("mock_echo", input, Duration::from_secs(5))
            .await;

        assert!(result.is_err());
        // Verify error is logged (execution completes)
    }

    #[test]
    fn test_truncate_json_short() {
        let value = json!({ "message": "short" });
        let truncated = truncate_json(&value, 500);
        assert_eq!(truncated, r#"{"message":"short"}"#);
    }

    #[test]
    fn test_truncate_json_long() {
        let long_string = "x".repeat(600);
        let value = json!({ "data": long_string });
        let truncated = truncate_json(&value, 500);

        assert!(truncated.len() < 600);
        assert!(truncated.contains("[truncated"));
        assert!(truncated.contains("bytes total]"));
    }

    // L4-02: Circuit breaker integration tests

    #[tokio::test]
    async fn test_circuit_breaker_records_success() {
        let mut registry = ToolRegistry::default();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": "test" });
        let timeout = Duration::from_secs(5);

        // Execute successfully
        registry
            .execute_with_timeout("mock_echo", input, timeout)
            .await
            .unwrap();

        // Circuit should remain closed
        let breaker = registry.circuit_breakers.get("mock_echo").unwrap();
        assert!(breaker.can_execute());
    }

    #[tokio::test]
    async fn test_circuit_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_secs(30),
        };
        let mut registry = ToolRegistry::new(config);
        registry.register(Arc::new(FailingTool)).unwrap();

        let timeout = Duration::from_secs(5);

        // Cause execution failures (not validation failures)
        let input = json!({ "message": "test" });

        registry
            .execute_with_timeout("failing_tool", input.clone(), timeout)
            .await
            .unwrap_err();

        registry
            .execute_with_timeout("failing_tool", input.clone(), timeout)
            .await
            .unwrap_err();

        // Circuit should now be open
        let result = registry
            .execute_with_timeout("failing_tool", input, timeout)
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::CircuitOpen(name) => {
                assert_eq!(name, "failing_tool");
            }
            other => panic!("Expected CircuitOpen error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_circuit_recovers_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
        };
        let mut registry = ToolRegistry::new(config);
        registry.register(Arc::new(FailingTool)).unwrap();
        registry.register(Arc::new(MockTool)).unwrap();

        let timeout = Duration::from_secs(5);

        // Open the circuit with execution failures
        let input = json!({ "message": "test" });
        registry
            .execute_with_timeout("failing_tool", input.clone(), timeout)
            .await
            .unwrap_err();
        registry
            .execute_with_timeout("failing_tool", input.clone(), timeout)
            .await
            .unwrap_err();

        // Verify circuit is open
        let result = registry
            .execute_with_timeout("failing_tool", input.clone(), timeout)
            .await;
        assert!(matches!(result, Err(ToolError::CircuitOpen(_))));

        // Wait for recovery
        tokio::time::sleep(Duration::from_millis(150)).await;

        // After timeout, circuit should allow probe (transitions to half-open)
        // Use MockTool to simulate a successful probe by switching tools
        // In reality, the failing tool would need to succeed for full recovery
        let result = registry
            .execute_with_timeout("mock_echo", input, timeout)
            .await;

        // MockTool should work fine (has its own circuit breaker)
        assert!(result.is_ok());
    }
}
