//! Mock tool for testing.
//!
//! Provides a simple tool implementation that echoes back its input,
//! used for testing the tool trait and registry without external dependencies.

use super::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};

/// A mock tool that echoes back its input.
///
/// Used for testing tool dispatch, validation, and tracing without
/// requiring external services.
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "message": { "type": "string" }
///   },
///   "required": ["message"]
/// }
/// ```
///
/// # Output Format
///
/// ```json
/// {
///   "echoed": "the input message",
///   "timestamp": 1234567890
/// }
/// ```
pub struct MockTool;

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        "mock_echo"
    }

    fn description(&self) -> &str {
        "A mock tool that echoes back the input message"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // Extract the message from input
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("missing or invalid 'message' field".to_string())
            })?;

        // Return echoed message with timestamp
        Ok(json!({
            "echoed": message,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_mock_tool_basic() {
        let tool = MockTool;

        // Verify metadata
        assert_eq!(tool.name(), "mock_echo");
        assert!(tool.description().contains("echo"));

        // Verify schema
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["message"].is_object());
        assert_eq!(schema["required"][0], "message");
    }

    #[tokio::test]
    async fn test_mock_tool_execute_success() {
        let tool = MockTool;
        let input = json!({ "message": "Hello, world!" });

        let result = tool.execute(input).await.unwrap();

        assert_eq!(result["echoed"], "Hello, world!");
        assert!(result["timestamp"].is_u64());
    }

    #[tokio::test]
    async fn test_mock_tool_missing_message() {
        let tool = MockTool;
        let input = json!({});

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("message"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_mock_tool_wrong_type() {
        let tool = MockTool;
        let input = json!({ "message": 42 });

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }
}
