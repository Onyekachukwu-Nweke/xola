//! URL fetching tool using reqwest HTTP client.
//!
//! Fetches the text content of a URL and returns it along with metadata
//! (status code, content length). Used for web content retrieval, documentation
//! lookup, and information gathering tasks.

use super::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Fetches the text content of a URL using reqwest.
///
/// Makes an HTTP GET request to the specified URL and returns the response
/// body as a string along with metadata (status code, content length).
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "url": { "type": "string", "description": "The URL to fetch" }
///   },
///   "required": ["url"]
/// }
/// ```
///
/// # Output Format
///
/// ```json
/// {
///   "content": "the response body as text",
///   "status_code": 200,
///   "content_length": 12345
/// }
/// ```
///
/// # Error Handling
///
/// Returns `ToolError::ExecutionFailed` for:
/// - Network errors (connection refused, DNS failure, etc.)
/// - Non-2xx HTTP status codes
/// - Response body decoding errors (non-UTF8 content)
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::{UrlFetchTool, Tool};
/// use serde_json::json;
///
/// let tool = UrlFetchTool;
/// let input = json!({ "url": "https://example.com" });
/// let result = tool.execute(input).await?;
///
/// assert_eq!(result["status_code"], 200);
/// assert!(result["content"].is_string());
/// ```
pub struct UrlFetchTool;

#[async_trait]
impl Tool for UrlFetchTool {
    fn name(&self) -> &str {
        "url_fetch"
    }

    fn description(&self) -> &str {
        "Fetches the text content of a URL"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must include http:// or https://)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // Extract and validate URL
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing or invalid 'url' field".to_string()))?;

        // Make HTTP GET request
        let response = reqwest::get(url)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        // Check status code
        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "HTTP request failed with status {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Get content length before consuming body
        let content_length = response.content_length().unwrap_or(0);

        // Read response body as text
        let content = response.text().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read response body: {}", e))
        })?;

        // Return structured output
        Ok(json!({
            "content": content,
            "status_code": status.as_u16(),
            "content_length": content_length
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_url_fetch_basic() {
        let tool = UrlFetchTool;

        // Verify metadata
        assert_eq!(tool.name(), "url_fetch");
        assert!(tool.description().contains("Fetches"));

        // Verify schema
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        assert_eq!(schema["required"][0], "url");
    }

    #[tokio::test]
    async fn test_url_fetch_missing_url() {
        let tool = UrlFetchTool;
        let input = json!({});

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("url"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_url_fetch_wrong_type() {
        let tool = UrlFetchTool;
        let input = json!({ "url": 42 });

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    // Integration test - marked with #[ignore] to avoid network calls in CI
    #[tokio::test]
    #[ignore]
    async fn test_url_fetch_real_request() {
        let tool = UrlFetchTool;
        let input = json!({ "url": "https://example.com" });

        let result = tool.execute(input).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output["content"].is_string());
        assert_eq!(output["status_code"], 200);
        assert!(output["content_length"].as_u64().unwrap() > 0);
    }

    // Test non-2xx status handling
    #[tokio::test]
    #[ignore]
    async fn test_url_fetch_404() {
        let tool = UrlFetchTool;
        let input = json!({ "url": "https://httpbin.org/status/404" });

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed(msg) => {
                assert!(msg.contains("404"));
            }
            _ => panic!("Expected ExecutionFailed error"),
        }
    }
}
