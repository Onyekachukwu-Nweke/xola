//! Web search tool using the Serper API (Google Search).
//!
//! Performs web searches and returns structured results including title, URL,
//! and snippet for each result. Used for research tasks, information gathering,
//! and finding relevant documentation.

use super::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Searches the web using the Serper API.
///
/// Makes a POST request to the Serper API (Google Search) and returns structured
/// search results including title, URL, and snippet for each result.
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "query": { "type": "string", "description": "The search query" },
///     "num_results": { "type": "integer", "default": 5, "description": "Number of results to return (1-10)" }
///   },
///   "required": ["query"]
/// }
/// ```
///
/// # Output Format
///
/// ```json
/// {
///   "results": [
///     {
///       "title": "Page title",
///       "url": "https://example.com/page",
///       "snippet": "Brief description of the page content"
///     }
///   ]
/// }
/// ```
///
/// # Error Handling
///
/// Returns `ToolError::ExecutionFailed` for:
/// - Missing or invalid SERPER_API_KEY environment variable
/// - Network errors (connection refused, DNS failure, etc.)
/// - Non-2xx HTTP status codes (authentication failure, rate limit, etc.)
/// - Response body parsing errors
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::{WebSearchTool, Tool};
/// use serde_json::json;
///
/// std::env::set_var("SERPER_API_KEY", "your_api_key");
///
/// let tool = WebSearchTool;
/// let input = json!({ "query": "Rust programming language" });
/// let result = tool.execute(input).await?;
///
/// assert!(result["results"].is_array());
/// ```
pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Searches the web using Google Search API"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // Extract and validate query
        let query = input.get("query").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("missing or invalid 'query' field".to_string())
        })?;

        // Extract num_results with default, clamped to 1-10 range
        let num_results = input
            .get("num_results")
            .and_then(|v| v.as_i64())
            .unwrap_or(5)
            .clamp(1, 10) as u64;

        // Get API key from environment
        let api_key = std::env::var("SERPER_API_KEY").map_err(|_| {
            ToolError::ExecutionFailed("SERPER_API_KEY environment variable not set".to_string())
        })?;

        // Build request body
        let request_body = json!({
            "q": query,
            "num": num_results
        });

        // Make HTTP POST request to Serper API
        let client = reqwest::Client::new();
        let response = client
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        // Check status code
        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "Search API request failed with status {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Parse response JSON
        let response_json: Value = response
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to parse response: {}", e)))?;

        // Extract organic search results
        let organic_results = response_json
            .get("organic")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ToolError::ExecutionFailed("Missing 'organic' field in API response".to_string())
            })?;

        // Transform to simplified result format
        let results: Vec<Value> = organic_results
            .iter()
            .map(|result| {
                json!({
                    "title": result.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                    "url": result.get("link").and_then(|v| v.as_str()).unwrap_or(""),
                    "snippet": result.get("snippet").and_then(|v| v.as_str()).unwrap_or("")
                })
            })
            .collect();

        // Return structured output
        Ok(json!({
            "results": results
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_web_search_basic() {
        let tool = WebSearchTool;

        // Verify metadata
        assert_eq!(tool.name(), "web_search");
        assert!(tool.description().contains("Search"));

        // Verify schema
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["properties"]["num_results"].is_object());
        assert_eq!(schema["required"][0], "query");
    }

    #[tokio::test]
    async fn test_web_search_missing_query() {
        let tool = WebSearchTool;
        let input = json!({});

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("query"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_web_search_wrong_type() {
        let tool = WebSearchTool;
        let input = json!({ "query": 42 });

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_web_search_missing_api_key() {
        let tool = WebSearchTool;
        let input = json!({ "query": "test" });

        // Ensure API key is not set
        std::env::remove_var("SERPER_API_KEY");

        let result = tool.execute(input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed(msg) => {
                assert!(msg.contains("SERPER_API_KEY"));
            }
            _ => panic!("Expected ExecutionFailed error"),
        }
    }

    #[tokio::test]
    async fn test_web_search_num_results_default() {
        let tool = WebSearchTool;

        // Verify schema has default
        let schema = tool.input_schema();
        assert_eq!(schema["properties"]["num_results"]["default"], 5);
    }

    // Integration test - marked with #[ignore] to avoid API calls in CI
    #[tokio::test]
    #[ignore]
    async fn test_web_search_real_request() {
        let tool = WebSearchTool;
        let input = json!({ "query": "Rust programming language", "num_results": 3 });

        // This test requires SERPER_API_KEY to be set
        let result = tool.execute(input).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output["results"].is_array());
        let results = output["results"].as_array().unwrap();
        assert!(results.len() > 0);
        assert!(results[0]["title"].is_string());
        assert!(results[0]["url"].is_string());
        assert!(results[0]["snippet"].is_string());
    }
}
