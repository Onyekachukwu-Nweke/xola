//! Integration tests for L1-09: End-to-end tool execution
//!
//! These tests verify that all tools work correctly when called through
//! the ToolRegistry's execute_with_timeout method. They exercise the full
//! execution flow: validation → lookup → timeout-wrapped execution → tracing.
//!
//! All tests are marked with #[ignore] because they require external dependencies:
//! - MockTool: No dependencies (but marked for consistency)
//! - UrlFetchTool: Requires internet connection
//! - WebSearchTool: Requires SERPER_API_KEY environment variable
//! - CodeExecTool: Requires Docker daemon with pre-pulled images
//!
//! Run with: cargo test -- --ignored

use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use xola_runtime::tools::{
    mock::MockTool, CodeExecTool, ToolRegistry, UrlFetchTool, WebSearchTool,
};

/// Helper function to create a registry with all tools registered.
fn create_registry_with_all_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("Failed to register MockTool");
    registry
        .register(Arc::new(UrlFetchTool))
        .expect("Failed to register UrlFetchTool");
    registry
        .register(Arc::new(WebSearchTool))
        .expect("Failed to register WebSearchTool");
    registry
        .register(Arc::new(CodeExecTool))
        .expect("Failed to register CodeExecTool");
    registry
}

// ---------------------------------------------------------------------------
// MockTool Integration Tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Marked for consistency with other integration tests
async fn test_mock_tool_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "message": "integration test" });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("mock_echo", input, timeout)
        .await
        .expect("MockTool execution failed");

    assert_eq!(result["echoed"], "integration test");
    assert!(result["timestamp"].is_number());
}

#[tokio::test]
#[ignore]
async fn test_mock_tool_validation_through_registry() {
    let registry = create_registry_with_all_tools();

    // Invalid input (missing required field)
    let input = json!({ "wrong": "field" });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("mock_echo", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid input"));
}

// ---------------------------------------------------------------------------
// UrlFetchTool Integration Tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_url_fetch_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "url": "https://example.com" });
    let timeout = Duration::from_secs(30);

    let result = registry
        .execute_with_timeout("url_fetch", input, timeout)
        .await
        .expect("UrlFetchTool execution failed");

    assert_eq!(result["status_code"], 200);
    assert!(result["content"].is_string());
    assert!(result["content_length"].is_number());

    let content = result["content"].as_str().unwrap();
    assert!(content.contains("Example Domain"));
}

#[tokio::test]
#[ignore]
async fn test_url_fetch_404_error() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "url": "https://httpbin.org/status/404" });
    let timeout = Duration::from_secs(30);

    let result = registry
        .execute_with_timeout("url_fetch", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("404"));
}

#[tokio::test]
#[ignore]
async fn test_url_fetch_validation_missing_url() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "wrong_field": "value" });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("url_fetch", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid input"));
}

// ---------------------------------------------------------------------------
// WebSearchTool Integration Tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_web_search_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "query": "Rust programming language",
        "num_results": 3
    });
    let timeout = Duration::from_secs(15);

    let result = registry
        .execute_with_timeout("web_search", input, timeout)
        .await
        .expect("WebSearchTool execution failed");

    assert!(result["results"].is_array());
    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 3);

    // Verify result structure
    for result_item in results {
        assert!(result_item["title"].is_string());
        assert!(result_item["url"].is_string());
        assert!(result_item["snippet"].is_string());
    }
}

#[tokio::test]
#[ignore]
async fn test_web_search_num_results_default() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "query": "test query" });
    let timeout = Duration::from_secs(15);

    let result = registry
        .execute_with_timeout("web_search", input, timeout)
        .await
        .expect("WebSearchTool execution failed");

    assert!(result["results"].is_array());
    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty());
    // Default is 5 results, but API may return fewer
    assert!(results.len() <= 5);
}

#[tokio::test]
#[ignore]
async fn test_web_search_validation_missing_query() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "num_results": 5 });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("web_search", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid input"));
}

// ---------------------------------------------------------------------------
// CodeExecTool Integration Tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_code_exec_python_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "python",
        "code": "print('Hello from Python')",
        "timeout_seconds": 5
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await
        .expect("CodeExecTool execution failed");

    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"]
        .as_str()
        .unwrap()
        .contains("Hello from Python"));
    assert_eq!(result["stderr"].as_str().unwrap(), "");
}

#[tokio::test]
#[ignore]
async fn test_code_exec_javascript_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "javascript",
        "code": "console.log('Hello from JavaScript')",
        "timeout_seconds": 5
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await
        .expect("CodeExecTool execution failed");

    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"]
        .as_str()
        .unwrap()
        .contains("Hello from JavaScript"));
}

#[tokio::test]
#[ignore]
async fn test_code_exec_bash_end_to_end() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "bash",
        "code": "echo 'Hello from Bash'",
        "timeout_seconds": 5
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await
        .expect("CodeExecTool execution failed");

    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"]
        .as_str()
        .unwrap()
        .contains("Hello from Bash"));
}

#[tokio::test]
#[ignore]
async fn test_code_exec_stderr_captured() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "python",
        "code": "import sys; sys.stderr.write('error output')",
        "timeout_seconds": 5
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await
        .expect("CodeExecTool execution failed");

    assert!(result["stderr"].as_str().unwrap().contains("error output"));
}

#[tokio::test]
#[ignore]
async fn test_code_exec_nonzero_exit_code() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "python",
        "code": "import sys; sys.exit(42)",
        "timeout_seconds": 5
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await
        .expect("CodeExecTool execution failed");

    assert_eq!(result["exit_code"], 42);
}

#[tokio::test]
#[ignore]
async fn test_code_exec_timeout() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "python",
        "code": "import time; time.sleep(60)",
        "timeout_seconds": 2
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("timeout") || err.to_string().contains("exceeded"));
}

#[tokio::test]
#[ignore]
async fn test_code_exec_oom() {
    let registry = create_registry_with_all_tools();

    let input = json!({
        "language": "python",
        "code": "x = [0] * (200 * 1024 * 1024)"  // Try to allocate 200 MB
    });
    let timeout = Duration::from_secs(10);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().to_lowercase().contains("memory") || err.to_string().contains("OOM"));
}

#[tokio::test]
#[ignore]
async fn test_code_exec_validation_missing_language() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "code": "print('test')" });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("code_exec", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid input"));
}

// ---------------------------------------------------------------------------
// Cross-Tool Tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_all_tools_registered() {
    let registry = create_registry_with_all_tools();

    let tool_names = registry.list_names();
    assert_eq!(tool_names.len(), 4);
    assert!(tool_names.contains(&"mock_echo".to_string()));
    assert!(tool_names.contains(&"url_fetch".to_string()));
    assert!(tool_names.contains(&"web_search".to_string()));
    assert!(tool_names.contains(&"code_exec".to_string()));
}

#[tokio::test]
#[ignore]
async fn test_tool_not_found_error() {
    let registry = create_registry_with_all_tools();

    let input = json!({ "message": "test" });
    let timeout = Duration::from_secs(5);

    let result = registry
        .execute_with_timeout("nonexistent_tool", input, timeout)
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"));
    assert!(err.to_string().contains("nonexistent_tool"));
}
