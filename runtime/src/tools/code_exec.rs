//! Code execution tool using Docker containers as a sandbox.
//!
//! Runs untrusted code inside an ephemeral Docker container with:
//! - No network access
//! - 128 MB memory limit
//! - Read-only root filesystem
//! - Automatic container cleanup on every exit path
//!
//! Supports Python, JavaScript (Node.js), and Bash.
//! Timeout enforcement is handled by the registry dispatcher (L1-04).
//!
//! Related task: L1-07 in AGENTS.md

use super::{Tool, ToolError};
use async_trait::async_trait;
use bollard::container::LogOutput;
use bollard::models::{ContainerCreateBody, HostConfig};
use bollard::query_parameters::{
    CreateContainerOptions, LogsOptions, RemoveContainerOptions, StartContainerOptions,
    WaitContainerOptions,
};
use bollard::Docker;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tracing::{debug, warn};

/// Maximum timeout the tool will honour regardless of what the caller requests.
const MAX_TIMEOUT_SECONDS: i64 = 30;

/// Default timeout when the caller does not supply one.
const DEFAULT_TIMEOUT_SECONDS: i64 = 10;

/// Memory limit applied to every container (128 MiB).
const MEMORY_LIMIT_BYTES: i64 = 128 * 1024 * 1024;

/// Executes code in a sandboxed Docker container.
///
/// Each call creates a fresh container, runs the supplied code snippet,
/// collects stdout/stderr, waits for the process to exit, removes the
/// container, and returns the result.
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "language":        { "type": "string", "enum": ["python", "javascript", "bash"] },
///     "code":            { "type": "string", "description": "Source code to execute" },
///     "timeout_seconds": { "type": "integer", "minimum": 1, "maximum": 30, "default": 10 }
///   },
///   "required": ["language", "code"]
/// }
/// ```
///
/// # Output Format
///
/// ```json
/// { "stdout": "...", "stderr": "...", "exit_code": 0 }
/// ```
///
/// # Error Handling
///
/// Returns `ToolError::ExecutionFailed` for:
/// - Docker daemon unavailable or permissions error
/// - Unknown / unsupported language
/// - Container creation or start failure
///
/// # Security
///
/// Containers run with `NetworkDisabled`, a 128 MiB memory cap, and a
/// read-only root filesystem. No bind mounts giving access to host paths.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::{CodeExecTool, Tool};
/// use serde_json::json;
///
/// let tool = CodeExecTool;
/// let input = json!({ "language": "python", "code": "print('hello')" });
/// let result = tool.execute(input).await?;
///
/// assert_eq!(result["exit_code"], 0);
/// assert!(result["stdout"].as_str().unwrap().contains("hello"));
/// ```
pub struct CodeExecTool;

// ---------------------------------------------------------------------------
// Language → image + command helpers
// ---------------------------------------------------------------------------

/// Maps a supported language name to its Docker image tag.
fn image_for_language(language: &str) -> Result<&'static str, ToolError> {
    match language {
        "python" => Ok("python:3.12-alpine"),
        "javascript" => Ok("node:22-alpine"),
        "bash" => Ok("alpine:3.21"),
        other => Err(ToolError::ExecutionFailed(format!(
            "Unsupported language '{}'. Supported: python, javascript, bash",
            other
        ))),
    }
}

/// Builds the container command that executes the supplied code snippet.
fn cmd_for_language(language: &str, code: &str) -> Vec<String> {
    match language {
        "python" => vec!["python3".into(), "-c".into(), code.into()],
        "javascript" => vec!["node".into(), "-e".into(), code.into()],
        _ => vec!["sh".into(), "-c".into(), code.into()],
    }
}

// ---------------------------------------------------------------------------
// Tool impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Tool for CodeExecTool {
    fn name(&self) -> &str {
        "code_exec"
    }

    fn description(&self) -> &str {
        "Executes code in a sandboxed Docker container and returns stdout, stderr, and exit code"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Programming language to use",
                    "enum": ["python", "javascript", "bash"]
                },
                "code": {
                    "type": "string",
                    "description": "Source code snippet to execute"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (1-30, default 10)",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS
                }
            },
            "required": ["language", "code"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        // -------------------------------------------------------------------
        // 1. Extract inputs
        // -------------------------------------------------------------------
        let language = input
            .get("language")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("missing or invalid 'language' field".to_string())
            })?;

        let code = input.get("code").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidInput("missing or invalid 'code' field".to_string())
        })?;

        let timeout_seconds = input
            .get("timeout_seconds")
            .and_then(|v| v.as_i64())
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
            .clamp(1, MAX_TIMEOUT_SECONDS);

        debug!(
            language,
            timeout_seconds, "CodeExecTool executing code snippet"
        );

        // -------------------------------------------------------------------
        // 2. Resolve image and command
        // -------------------------------------------------------------------
        let image = image_for_language(language)?;
        let cmd_owned = cmd_for_language(language, code);

        // -------------------------------------------------------------------
        // 3. Connect to Docker daemon
        // -------------------------------------------------------------------
        let docker = Docker::connect_with_socket_defaults().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to connect to Docker daemon: {}", e))
        })?;

        // -------------------------------------------------------------------
        // 4. Create container with security constraints
        // -------------------------------------------------------------------
        let host_config = HostConfig {
            memory: Some(MEMORY_LIMIT_BYTES),
            readonly_rootfs: Some(true),
            // Isolate from all networks
            network_mode: Some("none".to_string()),
            // Limit runaway processes
            pids_limit: Some(64),
            ..Default::default()
        };

        let container_body = ContainerCreateBody {
            image: Some(image.to_string()),
            cmd: Some(cmd_owned),
            network_disabled: Some(true),
            host_config: Some(host_config),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            ..Default::default()
        };

        let create_opts = CreateContainerOptions {
            name: Some(String::new()),
            ..Default::default()
        };

        let container = docker
            .create_container(Some(create_opts), container_body)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to create container: {}", e))
            })?;

        let container_id = container.id.clone();
        debug!(container_id = %container_id, "Container created");

        // -------------------------------------------------------------------
        // 5. Start container, collect output, always clean up
        // -------------------------------------------------------------------
        let result = run_container(&docker, &container_id, timeout_seconds).await;

        // Best-effort cleanup — log failures, never propagate them
        if let Err(e) = docker
            .remove_container(
                &container_id,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await
        {
            warn!(container_id = %container_id, error = %e, "Failed to remove container (cleanup)");
        } else {
            debug!(container_id = %container_id, "Container removed");
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Internal: start → stream logs → wait for exit
// ---------------------------------------------------------------------------

async fn run_container(
    docker: &Docker,
    container_id: &str,
    _timeout_seconds: i64,
) -> Result<Value, ToolError> {
    // Start
    docker
        .start_container(container_id, None::<StartContainerOptions>)
        .await
        .map_err(|e| ToolError::ExecutionFailed(format!("Failed to start container: {}", e)))?;

    // Stream logs (stdout + stderr) until the container exits
    let log_opts = LogsOptions {
        follow: true,
        stdout: true,
        stderr: true,
        ..Default::default()
    };

    let mut log_stream = docker.logs(container_id, Some(log_opts));
    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();

    while let Some(msg) = log_stream.next().await {
        match msg {
            Ok(LogOutput::StdOut { message }) => {
                stdout_buf.push_str(&String::from_utf8_lossy(&message));
            }
            Ok(LogOutput::StdErr { message }) => {
                stderr_buf.push_str(&String::from_utf8_lossy(&message));
            }
            Ok(_) => {} // StdIn / Console — ignore
            Err(e) => {
                warn!(container_id, error = %e, "Error reading container log stream");
            }
        }
    }

    // Wait for exit and capture status code
    let wait_opts = WaitContainerOptions {
        condition: "not-running".to_string(),
    };

    let mut wait_stream = docker.wait_container(container_id, Some(wait_opts));
    let exit_code = if let Some(wait_result) = wait_stream.next().await {
        match wait_result {
            Ok(response) => response.status_code,
            Err(e) => {
                warn!(container_id, error = %e, "Error waiting for container exit");
                -1
            }
        }
    } else {
        -1
    };

    debug!(container_id, exit_code, "Container exited");

    Ok(json!({
        "stdout":    stdout_buf,
        "stderr":    stderr_buf,
        "exit_code": exit_code
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------
    // Unit tests — no Docker required
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_code_exec_metadata() {
        let tool = CodeExecTool;
        assert_eq!(tool.name(), "code_exec");
        assert!(tool.description().contains("Docker"));
    }

    #[tokio::test]
    async fn test_code_exec_input_schema() {
        let tool = CodeExecTool;
        let schema = tool.input_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["language"].is_object());
        assert!(schema["properties"]["code"].is_object());
        assert!(schema["properties"]["timeout_seconds"].is_object());
        assert_eq!(schema["required"][0], "language");
        assert_eq!(schema["required"][1], "code");
        assert_eq!(
            schema["properties"]["timeout_seconds"]["default"],
            DEFAULT_TIMEOUT_SECONDS
        );
        assert_eq!(
            schema["properties"]["timeout_seconds"]["maximum"],
            MAX_TIMEOUT_SECONDS
        );
    }

    #[tokio::test]
    async fn test_code_exec_missing_language() {
        let tool = CodeExecTool;
        let input = json!({ "code": "print('hi')" });

        let result = tool.execute(input).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(
                    msg.contains("language"),
                    "Expected 'language' in error: {msg}"
                );
            }
            e => panic!("Expected InvalidInput, got: {e}"),
        }
    }

    #[tokio::test]
    async fn test_code_exec_missing_code() {
        let tool = CodeExecTool;
        let input = json!({ "language": "python" });

        let result = tool.execute(input).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("code"), "Expected 'code' in error: {msg}");
            }
            e => panic!("Expected InvalidInput, got: {e}"),
        }
    }

    #[tokio::test]
    async fn test_code_exec_wrong_language_type() {
        let tool = CodeExecTool;
        let input = json!({ "language": 42, "code": "print('hi')" });

        let result = tool.execute(input).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            e => panic!("Expected InvalidInput, got: {e}"),
        }
    }

    #[test]
    fn test_image_for_known_languages() {
        assert_eq!(image_for_language("python").unwrap(), "python:3.12-alpine");
        assert_eq!(image_for_language("javascript").unwrap(), "node:22-alpine");
        assert_eq!(image_for_language("bash").unwrap(), "alpine:3.21");
    }

    #[test]
    fn test_image_for_unknown_language() {
        let result = image_for_language("cobol");
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed(msg) => {
                assert!(msg.contains("cobol"));
                assert!(msg.contains("Unsupported"));
            }
            e => panic!("Expected ExecutionFailed, got: {e}"),
        }
    }

    #[test]
    fn test_cmd_for_python() {
        let cmd = cmd_for_language("python", "print('hi')");
        assert_eq!(cmd, vec!["python3", "-c", "print('hi')"]);
    }

    #[test]
    fn test_cmd_for_javascript() {
        let cmd = cmd_for_language("javascript", "console.log('hi')");
        assert_eq!(cmd, vec!["node", "-e", "console.log('hi')"]);
    }

    #[test]
    fn test_cmd_for_bash() {
        let cmd = cmd_for_language("bash", "echo hi");
        assert_eq!(cmd, vec!["sh", "-c", "echo hi"]);
    }

    #[test]
    fn test_timeout_constants() {
        assert!(DEFAULT_TIMEOUT_SECONDS > 0);
        assert!(MAX_TIMEOUT_SECONDS >= DEFAULT_TIMEOUT_SECONDS);
        assert!(MAX_TIMEOUT_SECONDS <= 60);
    }

    // ------------------------------------------------------------------
    // Integration tests — require Docker daemon, skipped in CI
    // ------------------------------------------------------------------

    #[tokio::test]
    #[ignore]
    async fn test_code_exec_real_python_hello() {
        let tool = CodeExecTool;
        let input = json!({
            "language": "python",
            "code": "print('hello from python')"
        });

        let result = tool.execute(input).await.expect("Execution failed");
        assert_eq!(result["exit_code"], 0);
        assert!(
            result["stdout"]
                .as_str()
                .unwrap()
                .contains("hello from python"),
            "stdout: {}",
            result["stdout"]
        );
        assert_eq!(result["stderr"].as_str().unwrap().trim(), "");
    }

    #[tokio::test]
    #[ignore]
    async fn test_code_exec_real_python_exit_code() {
        let tool = CodeExecTool;
        let input = json!({
            "language": "python",
            "code": "import sys; sys.exit(42)"
        });

        let result = tool.execute(input).await.expect("Execution failed");
        assert_eq!(result["exit_code"], 42);
    }

    #[tokio::test]
    #[ignore]
    async fn test_code_exec_real_javascript_hello() {
        let tool = CodeExecTool;
        let input = json!({
            "language": "javascript",
            "code": "console.log('hello from node')"
        });

        let result = tool.execute(input).await.expect("Execution failed");
        assert_eq!(result["exit_code"], 0);
        assert!(
            result["stdout"]
                .as_str()
                .unwrap()
                .contains("hello from node"),
            "stdout: {}",
            result["stdout"]
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_code_exec_real_bash_hello() {
        let tool = CodeExecTool;
        let input = json!({
            "language": "bash",
            "code": "echo 'hello from bash'"
        });

        let result = tool.execute(input).await.expect("Execution failed");
        assert_eq!(result["exit_code"], 0);
        assert!(
            result["stdout"]
                .as_str()
                .unwrap()
                .contains("hello from bash"),
            "stdout: {}",
            result["stdout"]
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_code_exec_real_stderr_captured() {
        let tool = CodeExecTool;
        let input = json!({
            "language": "python",
            "code": "import sys; sys.stderr.write('error output\\n')"
        });

        let result = tool.execute(input).await.expect("Execution failed");
        assert!(
            result["stderr"].as_str().unwrap().contains("error output"),
            "stderr: {}",
            result["stderr"]
        );
    }
}
