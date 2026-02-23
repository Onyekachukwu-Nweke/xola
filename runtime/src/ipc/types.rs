//! Serde types for the IPC contract between Rust and Python.
//!
//! These mirror the Pydantic models in `llm_surface/server.py`.
//! See the IPC Contract section in `AGENTS.md` for the canonical schema.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// /reason
// ---------------------------------------------------------------------------

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcMessage {
    pub role: String,
    pub content: String,
}

/// Request body for `POST /reason`.
#[derive(Debug, Clone, Serialize)]
pub struct ReasonRequest {
    pub messages: Vec<IpcMessage>,
    pub tool_schemas: Vec<Value>,
    pub memory_context: Vec<String>,
    pub task_goal: String,
}

/// Response body from `POST /reason`.
#[derive(Debug, Clone, Deserialize)]
pub struct ReasonResponse {
    pub thought: String,
    pub action: Option<String>,
    #[serde(default)]
    pub action_input: Value,
    pub is_final: bool,
    pub final_answer: Option<String>,
}

// ---------------------------------------------------------------------------
// /embed
// ---------------------------------------------------------------------------

/// Request body for `POST /embed`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbedRequest {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Response body from `POST /embed`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbedResponse {
    pub vector: Vec<f32>,
    pub token_count: u32,
}

// ---------------------------------------------------------------------------
// /summarize
// ---------------------------------------------------------------------------

/// Request body for `POST /summarize`.
#[derive(Debug, Clone, Serialize)]
pub struct SummarizeRequest {
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_summary_tokens: Option<u32>,
}

/// Response body from `POST /summarize`.
#[derive(Debug, Clone, Deserialize)]
pub struct SummarizeResponse {
    pub summary: String,
    pub token_count: u32,
}
