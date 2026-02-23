//! IPC client for communicating with the Python LLM surface.
//!
//! The Rust runtime calls the Python server via HTTP. The client supports
//! TCP (default for development) with a Unix socket option for production.
//!
//! All state lives in Rust; Python is stateless between requests.
//! Data flows one direction: Rust → Python.

pub mod client;
pub mod error;
pub mod types;

pub use client::IpcClient;
pub use error::IpcError;
pub use types::*;

/// Trait abstracting IPC calls — enables mock implementations in tests.
///
/// `IpcClient` implements this over HTTP. Test code can provide a
/// `MockIpcTransport` that returns canned responses.
#[async_trait::async_trait]
pub trait IpcTransport: Send + Sync {
    /// Execute one ReAct reasoning step (`POST /reason`).
    async fn reason(&self, request: &ReasonRequest) -> Result<ReasonResponse, IpcError>;

    /// Generate an embedding vector (`POST /embed`).
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, IpcError>;

    /// Condense messages into a summary (`POST /summarize`).
    async fn summarize(&self, request: &SummarizeRequest) -> Result<SummarizeResponse, IpcError>;
}
