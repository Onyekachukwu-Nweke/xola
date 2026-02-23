//! Error types for IPC communication with the Python LLM surface.

use thiserror::Error;

/// Errors that can occur when communicating with the Python IPC server.
#[derive(Error, Debug)]
pub enum IpcError {
    /// HTTP request to the Python server failed (network, timeout, etc.)
    #[error("IPC transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// Python server returned a non-2xx status code.
    #[error("IPC server error: status {status}, body: {body}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Response body text.
        body: String,
    },

    /// Failed to deserialise the response body into the expected type.
    #[error("IPC deserialisation error: {0}")]
    Deserialisation(String),
}
