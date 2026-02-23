//! HTTP client for the Python LLM surface IPC server.
//!
//! Uses `reqwest` over TCP. The transport is abstracted behind the
//! [`IpcTransport`](super::IpcTransport) trait so a Unix socket
//! implementation can be added later without changing callers.

use std::time::Duration;

use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::{debug, error, instrument};

use super::error::IpcError;
use super::types::*;
use super::IpcTransport;

/// Default timeout for IPC calls to the Python server.
const DEFAULT_IPC_TIMEOUT: Duration = Duration::from_secs(120);

/// HTTP-based IPC client for the Python LLM surface.
///
/// # Thread Safety
///
/// `IpcClient` is `Clone + Send + Sync` — safe to share across tokio tasks
/// via `Arc<IpcClient>`.
#[derive(Clone, Debug)]
pub struct IpcClient {
    http: Client,
    base_url: String,
}

impl IpcClient {
    /// Create a new client pointed at the given base URL.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = IpcClient::new("http://127.0.0.1:8001")?;
    /// ```
    pub fn new(base_url: impl Into<String>) -> Result<Self, IpcError> {
        let http = Client::builder()
            .timeout(DEFAULT_IPC_TIMEOUT)
            .build()
            .map_err(IpcError::Transport)?;
        Ok(Self {
            http,
            base_url: base_url.into(),
        })
    }

    /// POST JSON to an endpoint and deserialise the response.
    async fn post<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp, IpcError> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.http.post(&url).json(body).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body_text, path, "IPC call failed");
            return Err(IpcError::ServerError {
                status: status.as_u16(),
                body: body_text,
            });
        }
        response
            .json::<Resp>()
            .await
            .map_err(|e| IpcError::Deserialisation(e.to_string()))
    }
}

#[async_trait::async_trait]
impl IpcTransport for IpcClient {
    #[instrument(skip(self, request), fields(task_goal = %request.task_goal))]
    async fn reason(&self, request: &ReasonRequest) -> Result<ReasonResponse, IpcError> {
        debug!("Calling POST /reason");
        self.post("/reason", request).await
    }

    #[instrument(skip(self, request))]
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, IpcError> {
        debug!("Calling POST /embed");
        self.post("/embed", request).await
    }

    #[instrument(skip(self, request))]
    async fn summarize(&self, request: &SummarizeRequest) -> Result<SummarizeResponse, IpcError> {
        debug!("Calling POST /summarize");
        self.post("/summarize", request).await
    }
}
