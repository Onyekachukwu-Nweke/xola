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
    #[instrument(
        name = "ipc_reason",
        skip(self, request),
        fields(
            ipc.endpoint = "/reason",
            task_goal = %request.task_goal,
            llm.model = tracing::field::Empty,
            llm.tokens_in = tracing::field::Empty,
            llm.tokens_out = tracing::field::Empty,
            llm.cost_usd = tracing::field::Empty,
        )
    )]
    async fn reason(&self, request: &ReasonRequest) -> Result<ReasonResponse, IpcError> {
        debug!("Calling POST /reason");
        let response: ReasonResponse = self.post("/reason", request).await?;

        // Attach cost data to the current span (L5-04).
        if let Some(ref usage) = response.usage {
            let span = tracing::Span::current();
            span.record("llm.model", &usage.model.as_str());
            span.record("llm.tokens_in", usage.tokens_in);
            span.record("llm.tokens_out", usage.tokens_out);
            span.record("llm.cost_usd", usage.cost_usd);
        }

        Ok(response)
    }

    #[instrument(name = "ipc_embed", skip(self, request), fields(ipc.endpoint = "/embed"))]
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, IpcError> {
        debug!("Calling POST /embed");
        self.post("/embed", request).await
    }

    #[instrument(
        name = "ipc_summarize",
        skip(self, request),
        fields(ipc.endpoint = "/summarize")
    )]
    async fn summarize(&self, request: &SummarizeRequest) -> Result<SummarizeResponse, IpcError> {
        debug!("Calling POST /summarize");
        self.post("/summarize", request).await
    }
}
