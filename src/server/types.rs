//! Request/response types for the OpenAI-compatible HTTP API.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::server::AppState;

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    /// Tool definitions the model may call.
    #[serde(default)]
    pub tools: Vec<crate::tools::ToolDefinition>,
    /// How the model should choose tools.
    #[serde(default)]
    pub tool_choice: Option<crate::tools::ToolChoice>,
    /// Token budget pool name (defaults to "default").
    #[serde(default = "default_pool_name")]
    pub pool: String,
}

fn default_pool_name() -> String {
    "default".into()
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<crate::tools::ToolCall>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Serialize)]
pub(crate) struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Serialize)]
pub(crate) struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<crate::tools::ToolCall>,
}

#[derive(Serialize)]
pub(crate) struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub(crate) struct ErrorDetail {
    pub message: String,
    pub r#type: &'static str,
    pub code: Option<String>,
}

pub(crate) fn error_response(
    status: axum::http::StatusCode,
    message: impl Into<String>,
) -> impl axum::response::IntoResponse {
    (
        status,
        axum::Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                r#type: "error",
                code: None,
            },
        }),
    )
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub(crate) struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: String,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub providers_configured: usize,
}

#[derive(Serialize)]
pub(crate) struct ProviderHealth {
    pub provider: String,
    pub base_url: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consecutive_failures: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

// ---------------------------------------------------------------------------
// Token budget
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct TokenCheckRequest {
    pub pool: String,
    pub tokens: u64,
}

#[derive(Serialize)]
pub(crate) struct TokenCheckResponse {
    pub allowed: bool,
    pub available: u64,
}

#[derive(Deserialize)]
pub(crate) struct TokenReserveRequest {
    pub pool: String,
    pub tokens: u64,
}

#[derive(Serialize)]
pub(crate) struct TokenReserveResponse {
    pub reserved: bool,
    pub available: u64,
}

#[derive(Deserialize)]
pub(crate) struct TokenReportRequest {
    pub pool: String,
    pub reserved: u64,
    pub actual: u64,
}

#[derive(Serialize)]
pub(crate) struct TokenReportResponse {
    pub used: u64,
    pub available: u64,
}

// ---------------------------------------------------------------------------
// MCP tools
// ---------------------------------------------------------------------------

#[cfg(feature = "tools")]
#[derive(Deserialize)]
pub(crate) struct ToolCallRequest {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Cost tracking
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct CostsResponse {
    pub records: Vec<crate::cost::ProviderCostRecord>,
    pub total_cost_usd: f64,
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(crate) struct AuditResponse {
    pub entries: Vec<crate::audit::AuditEntry>,
    pub total: usize,
    pub chain_valid: bool,
}

// ---------------------------------------------------------------------------
// Streaming budget guard
// ---------------------------------------------------------------------------

/// Drop guard that reports budget, cost, metrics, and events when a stream ends.
pub(crate) struct StreamBudgetGuard {
    pub state: Arc<AppState>,
    pub pool: String,
    pub estimated: u64,
    pub actual: Arc<std::sync::atomic::AtomicU64>,
    pub provider: String,
    pub model: String,
    pub start: std::time::Instant,
}

impl Drop for StreamBudgetGuard {
    fn drop(&mut self) {
        let actual = self.actual.load(std::sync::atomic::Ordering::Relaxed);
        let latency_ms = self.start.elapsed().as_millis() as u64;

        // Budget reporting
        match self.state.budget.try_lock() {
            Ok(mut budget) => budget.report(&self.pool, self.estimated, actual),
            Err(std::sync::TryLockError::Poisoned(e)) => {
                e.into_inner().report(&self.pool, self.estimated, actual);
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                let state = self.state.clone();
                let pool = self.pool.clone();
                let estimated = self.estimated;
                tokio::spawn(async move {
                    let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
                    budget.report(&pool, estimated, actual);
                });
            }
        }

        // Metrics
        crate::metrics::record_request(
            &self.provider,
            &self.model,
            "success",
            latency_ms as f64 / 1000.0,
            0,
            actual as u32,
        );

        // Event bus
        self.state.event_bus.publish(
            crate::events::topics::INFERENCE,
            crate::events::ProviderEvent::InferenceCompleted {
                provider: self.provider.clone(),
                model: self.model.clone(),
                latency_ms,
                tokens: actual as u32,
            },
        );
    }
}
