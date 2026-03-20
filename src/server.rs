//! Axum HTTP server exposing the OpenAI-compatible API.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::stream;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

use crate::budget::{TokenBudget, TokenPool};
use crate::cache::{CacheConfig, ResponseCache};
use crate::router::{self as hoosh_router, ProviderRoute, RoutingStrategy};

/// Shared server state.
pub struct AppState {
    pub router: hoosh_router::Router,
    pub cache: ResponseCache,
    pub budget: std::sync::Mutex<TokenBudget>,
}

/// Server configuration.
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub routes: Vec<ProviderRoute>,
    pub strategy: RoutingStrategy,
    pub cache_config: CacheConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".into(),
            port: 8088,
            routes: Vec::new(),
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
        }
    }
}

/// Start the hoosh HTTP server.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        router: hoosh_router::Router::new(config.routes, config.strategy),
        cache: ResponseCache::new(config.cache_config),
        budget: std::sync::Mutex::new(TokenBudget::new()),
    });

    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // Health
        .route("/v1/health", get(health))
        .route("/v1/health/providers", get(health_providers))
        // Token budget
        .route("/v1/tokens/check", post(tokens_check))
        .route("/v1/tokens/reserve", post(tokens_reserve))
        .route("/v1/tokens/report", post(tokens_report))
        .route("/v1/tokens/pools", get(tokens_pools))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", config.bind, config.port);
    tracing::info!(
        "hoosh v{} listening on {}",
        env!("CARGO_PKG_VERSION"),
        addr
    );
    tracing::info!("OpenAI-compatible API: http://{}/v1/chat/completions", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    tracing::info!("shutting down");
}

// ---------------------------------------------------------------------------
// OpenAI-compatible: /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields consumed by future provider backends
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields consumed by future provider backends
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: Option<String>,
}

fn error_response(status: StatusCode, message: impl Into<String>) -> impl IntoResponse {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                r#type: "error",
                code: None,
            },
        }),
    )
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    // Check if we have a provider for this model
    let route = match state.router.select(&req.model) {
        Some(r) => r,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!(
                    "No provider configured for model '{}'. Configure provider routes to handle this model.",
                    req.model
                ),
            )
            .into_response();
        }
    };

    if req.stream {
        // Return SSE stream with a single done event (no provider backends yet)
        let model = req.model.clone();
        let provider = route.provider.to_string();
        let s = stream::iter(vec![
            Ok::<_, std::convert::Infallible>(
                Event::default().data(format!(
                    "{{\"id\":\"chatcmpl-stub\",\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"[hoosh] Provider '{}' matched but no backend implementation yet.\"}},\"finish_reason\":\"stop\"}}]}}",
                    model, provider
                )),
            ),
            Ok(Event::default().data("[DONE]")),
        ]);
        return Sse::new(s).keep_alive(KeepAlive::default()).into_response();
    }

    // Non-streaming stub response
    let provider = route.provider.to_string();
    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion",
        created: chrono::Utc::now().timestamp(),
        model: req.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: format!(
                    "[hoosh] Provider '{}' matched for model '{}', but no backend implementation yet. \
                     This confirms routing works — provider backends ship in v0.5.0+.",
                    provider, req.model
                ),
            },
            finish_reason: "stop",
        }],
        usage: ChatUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    (StatusCode::OK, Json(resp)).into_response()
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    owned_by: String,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    // List model patterns from configured routes (no live provider queries yet)
    let mut models = Vec::new();
    for route in state.router.routes() {
        if !route.enabled {
            continue;
        }
        for pattern in &route.model_patterns {
            models.push(ModelObject {
                id: pattern.clone(),
                object: "model",
                owned_by: route.provider.to_string(),
            });
        }
    }
    Json(ModelsResponse {
        object: "list",
        data: models,
    })
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    providers_configured: usize,
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        providers_configured: state.router.routes().len(),
    })
}

#[derive(Serialize)]
struct ProviderHealth {
    provider: String,
    base_url: String,
    enabled: bool,
    // No live health checks yet — backends aren't implemented
    status: &'static str,
}

async fn health_providers(State(state): State<Arc<AppState>>) -> Json<Vec<ProviderHealth>> {
    let providers: Vec<ProviderHealth> = state
        .router
        .routes()
        .iter()
        .map(|r| ProviderHealth {
            provider: r.provider.to_string(),
            base_url: r.base_url.clone(),
            enabled: r.enabled,
            status: if r.enabled { "unknown" } else { "disabled" },
        })
        .collect();
    Json(providers)
}

// ---------------------------------------------------------------------------
// Token budget endpoints
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenCheckRequest {
    pool: String,
    tokens: u64,
}

#[derive(Serialize)]
struct TokenCheckResponse {
    allowed: bool,
    available: u64,
}

async fn tokens_check(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenCheckRequest>,
) -> impl IntoResponse {
    let budget = state.budget.lock().unwrap();
    match budget.get_pool(&req.pool) {
        Some(pool) => (
            StatusCode::OK,
            Json(TokenCheckResponse {
                allowed: pool.can_reserve(req.tokens),
                available: pool.available(),
            }),
        )
            .into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("Token pool '{}' not found", req.pool),
        )
        .into_response(),
    }
}

#[derive(Deserialize)]
struct TokenReserveRequest {
    pool: String,
    tokens: u64,
}

#[derive(Serialize)]
struct TokenReserveResponse {
    reserved: bool,
    available: u64,
}

async fn tokens_reserve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenReserveRequest>,
) -> impl IntoResponse {
    let mut budget = state.budget.lock().unwrap();
    let reserved = budget.reserve(&req.pool, req.tokens);
    match budget.get_pool(&req.pool) {
        Some(pool) => (
            StatusCode::OK,
            Json(TokenReserveResponse {
                reserved,
                available: pool.available(),
            }),
        )
            .into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("Token pool '{}' not found", req.pool),
        )
        .into_response(),
    }
}

#[derive(Deserialize)]
struct TokenReportRequest {
    pool: String,
    reserved: u64,
    actual: u64,
}

#[derive(Serialize)]
struct TokenReportResponse {
    used: u64,
    available: u64,
}

async fn tokens_report(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenReportRequest>,
) -> impl IntoResponse {
    let mut budget = state.budget.lock().unwrap();
    budget.report(&req.pool, req.reserved, req.actual);
    match budget.get_pool(&req.pool) {
        Some(pool) => (
            StatusCode::OK,
            Json(TokenReportResponse {
                used: pool.used,
                available: pool.available(),
            }),
        )
            .into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("Token pool '{}' not found", req.pool),
        )
        .into_response(),
    }
}

async fn tokens_pools(State(state): State<Arc<AppState>>) -> Json<Vec<TokenPool>> {
    let budget = state.budget.lock().unwrap();
    let pools: Vec<TokenPool> = budget.pools().values().cloned().collect();
    Json(pools)
}
