//! Axum HTTP server exposing the OpenAI-compatible API.

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
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
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

use crate::budget::{TokenBudget, TokenPool};
use crate::cache::{CacheConfig, ResponseCache};
use crate::inference::{InferenceRequest, Message, Role};
use crate::provider::ProviderRegistry;
use crate::router::{self as hoosh_router, ProviderRoute, RoutingStrategy};

/// Shared server state.
pub struct AppState {
    pub router: hoosh_router::Router,
    pub cache: ResponseCache,
    pub budget: std::sync::Mutex<TokenBudget>,
    pub providers: ProviderRegistry,
    #[cfg(feature = "whisper")]
    pub whisper: Option<std::sync::Arc<crate::provider::whisper::WhisperProvider>>,
}

/// Server configuration.
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub routes: Vec<ProviderRoute>,
    pub strategy: RoutingStrategy,
    pub cache_config: CacheConfig,
    pub budget_pools: Vec<TokenPool>,
    /// Path to whisper model file (e.g. "models/ggml-base.en.bin").
    pub whisper_model: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".into(),
            port: 8088,
            routes: Vec::new(),
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: Vec::new(),
            whisper_model: None,
        }
    }
}

/// Build the axum Router from a ServerConfig. Useful for testing.
pub fn build_app(config: ServerConfig) -> Router {
    let mut providers = ProviderRegistry::new();
    for route in &config.routes {
        if route.enabled {
            providers.register_from_route(route);
        }
    }
    tracing::info!("{} provider backend(s) registered", providers.len());

    let mut budget = TokenBudget::new();
    if !config.budget_pools.iter().any(|p| p.name == "default") {
        // 10M tokens default — reasonable for local use, explicit config for production
        budget.add_pool(TokenPool::new("default", 10_000_000));
    }
    for pool in config.budget_pools {
        budget.add_pool(pool);
    }
    tracing::info!("{} token pool(s) configured", budget.pools().len());

    #[cfg(feature = "whisper")]
    let whisper = config.whisper_model.as_ref().and_then(|path| {
        match crate::provider::whisper::WhisperProvider::new(path) {
            Ok(w) => {
                tracing::info!("whisper model loaded: {path}");
                Some(std::sync::Arc::new(w))
            }
            Err(e) => {
                tracing::warn!("failed to load whisper model '{path}': {e}");
                None
            }
        }
    });

    let state = Arc::new(AppState {
        router: hoosh_router::Router::new(config.routes, config.strategy),
        cache: ResponseCache::new(config.cache_config),
        budget: std::sync::Mutex::new(budget),
        providers,
        #[cfg(feature = "whisper")]
        whisper,
    });

    #[allow(unused_mut)]
    let mut app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health))
        .route("/v1/health/providers", get(health_providers))
        .route("/v1/tokens/check", post(tokens_check))
        .route("/v1/tokens/reserve", post(tokens_reserve))
        .route("/v1/tokens/report", post(tokens_report))
        .route("/v1/tokens/pools", get(tokens_pools));

    #[cfg(feature = "whisper")]
    {
        app = app.route("/v1/audio/transcriptions", post(transcribe));
    }

    app.layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB max body (for audio uploads)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the hoosh HTTP server.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    let addr = format!("{}:{}", config.bind, config.port);
    let app = build_app(config);
    tracing::info!("hoosh v{} listening on {}", env!("CARGO_PKG_VERSION"), addr);
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
    /// Token budget pool name (defaults to "default").
    #[serde(default = "default_pool_name")]
    pool: String,
}

fn default_pool_name() -> String {
    "default".into()
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
    // Input validation
    if req.messages.len() > 256 {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("Too many messages: {} (max 256)", req.messages.len()),
        )
        .into_response();
    }
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages array is empty").into_response();
    }
    if req.model.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "model field is required").into_response();
    }

    // Find a route for this model
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

    // Look up the live provider backend
    let provider = match state.providers.get(route.provider, &route.base_url) {
        Some(p) => p,
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Provider '{}' matched for model '{}' but no backend is registered. \
                     Is the '{}' feature enabled?",
                    route.provider, req.model, route.provider
                ),
            )
            .into_response();
        }
    };

    // Convert ChatRequest → InferenceRequest
    let inference_req = InferenceRequest {
        model: req.model.clone(),
        prompt: String::new(),
        system: None,
        messages: req
            .messages
            .iter()
            .map(|m| Message {
                role: match m.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                },
                content: m.content.clone(),
            })
            .collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
    };

    // Token budget: validate pool exists, then atomically reserve
    let estimated_tokens = req.max_tokens.unwrap_or(1024) as u64;
    let pool_name = req.pool.clone();
    {
        let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
        match budget.get_pool(&pool_name) {
            None => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Token pool '{}' does not exist", pool_name),
                )
                .into_response();
            }
            Some(pool) if !pool.can_reserve(estimated_tokens) => {
                let remaining = pool.available();
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "Token budget exceeded: pool '{}' has {} tokens remaining, requested {}",
                        pool_name, remaining, estimated_tokens
                    ),
                )
                .into_response();
            }
            _ => {}
        }
        budget.reserve(&pool_name, estimated_tokens);
    }

    if req.stream {
        // Streaming response via SSE
        let model = req.model.clone();
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        let rx = match provider.infer_stream(inference_req).await {
            Ok(rx) => rx,
            Err(e) => {
                // Release reservation on error
                let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
                budget.report(&pool_name, estimated_tokens, 0);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {e}"),
                )
                .into_response();
            }
        };

        // Report budget after stream completes
        let state_clone = state.clone();
        let pool_clone = pool_name.clone();

        let s = async_stream::stream! {
            let mut rx = rx;
            let mut token_count: u64 = 0;
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(token) => {
                        token_count += 1;
                        let chunk = serde_json::json!({
                            "id": &id,
                            "object": "chat.completion.chunk",
                            "model": &model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": serde_json::Value::Null,
                            }]
                        });
                        yield Ok::<_, std::convert::Infallible>(
                            Event::default().data(chunk.to_string())
                        );
                    }
                    Err(e) => {
                        tracing::error!("stream error: {e}");
                        break;
                    }
                }
            }
            // Final chunk with finish_reason
            let done_chunk = serde_json::json!({
                "id": &id,
                "object": "chat.completion.chunk",
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }]
            });
            yield Ok(Event::default().data(done_chunk.to_string()));
            yield Ok(Event::default().data("[DONE]"));

            // Report actual token usage
            let mut budget = state_clone.budget.lock().unwrap_or_else(|e| e.into_inner());
            budget.report(&pool_clone, estimated_tokens, token_count);
        };

        return Sse::new(s).keep_alive(KeepAlive::default()).into_response();
    }

    // Non-streaming
    match provider.infer(&inference_req).await {
        Ok(result) => {
            // Report actual usage to budget
            let actual = result.usage.total_tokens as u64;
            {
                let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
                budget.report(&pool_name, estimated_tokens, actual);
            }

            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion",
                created: chrono::Utc::now().timestamp(),
                model: result.model.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant",
                        content: result.text,
                    },
                    finish_reason: "stop",
                }],
                usage: ChatUsage {
                    prompt_tokens: result.usage.prompt_tokens,
                    completion_tokens: result.usage.completion_tokens,
                    total_tokens: result.usage.total_tokens,
                },
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            // Release reservation on error
            let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
            budget.report(&pool_name, estimated_tokens, 0);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {e}"),
            )
            .into_response()
        }
    }
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
    let mut models = Vec::new();

    // Query live providers for real model lists
    for route in state.router.routes() {
        if !route.enabled {
            continue;
        }
        if let Some(provider) = state.providers.get(route.provider, &route.base_url) {
            match provider.list_models().await {
                Ok(live_models) => {
                    for m in live_models {
                        models.push(ModelObject {
                            id: m.id,
                            object: "model",
                            owned_by: m.provider,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("failed to list models from {}: {e}", route.provider);
                    // Fall back to configured patterns
                    for pattern in &route.model_patterns {
                        models.push(ModelObject {
                            id: pattern.clone(),
                            object: "model",
                            owned_by: route.provider.to_string(),
                        });
                    }
                }
            }
        } else {
            // No live backend — show configured patterns
            for pattern in &route.model_patterns {
                models.push(ModelObject {
                    id: pattern.clone(),
                    object: "model",
                    owned_by: route.provider.to_string(),
                });
            }
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
    status: String,
}

async fn health_providers(State(state): State<Arc<AppState>>) -> Json<Vec<ProviderHealth>> {
    let mut results = Vec::new();

    for route in state.router.routes() {
        let status = if !route.enabled {
            "disabled".to_string()
        } else if let Some(provider) = state.providers.get(route.provider, &route.base_url) {
            match provider.health_check().await {
                Ok(true) => "healthy".to_string(),
                Ok(false) => "unhealthy".to_string(),
                Err(e) => format!("error: {e}"),
            }
        } else {
            "no_backend".to_string()
        };

        results.push(ProviderHealth {
            provider: route.provider.to_string(),
            base_url: route.base_url.clone(),
            enabled: route.enabled,
            status,
        });
    }

    Json(results)
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
    let budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
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
    let budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
    let pools: Vec<TokenPool> = budget.pools().values().cloned().collect();
    Json(pools)
}

// ---------------------------------------------------------------------------
// Speech-to-text: /v1/audio/transcriptions (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper")]
async fn transcribe(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let whisper = match &state.whisper {
        Some(w) => w.clone(),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Whisper model not loaded. Set whisper_model in config.",
            )
            .into_response();
        }
    };

    let request = crate::inference::TranscriptionRequest {
        audio: body.to_vec(),
        language: None,
        word_timestamps: false,
    };

    match whisper.transcribe_async(request).await {
        Ok(result) => {
            let resp = serde_json::json!({
                "text": result.text,
                "language": result.language,
                "duration": result.duration_secs,
                "segments": result.segments.iter().map(|s| {
                    serde_json::json!({
                        "text": s.text,
                        "start": s.start_secs,
                        "end": s.end_secs,
                    })
                }).collect::<Vec<_>>(),
            });
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Transcription error: {e}"),
        )
        .into_response(),
    }
}
