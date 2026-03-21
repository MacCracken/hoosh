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
use crate::cost::CostTracker;
use crate::inference::{InferenceRequest, Message, Role};
use crate::provider::ProviderRegistry;
use crate::router::{self as hoosh_router, ProviderRoute, RoutingStrategy};

/// Shared server state.
pub struct AppState {
    pub router: std::sync::RwLock<hoosh_router::Router>,
    /// Path to config file for hot-reload. None = reload disabled.
    pub config_path: Option<String>,
    pub cache: ResponseCache,
    pub budget: std::sync::Mutex<TokenBudget>,
    pub providers: ProviderRegistry,
    pub cost_tracker: Arc<CostTracker>,
    pub audit: Option<Arc<crate::audit::AuditChain>>,
    pub auth_token_digests: Vec<crate::middleware::auth::TokenDigest>,
    pub rate_limiter: Arc<crate::middleware::rate_limit::RateLimitRegistry>,
    pub event_bus: Arc<crate::events::EventBus>,
    pub inference_queue: Arc<crate::queue::InferenceQueue>,
    pub health_map: crate::health::HealthMap,
    pub heartbeat: Arc<majra::heartbeat::ConcurrentHeartbeatTracker>,
    #[cfg(feature = "whisper")]
    pub whisper: Option<std::sync::Arc<crate::provider::whisper::WhisperProvider>>,
    #[cfg(feature = "piper")]
    pub tts: Option<std::sync::Arc<crate::provider::tts::TtsProvider>>,
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
    /// Path to piper TTS model config (e.g. "models/en_US-lessac-medium.onnx.json").
    pub tts_model: Option<String>,
    /// Whether audit logging is enabled.
    pub audit_enabled: bool,
    /// HMAC signing key for audit chain.
    pub audit_signing_key: Option<String>,
    /// Max audit entries to keep in memory.
    pub audit_max_entries: usize,
    /// Bearer tokens for authentication. Empty = auth disabled.
    pub auth_tokens: Vec<String>,
    /// OTLP endpoint — enables OpenTelemetry when set (requires `otel` feature).
    pub otlp_endpoint: Option<String>,
    /// Service name for OpenTelemetry traces.
    pub telemetry_service_name: String,
    /// Health check interval in seconds. 0 = disabled. Defaults to 30.
    pub health_check_interval_secs: u64,
    /// Path to config file for hot-reload. None = reload disabled.
    pub config_path: Option<String>,
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
            tts_model: None,
            audit_enabled: false,
            audit_signing_key: None,
            audit_max_entries: 10_000,
            auth_tokens: Vec::new(),
            otlp_endpoint: None,
            telemetry_service_name: "hoosh".into(),
            health_check_interval_secs: 30,
            config_path: None,
        }
    }
}

/// Build the axum Router from a ServerConfig. Useful for testing.
/// Returns both the axum Router and the shared AppState for SIGHUP reload.
pub fn build_app(config: ServerConfig) -> (Router, Arc<AppState>) {
    let mut providers = ProviderRegistry::new();
    for route in &config.routes {
        if route.enabled {
            providers.register_from_route(route);
        }
    }
    crate::metrics::set_providers_configured(providers.len() as i64);
    tracing::info!("{} provider backend(s) registered", providers.len());

    if config.auth_tokens.is_empty() {
        tracing::warn!(
            "authentication is DISABLED — all requests will be accepted without a bearer token"
        );
    }

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

    #[cfg(feature = "piper")]
    let tts = config.tts_model.as_ref().map(|url| {
        tracing::info!("TTS backend configured: {url}");
        std::sync::Arc::new(crate::provider::tts::TtsProvider::new(url, None))
    });

    let cost_tracker = Arc::new(CostTracker::new());

    let audit = if config.audit_enabled {
        let key = match &config.audit_signing_key {
            Some(k) => k.as_bytes().to_vec(),
            None => {
                let key: [u8; 32] = rand::random();
                tracing::info!("audit chain enabled with auto-generated signing key");
                key.to_vec()
            }
        };
        Some(Arc::new(crate::audit::AuditChain::new(
            &key,
            config.audit_max_entries,
        )))
    } else {
        None
    };

    // Build per-provider rate limiter from route config
    let rate_limiter = Arc::new(crate::middleware::rate_limit::RateLimitRegistry::new());
    for route in &config.routes {
        if route.enabled
            && let Some(rpm) = route.rate_limit_rpm
        {
            let key = format!("{}:{}", route.provider, route.base_url);
            rate_limiter.configure(&key, rpm);
            tracing::info!("rate limit: {} → {} rpm", key, rpm);
        }
    }

    // Create event bus and inference queue
    let event_bus = Arc::new(crate::events::new_event_bus());
    let inference_queue = Arc::new(crate::queue::InferenceQueue::new());

    // Create the health map, heartbeat tracker, and wire into the router
    let health_map = crate::health::new_health_map();
    let heartbeat = Arc::new(majra::heartbeat::ConcurrentHeartbeatTracker::new(
        majra::heartbeat::HeartbeatConfig {
            suspect_after: std::time::Duration::from_secs(30),
            offline_after: std::time::Duration::from_secs(90),
            eviction_policy: None,
        },
    ));
    // Register all enabled providers with the heartbeat tracker
    for route in &config.routes {
        if route.enabled {
            let node_id = format!("{}:{}", route.provider, route.base_url);
            heartbeat.register(
                &node_id,
                serde_json::json!({"provider": route.provider.to_string(), "base_url": &route.base_url}),
            );
        }
    }
    let routes_for_checker = config.routes.clone();
    let health_interval = config.health_check_interval_secs;
    let mut router = hoosh_router::Router::new(config.routes, config.strategy);
    router.set_health_map(health_map.clone());

    let state = Arc::new(AppState {
        router: std::sync::RwLock::new(router),
        config_path: config.config_path,
        cache: ResponseCache::new(config.cache_config),
        budget: std::sync::Mutex::new(budget),
        providers,
        cost_tracker,
        audit,
        auth_token_digests: config
            .auth_tokens
            .iter()
            .map(|t| crate::middleware::auth::hash_token(t))
            .collect(),
        rate_limiter,
        event_bus: event_bus.clone(),
        inference_queue,
        health_map: health_map.clone(),
        heartbeat: heartbeat.clone(),
        #[cfg(feature = "whisper")]
        whisper,
        #[cfg(feature = "piper")]
        tts,
    });

    // Spawn background health checker if enabled
    if health_interval > 0 {
        let mut checker_providers = ProviderRegistry::new();
        for route in &routes_for_checker {
            if route.enabled {
                checker_providers.register_from_route(route);
            }
        }
        let _health_handle = crate::health::spawn_health_checker(
            Arc::new(checker_providers),
            routes_for_checker,
            health_map,
            health_interval,
            event_bus,
            heartbeat,
        );
        tracing::info!(
            "background health checker started (interval: {}s)",
            health_interval
        );
    }

    // API routes with 1MB body limit
    let api_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health))
        .route("/v1/health/providers", get(health_providers))
        .route("/v1/health/heartbeat", get(health_heartbeat))
        .route("/v1/tokens/check", post(tokens_check))
        .route("/v1/tokens/reserve", post(tokens_reserve))
        .route("/v1/tokens/report", post(tokens_report))
        .route("/v1/tokens/pools", get(tokens_pools))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/costs", get(costs_get))
        .route("/v1/costs/reset", post(costs_reset))
        .route("/v1/audit", get(audit_log))
        .route("/v1/admin/reload", post(admin_reload))
        .route("/v1/queue/status", get(queue_status))
        .layer(DefaultBodyLimit::max(1024 * 1024)); // 1 MB for JSON API

    #[allow(unused_mut)]
    let mut app = api_routes.route("/metrics", get(prometheus_metrics));

    // Audio routes with 50MB body limit
    #[cfg(feature = "whisper")]
    {
        app = app.route("/v1/audio/transcriptions", post(transcribe));
    }
    #[cfg(feature = "piper")]
    {
        app = app.route("/v1/audio/speech", post(text_to_speech));
    }
    #[cfg(any(feature = "whisper", feature = "piper"))]
    {
        app = app.layer(DefaultBodyLimit::max(50 * 1024 * 1024));
    }

    let router = app
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::middleware::auth::auth_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    (router, state)
}

/// Start the hoosh HTTP server.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    let addr = format!("{}:{}", config.bind, config.port);
    let (app, app_state) = build_app(config);
    tracing::info!("hoosh v{} listening on {}", env!("CARGO_PKG_VERSION"), addr);
    tracing::info!("OpenAI-compatible API: http://{}/v1/chat/completions", addr);

    // Spawn SIGHUP listener for hot-reload
    if let Some(config_path) = app_state.config_path.clone() {
        let state = app_state.clone();
        tokio::spawn(async move {
            let mut sig = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
                .expect("failed to register SIGHUP handler");
            loop {
                sig.recv().await;
                tracing::info!("SIGHUP received, reloading config from {}", config_path);
                reload_config(&state, &config_path);
            }
        });
    }

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
// Hot-reload: config reload without restart
// ---------------------------------------------------------------------------

fn reload_config(state: &Arc<AppState>, config_path: &str) {
    match crate::config::HooshConfig::load(config_path) {
        Ok(config) => {
            let routes = config.routes();
            let strategy: RoutingStrategy = config.server.strategy.into();

            // Note: provider re-registration requires mutable access.
            // Only the router is hot-reloaded; provider backends persist.

            // Swap router atomically
            let mut router = state.router.write().unwrap_or_else(|e| e.into_inner());
            router.reload(routes, strategy);
            router.set_health_map(state.health_map.clone());

            tracing::info!("config reloaded: {} routes", router.routes().len());
        }
        Err(e) => {
            tracing::error!("config reload failed: {e}");
        }
    }
}

async fn admin_reload(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(path) = &state.config_path {
        reload_config(&state, path);
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "reloaded"})),
        )
            .into_response()
    } else {
        error_response(
            StatusCode::BAD_REQUEST,
            "no config path configured for reload",
        )
        .into_response()
    }
}

// ---------------------------------------------------------------------------
// Queue status: /v1/queue/status
// ---------------------------------------------------------------------------

async fn queue_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "queued": state.inference_queue.len(),
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Prometheus metrics: /metrics
// ---------------------------------------------------------------------------

async fn prometheus_metrics() -> impl IntoResponse {
    let body = crate::metrics::gather();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
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

/// Drop guard that reports budget, cost, metrics, and events when a stream ends.
struct StreamBudgetGuard {
    state: Arc<AppState>,
    pool: String,
    estimated: u64,
    actual: std::sync::Arc<std::sync::atomic::AtomicU64>,
    provider: String,
    model: String,
    start: std::time::Instant,
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

        // Metrics (streaming requests were previously uncounted)
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
    let request_id = uuid::Uuid::new_v4();
    tracing::info!(request_id = %request_id, model = %req.model, "chat_completions");

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
    if req.model.len() > 256
        || req
            .model
            .bytes()
            .any(|b| b < 0x20 || b == b'\\' || b == b'"')
    {
        return error_response(StatusCode::BAD_REQUEST, "invalid model name").into_response();
    }
    if let Some(temp) = req.temperature
        && !(0.0..=2.0).contains(&temp)
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("temperature must be between 0.0 and 2.0, got {temp}"),
        )
        .into_response();
    }
    if let Some(tp) = req.top_p
        && !(0.0..=1.0).contains(&tp)
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("top_p must be between 0.0 and 1.0, got {tp}"),
        )
        .into_response();
    }

    // Find a route for this model
    let route = match state
        .router
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .select(&req.model)
        .cloned()
    {
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

    // Per-provider rate limit check
    let rate_key = format!("{}:{}", route.provider, route.base_url);
    if !state.rate_limiter.check(&rate_key) {
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "Rate limit exceeded for provider '{}'. Please try again later.",
                route.provider
            ),
        )
        .into_response();
    }

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

    // Enforce provider-level max_tokens limit
    let max_tokens = match (req.max_tokens, route.max_tokens_limit) {
        (Some(requested), Some(limit)) if requested > limit => {
            tracing::warn!(
                "clamping max_tokens from {} to provider limit {}",
                requested,
                limit
            );
            Some(limit)
        }
        (requested, _) => requested,
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
        max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
    };

    // Token budget: validate pool exists, then atomically reserve
    let estimated_tokens = max_tokens.unwrap_or(1024) as u64;
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
            Ok(rx) => {
                // Record audit event for streaming request
                if let Some(ref audit) = state.audit {
                    audit.record(
                        "inference.request",
                        "info",
                        &format!("Streaming inference started for model {}", model),
                        Some(&route.provider.to_string()),
                        Some(&model),
                        None,
                    );
                }
                rx
            }
            Err(e) => {
                // Record audit event for streaming error
                if let Some(ref audit) = state.audit {
                    audit.record(
                        "inference.error",
                        "error",
                        &format!("Streaming inference error: {e}"),
                        Some(&route.provider.to_string()),
                        Some(&model),
                        None,
                    );
                }
                // Release reservation on error
                let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
                budget.report(&pool_name, estimated_tokens, 0);
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, {
                    tracing::error!("inference error: {e}");
                    "Inference request failed".to_string()
                })
                .into_response();
            }
        };

        // Use a shared counter so the drop guard can report even on disconnect
        let token_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let token_count_clone = token_count.clone();
        let state_clone = state.clone();
        let pool_clone = pool_name.clone();

        // Drop guard: reports budget when stream ends (including on client disconnect)
        let budget_guard = StreamBudgetGuard {
            state: state_clone,
            pool: pool_clone,
            estimated: estimated_tokens,
            actual: token_count_clone,
            provider: route.provider.to_string(),
            model: req.model.clone(),
            start: std::time::Instant::now(),
        };

        let s = async_stream::stream! {
            let _guard = budget_guard;
            let mut rx = rx;
            let mut buf = String::with_capacity(256);
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(token) => {
                        token_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Build JSON directly into a reusable buffer (avoids serde_json::Value allocation per token)
                        buf.clear();
                        let escaped = serde_json::to_string(&token).unwrap_or_default();
                        use std::fmt::Write;
                        let _ = write!(
                            buf,
                            r#"{{"id":"{}","object":"chat.completion.chunk","model":"{}","choices":[{{"index":0,"delta":{{"content":{}}},"finish_reason":null}}]}}"#,
                            &id, &model, escaped
                        );
                        yield Ok::<_, std::convert::Infallible>(
                            Event::default().data(buf.as_str())
                        );
                    }
                    Err(e) => {
                        tracing::error!("stream error: {e}");
                        break;
                    }
                }
            }
            buf.clear();
            use std::fmt::Write;
            let _ = write!(
                buf,
                r#"{{"id":"{}","object":"chat.completion.chunk","model":"{}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}"#,
                &id, &model
            );
            yield Ok(Event::default().data(buf.as_str()));
            yield Ok(Event::default().data("[DONE]"));
        };

        return Sse::new(s)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response();
    }

    // Non-streaming
    match provider.infer(&inference_req).await {
        Ok(result) => {
            // Report latency for routing decisions
            state
                .router
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .report_latency(route.provider, &route.base_url, result.latency_ms);

            // Report actual usage to budget
            let actual = result.usage.total_tokens as u64;
            {
                let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
                budget.report(&pool_name, estimated_tokens, actual);
            }

            // Track cost
            let cost = state.cost_tracker.record(
                route.provider,
                &route.base_url,
                &result.model,
                &result.usage,
            );
            tracing::debug!(cost_usd = cost, model = %result.model, "request cost");

            // Record metrics
            crate::metrics::record_request(
                &result.provider,
                &result.model,
                "success",
                result.latency_ms as f64 / 1000.0,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            );

            // Publish inference completed event
            state.event_bus.publish(
                crate::events::topics::INFERENCE,
                crate::events::ProviderEvent::InferenceCompleted {
                    provider: result.provider.clone(),
                    model: result.model.clone(),
                    latency_ms: result.latency_ms,
                    tokens: result.usage.total_tokens,
                },
            );

            // Record audit event for successful inference
            if let Some(ref audit) = state.audit {
                audit.record(
                    "inference.response",
                    "info",
                    &format!("Inference completed for model {}", result.model),
                    Some(&route.provider.to_string()),
                    Some(&result.model),
                    Some(serde_json::json!({
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    })),
                );
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
            // Publish inference failed event
            state.event_bus.publish(
                crate::events::topics::ERRORS,
                crate::events::ProviderEvent::InferenceFailed {
                    provider: route.provider.to_string(),
                    model: req.model.clone(),
                    error: e.to_string(),
                },
            );

            // Record audit event for inference error
            if let Some(ref audit) = state.audit {
                audit.record(
                    "inference.error",
                    "error",
                    &{
                        tracing::error!("inference error: {e}");
                        "Inference request failed".to_string()
                    },
                    Some(&route.provider.to_string()),
                    Some(&req.model),
                    None,
                );
            }

            // Record error metrics
            crate::metrics::record_request(
                &route.provider.to_string(),
                &req.model,
                "error",
                0.0,
                0,
                0,
            );

            // Release reservation on error
            let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
            budget.report(&pool_name, estimated_tokens, 0);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, {
                tracing::error!("inference error: {e}");
                "Inference request failed".to_string()
            })
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

    // Clone routes so we don't hold the lock across awaits
    let routes: Vec<_> = state
        .router
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .routes()
        .to_vec();

    // Query live providers for real model lists
    for route in &routes {
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
        providers_configured: state
            .router
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .routes()
            .len(),
    })
}

#[derive(Serialize)]
struct ProviderHealth {
    provider: String,
    base_url: String,
    enabled: bool,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    consecutive_failures: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

async fn health_providers(State(state): State<Arc<AppState>>) -> Json<Vec<ProviderHealth>> {
    let mut results = Vec::new();

    // Clone routes so we don't hold the lock across awaits
    let routes: Vec<_> = state
        .router
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .routes()
        .to_vec();

    for route in &routes {
        let key = (route.provider, route.base_url.clone());
        let bg_state = state.health_map.get(&key);

        let status = if !route.enabled {
            "disabled".to_string()
        } else if let Some(ref hs) = bg_state {
            // Use background health check state if available
            if hs.is_healthy {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            }
        } else if let Some(provider) = state.providers.get(route.provider, &route.base_url) {
            // Fall back to on-demand check if no background state
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
            consecutive_failures: bg_state.as_ref().map(|s| s.consecutive_failures),
            last_error: bg_state.as_ref().and_then(|s| s.last_error.clone()),
        });
    }

    Json(results)
}

// ---------------------------------------------------------------------------
// Heartbeat fleet stats: /v1/health/heartbeat
// ---------------------------------------------------------------------------

async fn health_heartbeat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.heartbeat.fleet_stats();
    (StatusCode::OK, Json(stats)).into_response()
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
// Cost tracking: /v1/costs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CostsResponse {
    records: Vec<crate::cost::ProviderCostRecord>,
    total_cost_usd: f64,
}

async fn costs_get(State(state): State<Arc<AppState>>) -> Json<CostsResponse> {
    let (records, total_cost_usd) = state.cost_tracker.all_with_total();
    Json(CostsResponse {
        records,
        total_cost_usd,
    })
}

async fn costs_reset(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    if let Some(ref audit) = state.audit {
        audit.record(
            "admin.costs_reset",
            "warn",
            "Cost tracking counters reset",
            None,
            None,
            None,
        );
    }
    state.cost_tracker.reset();
    Json(serde_json::json!({ "status": "ok" }))
}

// ---------------------------------------------------------------------------
// Audit log: /v1/audit
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AuditResponse {
    entries: Vec<crate::audit::AuditEntry>,
    total: usize,
    chain_valid: bool,
}

async fn audit_log(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.audit {
        Some(audit) => {
            let (entries, total, chain_valid) = audit.snapshot(100);
            (
                StatusCode::OK,
                Json(AuditResponse {
                    entries,
                    total,
                    chain_valid,
                }),
            )
                .into_response()
        }
        None => error_response(
            StatusCode::NOT_FOUND,
            "Audit logging is not enabled. Set [audit] enabled = true in config.",
        )
        .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Speech-to-text: /v1/audio/transcriptions (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper")]
async fn transcribe(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    // Validate content type — accept audio/* or application/octet-stream
    if let Some(ct) = headers.get("content-type") {
        let ct_str = ct.to_str().unwrap_or("");
        if !ct_str.starts_with("audio/")
            && !ct_str.starts_with("application/octet-stream")
            && !ct_str.starts_with("multipart/form-data")
        {
            return error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("expected audio/* content type, got: {ct_str}"),
            )
            .into_response();
        }
    }

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

// ---------------------------------------------------------------------------
// Text-to-speech: /v1/audio/speech (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[cfg(feature = "piper")]
async fn text_to_speech(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::inference::SpeechRequest>,
) -> impl IntoResponse {
    let tts = match &state.tts {
        Some(t) => t.clone(),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "TTS model not loaded. Set tts_model in config.",
            )
            .into_response();
        }
    };

    if req.input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "input text is required").into_response();
    }
    if req.input.len() > 4096 {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("input too long: {} chars (max 4096)", req.input.len()),
        )
        .into_response();
    }
    if !(0.25..=4.0).contains(&req.speed) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("speed must be between 0.25 and 4.0, got {}", req.speed),
        )
        .into_response();
    }

    match tts.synthesize(&req).await {
        Ok(result) => {
            let content_type = match result.format.as_str() {
                "pcm" => "audio/pcm",
                _ => "audio/wav",
            };
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, content_type)],
                result.audio,
            )
                .into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("TTS synthesis error: {e}"),
        )
        .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Embeddings: /v1/embeddings (OpenAI-compatible)
// ---------------------------------------------------------------------------

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::inference::EmbeddingsRequest>,
) -> impl IntoResponse {
    // Find a route for this model
    let route = match state
        .router
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .select(&req.model)
        .cloned()
    {
        Some(r) => r,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("No provider configured for model '{}'", req.model),
            )
            .into_response();
        }
    };

    let provider = match state.providers.get(route.provider, &route.base_url) {
        Some(p) => p,
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Provider '{}' not available", route.provider),
            )
            .into_response();
        }
    };

    match provider.embeddings(&req).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Embeddings error: {e}"),
        )
        .into_response(),
    }
}
