//! Axum HTTP server exposing the OpenAI-compatible API.

mod audio;
mod handlers;
mod types;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

use crate::budget::{TokenBudget, TokenPool};
use crate::cache::{CacheConfig, ResponseCache};
use crate::cost::CostTracker;
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
    #[cfg(feature = "tools")]
    pub mcp_bridge: Arc<crate::tools::McpBridge>,
    pub compactor: crate::context::compactor::ContextCompactor,
    pub model_registry: crate::provider::metadata::ModelMetadataRegistry,
    pub retry_manager: crate::provider::retry::RetryManager,
}

/// Server configuration.
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub routes: Vec<ProviderRoute>,
    pub strategy: RoutingStrategy,
    pub cache_config: CacheConfig,
    pub budget_pools: Vec<TokenPool>,
    pub whisper_model: Option<String>,
    pub tts_model: Option<String>,
    pub audit_enabled: bool,
    pub audit_signing_key: Option<String>,
    pub audit_max_entries: usize,
    pub auth_tokens: Vec<String>,
    pub otlp_endpoint: Option<String>,
    pub telemetry_service_name: String,
    pub health_check_interval_secs: u64,
    pub config_path: Option<String>,
    pub context_config: crate::config::ContextSection,
    pub retry_config: crate::provider::retry::RetryConfig,
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
            context_config: crate::config::ContextSection::default(),
            retry_config: crate::provider::retry::RetryConfig::default(),
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

    let event_bus = Arc::new(crate::events::new_event_bus());
    let inference_queue = Arc::new(crate::queue::InferenceQueue::new());

    #[cfg(feature = "tools")]
    let mcp_bridge = {
        let bridge = crate::tools::McpBridge::new();
        tracing::info!("MCP tool bridge enabled ({} tools)", bridge.tool_count());
        Arc::new(bridge)
    };

    let health_map = crate::health::new_health_map();
    let (eviction_tx, eviction_rx) = tokio::sync::mpsc::unbounded_channel();
    let heartbeat = Arc::new(majra::heartbeat::ConcurrentHeartbeatTracker::new(
        majra::heartbeat::HeartbeatConfig {
            suspect_after: std::time::Duration::from_secs(30),
            offline_after: std::time::Duration::from_secs(90),
            eviction_policy: Some(majra::heartbeat::EvictionPolicy {
                offline_cycles: 5,
                eviction_tx: Some(eviction_tx),
            }),
        },
    ));
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

    let compactor = crate::context::compactor::ContextCompactor::new(
        config.context_config.compaction_threshold,
        config.context_config.keep_last_messages,
        config.context_config.enabled,
    );

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
        #[cfg(feature = "tools")]
        mcp_bridge,
        compactor,
        model_registry: crate::provider::metadata::ModelMetadataRegistry::new(),
        retry_manager: crate::provider::retry::RetryManager::new(config.retry_config),
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
            Some(eviction_rx),
        );
        tracing::info!(
            "background health checker started (interval: {}s)",
            health_interval
        );
    }

    // API routes with 1MB body limit
    let api_routes = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/health", get(handlers::health))
        .route("/v1/health/providers", get(handlers::health_providers))
        .route("/v1/health/heartbeat", get(handlers::health_heartbeat))
        .route("/v1/tokens/check", post(handlers::tokens_check))
        .route("/v1/tokens/reserve", post(handlers::tokens_reserve))
        .route("/v1/tokens/report", post(handlers::tokens_report))
        .route("/v1/tokens/pools", get(handlers::tokens_pools))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/costs", get(handlers::costs_get))
        .route("/v1/costs/reset", post(handlers::costs_reset))
        .route("/v1/audit", get(handlers::audit_log))
        .route("/v1/admin/reload", post(handlers::admin_reload))
        .route("/v1/queue/status", get(handlers::queue_status))
        .route("/v1/cache/stats", get(handlers::cache_stats))
        .layer(DefaultBodyLimit::max(1024 * 1024));

    #[allow(unused_mut)]
    let mut app = api_routes.route("/metrics", get(handlers::prometheus_metrics));

    #[cfg(feature = "tools")]
    {
        app = app
            .route("/v1/tools/list", post(handlers::tools_list))
            .route("/v1/tools/call", post(handlers::tools_call));
    }

    #[cfg(feature = "whisper")]
    {
        app = app.route("/v1/audio/transcriptions", post(audio::transcribe));
    }
    #[cfg(feature = "piper")]
    {
        app = app.route("/v1/audio/speech", post(audio::text_to_speech));
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

    #[cfg(unix)]
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

fn reload_config(state: &Arc<AppState>, config_path: &str) {
    match crate::config::HooshConfig::load(config_path) {
        Ok(config) => {
            let routes = config.routes();
            let strategy: RoutingStrategy = config.server.strategy.into();

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
