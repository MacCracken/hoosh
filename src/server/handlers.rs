//! HTTP route handlers for the OpenAI-compatible API.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, KeepAlive, Sse},
    },
};

use crate::budget::TokenPool;
use crate::inference::{InferenceRequest, Message, MessageContent, Role};

use super::AppState;
use super::types::*;

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4();
    tracing::info!(request_id = %request_id, model = %req.model, "chat_completions");

    // Input validation
    if let Some(resp) = validate_chat_request(&req) {
        return resp;
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
    let mut inference_req = InferenceRequest {
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
                    "tool" => Role::Tool,
                    _ => Role::User,
                },
                content: MessageContent::Text(m.content.clone()),
                tool_call_id: m.tool_call_id.clone(),
                tool_calls: m.tool_calls.clone(),
            })
            .collect(),
        max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
    };

    // Context compaction: truncate if approaching model's context window.
    let counter = crate::context::tokens::ProviderTokenCounter::for_provider(route.provider);
    if let Some(result) = state.compactor.compact(
        &inference_req.model,
        &inference_req.messages,
        &state.model_registry,
        &counter,
    ) {
        tracing::info!(
            request_id = %request_id,
            original_tokens = result.original_tokens,
            compacted_tokens = result.compacted_tokens,
            messages_dropped = result.messages_dropped,
            "context compacted"
        );
        inference_req.messages = result.messages;
    }

    // Token budget: validate pool exists, then atomically reserve.
    // Estimate = input tokens (from messages) + output budget (max_tokens or default).
    let input_estimate =
        crate::context::tokens::TokenCounter::count_messages(&counter, &inference_req.messages);
    let output_budget = max_tokens.unwrap_or(1024);
    let estimated_tokens = (input_estimate.saturating_add(output_budget)) as u64;
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
        return handle_streaming(
            state,
            inference_req,
            &route,
            provider,
            pool_name,
            estimated_tokens,
            req.model.clone(),
        )
        .await;
    }

    // Non-streaming
    handle_non_streaming(
        state,
        inference_req,
        &route,
        provider,
        pool_name,
        estimated_tokens,
        req.model.clone(),
    )
    .await
}

fn validate_chat_request(req: &ChatRequest) -> Option<axum::response::Response> {
    if req.messages.len() > 256 {
        return Some(
            error_response(
                StatusCode::BAD_REQUEST,
                format!("Too many messages: {} (max 256)", req.messages.len()),
            )
            .into_response(),
        );
    }
    if req.messages.is_empty() {
        return Some(
            error_response(StatusCode::BAD_REQUEST, "messages array is empty").into_response(),
        );
    }
    if req.model.is_empty() {
        return Some(
            error_response(StatusCode::BAD_REQUEST, "model field is required").into_response(),
        );
    }
    if req.model.len() > 256
        || req
            .model
            .bytes()
            .any(|b| b < 0x20 || b == b'\\' || b == b'"')
    {
        return Some(error_response(StatusCode::BAD_REQUEST, "invalid model name").into_response());
    }
    if let Some(temp) = req.temperature
        && !(0.0..=2.0).contains(&temp)
    {
        return Some(
            error_response(
                StatusCode::BAD_REQUEST,
                format!("temperature must be between 0.0 and 2.0, got {temp}"),
            )
            .into_response(),
        );
    }
    if let Some(tp) = req.top_p
        && !(0.0..=1.0).contains(&tp)
    {
        return Some(
            error_response(
                StatusCode::BAD_REQUEST,
                format!("top_p must be between 0.0 and 1.0, got {tp}"),
            )
            .into_response(),
        );
    }
    None
}

async fn handle_streaming(
    state: Arc<AppState>,
    inference_req: InferenceRequest,
    route: &crate::router::ProviderRoute,
    provider: Arc<dyn crate::provider::LlmProvider>,
    pool_name: String,
    estimated_tokens: u64,
    model: String,
) -> axum::response::Response {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let rx = match provider.infer_stream(inference_req).await {
        Ok(rx) => {
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
            let mut budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
            budget.report(&pool_name, estimated_tokens, 0);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, {
                tracing::error!("inference error: {e}");
                "Inference request failed".to_string()
            })
            .into_response();
        }
    };

    let token_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let token_count_clone = token_count.clone();

    let budget_guard = StreamBudgetGuard {
        state: state.clone(),
        pool: pool_name,
        estimated: estimated_tokens,
        actual: token_count_clone,
        provider: route.provider.to_string(),
        model: model.clone(),
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

    Sse::new(s)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    inference_req: InferenceRequest,
    route: &crate::router::ProviderRoute,
    provider: Arc<dyn crate::provider::LlmProvider>,
    pool_name: String,
    estimated_tokens: u64,
    req_model: String,
) -> axum::response::Response {
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

            // Audit event
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
                        tool_calls: result.tool_calls.clone(),
                    },
                    finish_reason: if result.tool_calls.is_empty() {
                        "stop"
                    } else {
                        "tool_calls"
                    },
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
                    model: req_model.clone(),
                    error: e.to_string(),
                },
            );

            // Audit event
            if let Some(ref audit) = state.audit {
                audit.record(
                    "inference.error",
                    "error",
                    &{
                        tracing::error!("inference error: {e}");
                        "Inference request failed".to_string()
                    },
                    Some(&route.provider.to_string()),
                    Some(&req_model),
                    None,
                );
            }

            // Record error metrics
            crate::metrics::record_request(
                &route.provider.to_string(),
                &req_model,
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

pub(crate) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let mut models = Vec::new();

    let routes: Vec<_> = state
        .router
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .routes()
        .to_vec();

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

pub(crate) async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
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

pub(crate) async fn health_providers(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<ProviderHealth>> {
    let mut results = Vec::new();

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
            if hs.is_healthy {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            }
        } else if let Some(provider) = state.providers.get(route.provider, &route.base_url) {
            match provider.health_check().await {
                Ok(true) => "healthy".to_string(),
                Ok(false) => "unhealthy".to_string(),
                Err(e) => {
                    tracing::warn!("health check error for {}: {e}", route.provider);
                    "error".to_string()
                }
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
            last_error: bg_state.as_ref().and_then(|s| {
                s.last_error
                    .as_ref()
                    .map(|_| "health check failed".to_string())
            }),
        });
    }

    Json(results)
}

pub(crate) async fn health_heartbeat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.heartbeat.fleet_stats();
    (StatusCode::OK, Json(stats)).into_response()
}

// ---------------------------------------------------------------------------
// Token budget
// ---------------------------------------------------------------------------

pub(crate) async fn tokens_check(
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

pub(crate) async fn tokens_reserve(
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

pub(crate) async fn tokens_report(
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

pub(crate) async fn tokens_pools(State(state): State<Arc<AppState>>) -> Json<Vec<TokenPool>> {
    let budget = state.budget.lock().unwrap_or_else(|e| e.into_inner());
    let pools: Vec<TokenPool> = budget.pools().values().cloned().collect();
    Json(pools)
}

// ---------------------------------------------------------------------------
// Cost tracking
// ---------------------------------------------------------------------------

pub(crate) async fn costs_get(State(state): State<Arc<AppState>>) -> Json<CostsResponse> {
    let (records, total_cost_usd) = state.cost_tracker.all_with_total();
    Json(CostsResponse {
        records,
        total_cost_usd,
    })
}

pub(crate) async fn costs_reset(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
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
// Audit log
// ---------------------------------------------------------------------------

pub(crate) async fn audit_log(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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
// Admin & metrics
// ---------------------------------------------------------------------------

pub(crate) async fn admin_reload(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(path) = &state.config_path {
        super::reload_config(&state, path);
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

pub(crate) async fn queue_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "queued": state.inference_queue.len(),
        })),
    )
        .into_response()
}

pub(crate) async fn cache_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!(state.cache.stats()))).into_response()
}

pub(crate) async fn prometheus_metrics() -> impl IntoResponse {
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
// Embeddings
// ---------------------------------------------------------------------------

pub(crate) async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::inference::EmbeddingsRequest>,
) -> impl IntoResponse {
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

// ---------------------------------------------------------------------------
// MCP tools
// ---------------------------------------------------------------------------

#[cfg(feature = "tools")]
pub(crate) async fn tools_list(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = state.mcp_bridge.list_tools();
    (StatusCode::OK, Json(result)).into_response()
}

#[cfg(feature = "tools")]
pub(crate) async fn tools_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<super::types::ToolCallRequest>,
) -> impl IntoResponse {
    let (result, is_error) = state.mcp_bridge.call_tool(&req.name, req.arguments);
    let status = if is_error {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    (status, Json(result)).into_response()
}
