//! Bearer token authentication middleware.

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

use crate::server::AppState;

/// Pre-computed SHA-256 digest of an auth token for constant-time comparison.
pub type TokenDigest = [u8; 32];

/// Hash a token for storage (called once at startup per configured token).
pub fn hash_token(token: &str) -> TokenDigest {
    use sha2::{Digest, Sha256};
    Sha256::digest(token.as_bytes()).into()
}

/// Constant-time comparison of a provided token against a pre-hashed digest.
/// Hashes the provided token once, then does fixed-length XOR comparison.
fn verify_token(provided: &[u8], digest: &TokenDigest) -> bool {
    use sha2::{Digest, Sha256};
    let provided_hash: [u8; 32] = Sha256::digest(provided).into();
    provided_hash
        .iter()
        .zip(digest.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Axum middleware that validates `Authorization: Bearer <token>` headers.
///
/// If no tokens are configured (empty list), authentication is disabled and all
/// requests pass through. Otherwise, the request must carry a valid bearer token
/// or a 401 response with an OpenAI-compatible error body is returned.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let digests = &state.auth_token_digests;

    // No tokens configured — auth disabled, pass through.
    if digests.is_empty() {
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match auth_header {
        Some(token) if digests.iter().any(|d| verify_token(token.as_bytes(), d)) => {
            next.run(request).await
        }
        _ => {
            let body = serde_json::json!({
                "error": {
                    "message": "Invalid or missing bearer token.",
                    "type": "error",
                    "code": "unauthorized"
                }
            });
            (StatusCode::UNAUTHORIZED, Json(body)).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::{Router, body::Body, middleware, routing::get};
    use tower::ServiceExt;

    fn test_state(tokens: Vec<String>) -> Arc<AppState> {
        use crate::budget::TokenBudget;
        use crate::cache::{CacheConfig, ResponseCache};
        use crate::cost::CostTracker;
        use crate::middleware::rate_limit::RateLimitRegistry;
        use crate::provider::ProviderRegistry;
        use crate::router::RoutingStrategy;

        Arc::new(AppState {
            router: std::sync::RwLock::new(crate::router::Router::new(
                vec![],
                RoutingStrategy::Priority,
            )),
            config_path: None,
            cache: ResponseCache::new(CacheConfig::default()),
            budget: std::sync::Mutex::new(TokenBudget::new()),
            providers: ProviderRegistry::new(),
            cost_tracker: Arc::new(CostTracker::new()),
            audit: None,
            auth_token_digests: tokens.iter().map(|t| hash_token(t)).collect(),
            rate_limiter: Arc::new(RateLimitRegistry::new()),
            event_bus: Arc::new(crate::events::new_event_bus()),
            inference_queue: Arc::new(crate::queue::InferenceQueue::new()),
            health_map: crate::health::new_health_map(),
            heartbeat: Arc::new(majra::heartbeat::ConcurrentHeartbeatTracker::default()),
            #[cfg(feature = "whisper")]
            whisper: None,
            #[cfg(feature = "piper")]
            tts: None,
        })
    }

    async fn handler() -> &'static str {
        "ok"
    }

    fn app(tokens: Vec<String>) -> Router {
        let state = test_state(tokens);
        Router::new()
            .route("/test", get(handler))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state)
    }

    #[tokio::test]
    async fn empty_tokens_passes_all() {
        let app = app(vec![]);
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn valid_token_passes() {
        let app = app(vec!["secret-token".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", "Bearer secret-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_token_returns_401() {
        let app = app(vec!["secret-token".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_header_returns_401() {
        let app = app(vec!["secret-token".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn non_bearer_scheme_returns_401() {
        let app = app(vec!["secret-token".to_string()]);
        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", "Basic secret-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
