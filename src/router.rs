//! Request routing: provider selection, load balancing, fallback.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderType;

/// Routing strategy for selecting a provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Use the first healthy provider (ordered by priority).
    #[default]
    Priority,
    /// Round-robin across healthy providers.
    RoundRobin,
    /// Route to the provider with lowest latency.
    LowestLatency,
    /// Route to a specific provider (bypass routing).
    Direct,
}

/// Provider routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRoute {
    /// Provider type.
    pub provider: ProviderType,
    /// Priority (lower = preferred).
    pub priority: u32,
    /// Model patterns this provider handles (e.g. "llama*", "gpt-*").
    pub model_patterns: Vec<String>,
    /// Whether this provider is enabled.
    pub enabled: bool,
    /// Base URL for this provider.
    pub base_url: String,
    /// API key (for remote providers). Resolved from env var if prefixed with `$`.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Maximum tokens this provider supports per request.
    #[serde(default)]
    pub max_tokens_limit: Option<u32>,
    /// Maximum requests per minute for this provider.
    #[serde(default)]
    pub rate_limit_rpm: Option<u32>,
    /// TLS configuration for this provider.
    #[serde(skip)]
    pub tls_config: Option<crate::provider::TlsConfig>,
}

/// The router manages provider selection and fallback.
pub struct Router {
    routes: Vec<ProviderRoute>,
    strategy: RoutingStrategy,
    round_robin_index: std::sync::atomic::AtomicUsize,
    latencies: Arc<DashMap<(ProviderType, String), AtomicU64>>,
    health_status: Option<crate::health::HealthMap>,
}

impl Router {
    /// Create a new router with the given routes and strategy.
    pub fn new(mut routes: Vec<ProviderRoute>, strategy: RoutingStrategy) -> Self {
        routes.sort_by_key(|r| r.priority);
        Self {
            routes,
            strategy,
            round_robin_index: std::sync::atomic::AtomicUsize::new(0),
            latencies: Arc::new(DashMap::new()),
            health_status: None,
        }
    }

    /// Set the health map for background health check filtering.
    pub fn set_health_map(&mut self, map: crate::health::HealthMap) {
        self.health_status = Some(map);
    }

    /// Check if a provider is considered healthy.
    /// A provider is healthy if:
    /// - No health map is set (backward compatible), OR
    /// - Not in the health map (not checked yet = assume healthy), OR
    /// - Entry shows `is_healthy: true`
    fn is_provider_healthy(&self, provider: ProviderType, base_url: &str) -> bool {
        match &self.health_status {
            None => true,
            Some(map) => {
                let key = (provider, base_url.to_string());
                match map.get(&key) {
                    None => true,
                    Some(state) => state.is_healthy,
                }
            }
        }
    }

    /// Select the best provider for a given model.
    pub fn select(&self, model: &str) -> Option<&ProviderRoute> {
        let candidates: Vec<&ProviderRoute> = self
            .routes
            .iter()
            .filter(|r| {
                r.enabled
                    && self.matches_model(r, model)
                    && self.is_provider_healthy(r.provider, &r.base_url)
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        match self.strategy {
            RoutingStrategy::Priority | RoutingStrategy::Direct => candidates.first().copied(),
            RoutingStrategy::LowestLatency => {
                let mut sorted = candidates;
                sorted.sort_by_key(|r| {
                    let key = (r.provider, r.base_url.clone());
                    match self.latencies.get(&key) {
                        Some(entry) => entry.value().load(Ordering::Relaxed),
                        None => u64::MAX, // no data → deprioritize
                    }
                });
                sorted.first().copied()
            }
            RoutingStrategy::RoundRobin => {
                let idx = self
                    .round_robin_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(candidates[idx % candidates.len()])
            }
        }
    }

    /// Report observed latency for a provider, updating an exponential moving average.
    pub fn report_latency(&self, provider: ProviderType, base_url: &str, latency_ms: u64) {
        let key = (provider, base_url.to_string());
        self.latencies
            .entry(key)
            .and_modify(|existing| {
                let old = existing.load(Ordering::Relaxed);
                let new_avg = (old * 7 + latency_ms * 3) / 10;
                existing.store(new_avg, Ordering::Relaxed);
            })
            .or_insert_with(|| AtomicU64::new(latency_ms));
    }

    /// Replace routes and strategy atomically.
    pub fn reload(&mut self, mut routes: Vec<ProviderRoute>, strategy: RoutingStrategy) {
        routes.sort_by_key(|r| r.priority);
        self.routes = routes;
        self.strategy = strategy;
        self.round_robin_index
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// All configured routes.
    pub fn routes(&self) -> &[ProviderRoute] {
        &self.routes
    }

    /// Check if a route's model patterns match the requested model.
    fn matches_model(&self, route: &ProviderRoute, model: &str) -> bool {
        if route.model_patterns.is_empty() {
            return true; // wildcard — accepts all models
        }
        route.model_patterns.iter().any(|pattern| {
            if pattern.contains('*') {
                let prefix = pattern.trim_end_matches('*');
                model.starts_with(prefix)
            } else {
                model == pattern
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(provider: ProviderType, priority: u32, patterns: Vec<&str>) -> ProviderRoute {
        ProviderRoute {
            provider,
            priority,
            model_patterns: patterns.into_iter().map(String::from).collect(),
            enabled: true,
            base_url: "http://localhost".into(),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
        }
    }

    #[test]
    fn priority_routing() {
        let routes = vec![
            make_route(ProviderType::Ollama, 1, vec!["llama*"]),
            make_route(ProviderType::OpenAi, 2, vec!["gpt-*"]),
        ];
        let router = Router::new(routes, RoutingStrategy::Priority);

        let selected = router.select("llama3").unwrap();
        assert_eq!(selected.provider, ProviderType::Ollama);

        let selected = router.select("gpt-4o").unwrap();
        assert_eq!(selected.provider, ProviderType::OpenAi);
    }

    #[test]
    fn wildcard_route() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec![])];
        let router = Router::new(routes, RoutingStrategy::Priority);
        assert!(router.select("anything").is_some());
    }

    #[test]
    fn no_matching_provider() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec!["llama*"])];
        let router = Router::new(routes, RoutingStrategy::Priority);
        assert!(router.select("gpt-4o").is_none());
    }

    #[test]
    fn disabled_route_skipped() {
        let mut route = make_route(ProviderType::Ollama, 1, vec![]);
        route.enabled = false;
        let router = Router::new(vec![route], RoutingStrategy::Priority);
        assert!(router.select("llama3").is_none());
    }

    #[test]
    fn round_robin() {
        let routes = vec![
            make_route(ProviderType::Ollama, 1, vec![]),
            make_route(ProviderType::LlamaCpp, 1, vec![]),
        ];
        let router = Router::new(routes, RoutingStrategy::RoundRobin);

        let first = router.select("llama3").unwrap().provider;
        let second = router.select("llama3").unwrap().provider;
        assert_ne!(first, second);
    }

    #[test]
    fn routing_strategy_default() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::Priority);
    }

    fn make_route_with_url(
        provider: ProviderType,
        priority: u32,
        patterns: Vec<&str>,
        base_url: &str,
    ) -> ProviderRoute {
        ProviderRoute {
            provider,
            priority,
            model_patterns: patterns.into_iter().map(String::from).collect(),
            enabled: true,
            base_url: base_url.to_string(),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
        }
    }

    #[test]
    fn report_latency_records_values() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec![])];
        let router = Router::new(routes, RoutingStrategy::LowestLatency);
        let key = (ProviderType::Ollama, "http://localhost".to_string());

        router.report_latency(ProviderType::Ollama, "http://localhost", 100);
        {
            let latency = router.latencies.get(&key).unwrap();
            assert_eq!(latency.value().load(Ordering::Relaxed), 100);
        }

        // Second report should compute EMA: (100 * 7 + 200 * 3) / 10 = 130
        router.report_latency(ProviderType::Ollama, "http://localhost", 200);
        {
            let latency = router.latencies.get(&key).unwrap();
            assert_eq!(latency.value().load(Ordering::Relaxed), 130);
        }
    }

    #[test]
    fn lowest_latency_picks_fastest() {
        let routes = vec![
            make_route_with_url(ProviderType::Ollama, 2, vec![], "http://ollama"),
            make_route_with_url(ProviderType::LlamaCpp, 1, vec![], "http://llamacpp"),
        ];
        let router = Router::new(routes, RoutingStrategy::LowestLatency);

        // LlamaCpp has lower priority number (higher priority) but higher latency
        router.report_latency(ProviderType::LlamaCpp, "http://llamacpp", 500);
        router.report_latency(ProviderType::Ollama, "http://ollama", 50);

        let selected = router.select("any-model").unwrap();
        assert_eq!(selected.provider, ProviderType::Ollama);
    }

    #[test]
    fn lowest_latency_deprioritizes_unknown() {
        let routes = vec![
            make_route_with_url(ProviderType::Ollama, 1, vec![], "http://ollama"),
            make_route_with_url(ProviderType::LlamaCpp, 2, vec![], "http://llamacpp"),
        ];
        let router = Router::new(routes, RoutingStrategy::LowestLatency);

        // Only report latency for LlamaCpp — Ollama has no data and should go last
        router.report_latency(ProviderType::LlamaCpp, "http://llamacpp", 100);

        let selected = router.select("any-model").unwrap();
        assert_eq!(selected.provider, ProviderType::LlamaCpp);
    }

    #[test]
    fn select_no_health_map_allows_all() {
        // Backward compatibility: no health map set means all providers allowed
        let routes = vec![
            make_route(ProviderType::Ollama, 1, vec![]),
            make_route(ProviderType::LlamaCpp, 2, vec![]),
        ];
        let router = Router::new(routes, RoutingStrategy::Priority);
        assert!(router.select("any-model").is_some());
        assert_eq!(router.select("any-model").unwrap().provider, ProviderType::Ollama);
    }

    #[test]
    fn select_filters_unhealthy_providers() {
        let routes = vec![
            make_route(ProviderType::Ollama, 1, vec![]),
            make_route(ProviderType::LlamaCpp, 2, vec![]),
        ];
        let mut router = Router::new(routes, RoutingStrategy::Priority);

        let health_map = crate::health::new_health_map();
        // Mark Ollama as unhealthy
        health_map.insert(
            (ProviderType::Ollama, "http://localhost".to_string()),
            crate::health::ProviderHealthState {
                is_healthy: false,
                last_check: std::time::Instant::now(),
                consecutive_failures: 3,
                last_error: Some("connection refused".into()),
            },
        );
        router.set_health_map(health_map);

        // Should skip Ollama and pick LlamaCpp
        let selected = router.select("any-model").unwrap();
        assert_eq!(selected.provider, ProviderType::LlamaCpp);
    }

    #[test]
    fn select_unchecked_provider_assumed_healthy() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec![])];
        let mut router = Router::new(routes, RoutingStrategy::Priority);

        // Set health map but don't add any entries — unchecked providers assumed healthy
        let health_map = crate::health::new_health_map();
        router.set_health_map(health_map);

        assert!(router.select("any-model").is_some());
    }

    #[test]
    fn select_all_unhealthy_returns_none() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec![])];
        let mut router = Router::new(routes, RoutingStrategy::Priority);

        let health_map = crate::health::new_health_map();
        health_map.insert(
            (ProviderType::Ollama, "http://localhost".to_string()),
            crate::health::ProviderHealthState {
                is_healthy: false,
                last_check: std::time::Instant::now(),
                consecutive_failures: 5,
                last_error: Some("down".into()),
            },
        );
        router.set_health_map(health_map);

        assert!(router.select("any-model").is_none());
    }

    #[test]
    fn reload_changes_routes() {
        let routes = vec![make_route(ProviderType::Ollama, 1, vec!["llama*"])];
        let mut router = Router::new(routes, RoutingStrategy::Priority);
        assert_eq!(router.routes().len(), 1);
        assert!(router.select("llama3").is_some());
        assert!(router.select("gpt-4o").is_none());

        // Reload with a different set of routes
        let new_routes = vec![
            make_route(ProviderType::OpenAi, 1, vec!["gpt-*"]),
            make_route(ProviderType::LlamaCpp, 2, vec!["gguf-*"]),
        ];
        router.reload(new_routes, RoutingStrategy::RoundRobin);
        assert_eq!(router.routes().len(), 2);
        assert!(router.select("gpt-4o").is_some());
        assert!(router.select("llama3").is_none());
    }

    #[test]
    fn reload_resets_round_robin_index() {
        let routes = vec![
            make_route(ProviderType::Ollama, 1, vec![]),
            make_route(ProviderType::LlamaCpp, 1, vec![]),
        ];
        let mut router = Router::new(routes, RoutingStrategy::RoundRobin);

        // Advance the round-robin index
        let _ = router.select("any");
        let _ = router.select("any");
        let _ = router.select("any");

        // Reload — index should reset to 0
        let new_routes = vec![
            make_route(ProviderType::Ollama, 1, vec![]),
            make_route(ProviderType::LlamaCpp, 1, vec![]),
        ];
        router.reload(new_routes, RoutingStrategy::RoundRobin);

        // After reload, first select should pick index 0
        let first = router.select("any").unwrap().provider;
        let second = router.select("any").unwrap().provider;
        assert_ne!(first, second, "round-robin should alternate after reload");
    }

    #[test]
    fn rwlock_concurrent_reads() {
        use std::sync::{Arc, RwLock};

        let routes = vec![make_route(ProviderType::Ollama, 1, vec![])];
        let router = Arc::new(RwLock::new(Router::new(routes, RoutingStrategy::Priority)));

        // Multiple concurrent readers should not block
        let r1 = router.read().unwrap();
        let r2 = router.read().unwrap();
        assert_eq!(r1.routes().len(), 1);
        assert_eq!(r2.routes().len(), 1);
    }
}
