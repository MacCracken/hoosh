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
}

/// The router manages provider selection and fallback.
pub struct Router {
    routes: Vec<ProviderRoute>,
    strategy: RoutingStrategy,
    round_robin_index: std::sync::atomic::AtomicUsize,
    latencies: Arc<DashMap<(ProviderType, String), AtomicU64>>,
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
        }
    }

    /// Select the best provider for a given model.
    pub fn select(&self, model: &str) -> Option<&ProviderRoute> {
        let candidates: Vec<&ProviderRoute> = self
            .routes
            .iter()
            .filter(|r| r.enabled && self.matches_model(r, model))
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
}
