//! Background health checker — periodic provider health monitoring with automatic failover.

use std::sync::Arc;

use dashmap::DashMap;

use crate::provider::{ProviderRegistry, ProviderType};
use crate::router::ProviderRoute;

/// Health state for a single provider.
#[derive(Debug, Clone)]
pub struct ProviderHealthState {
    pub is_healthy: bool,
    pub last_check: std::time::Instant,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

/// Shared health status map, used by the router for filtering.
pub type HealthMap = Arc<DashMap<(ProviderType, String), ProviderHealthState>>;

/// Consecutive failures before marking a provider unhealthy.
const UNHEALTHY_THRESHOLD: u32 = 3;

/// Create a new empty health map.
pub fn new_health_map() -> HealthMap {
    Arc::new(DashMap::new())
}

/// Handle a health check failure (shared by Ok(false) and Err branches).
fn handle_check_failure(
    health_map: &HealthMap,
    event_bus: &crate::events::EventBus,
    key: &(ProviderType, String),
    was_healthy: bool,
    error_msg: String,
) {
    let failures = health_map
        .get(key)
        .map(|s| s.consecutive_failures + 1)
        .unwrap_or(1);
    let new_healthy = failures < UNHEALTHY_THRESHOLD;
    health_map.insert(
        key.clone(),
        ProviderHealthState {
            is_healthy: new_healthy,
            last_check: std::time::Instant::now(),
            consecutive_failures: failures,
            last_error: Some(error_msg),
        },
    );
    if was_healthy && !new_healthy {
        event_bus.publish(
            crate::events::topics::HEALTH,
            crate::events::ProviderEvent::HealthChanged {
                provider: key.0.to_string(),
                base_url: key.1.clone(),
                healthy: false,
            },
        );
    }
    if failures >= UNHEALTHY_THRESHOLD {
        tracing::warn!(
            "provider {}@{} marked unhealthy after {} failures",
            key.0,
            key.1,
            failures
        );
    }
}

/// Spawn a background task that periodically checks all provider health.
/// Returns a JoinHandle that can be used to cancel the task.
pub fn spawn_health_checker(
    providers: Arc<ProviderRegistry>,
    routes: Vec<ProviderRoute>,
    health_map: HealthMap,
    interval_secs: u64,
    event_bus: Arc<crate::events::EventBus>,
    heartbeat: Arc<majra::heartbeat::ConcurrentHeartbeatTracker>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        // Don't run immediately — let providers warm up
        interval.tick().await;

        loop {
            interval.tick().await;

            for route in &routes {
                if !route.enabled {
                    continue;
                }
                let key = (route.provider, route.base_url.clone());
                let node_id = format!("{}:{}", route.provider, route.base_url);

                if let Some(provider) = providers.get(route.provider, &route.base_url) {
                    let was_healthy = health_map.get(&key).map(|s| s.is_healthy).unwrap_or(true);

                    match provider.health_check().await {
                        Ok(true) => {
                            // Send heartbeat to majra tracker
                            heartbeat.heartbeat(&node_id);

                            health_map.insert(
                                key.clone(),
                                ProviderHealthState {
                                    is_healthy: true,
                                    last_check: std::time::Instant::now(),
                                    consecutive_failures: 0,
                                    last_error: None,
                                },
                            );
                            if !was_healthy {
                                event_bus.publish(
                                    crate::events::topics::HEALTH,
                                    crate::events::ProviderEvent::HealthChanged {
                                        provider: key.0.to_string(),
                                        base_url: key.1.clone(),
                                        healthy: true,
                                    },
                                );
                            }
                        }
                        Ok(false) => {
                            handle_check_failure(
                                &health_map,
                                &event_bus,
                                &key,
                                was_healthy,
                                "health check returned false".into(),
                            );
                        }
                        Err(e) => {
                            handle_check_failure(
                                &health_map,
                                &event_bus,
                                &key,
                                was_healthy,
                                e.to_string(),
                            );
                        }
                    }
                }
            }

            // Sweep heartbeat tracker to update Online→Suspect→Offline transitions
            let transitions = heartbeat.update_statuses();
            for (node_id, status) in &transitions {
                tracing::info!("heartbeat: {} → {}", node_id, status);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_map_starts_empty() {
        let map = new_health_map();
        assert!(map.is_empty());
    }

    #[test]
    fn provider_health_state_fields() {
        let state = ProviderHealthState {
            is_healthy: true,
            last_check: std::time::Instant::now(),
            consecutive_failures: 0,
            last_error: None,
        };
        assert!(state.is_healthy);
        assert_eq!(state.consecutive_failures, 0);
        assert!(state.last_error.is_none());

        let state2 = ProviderHealthState {
            is_healthy: false,
            last_check: std::time::Instant::now(),
            consecutive_failures: 3,
            last_error: Some("connection refused".into()),
        };
        assert!(!state2.is_healthy);
        assert_eq!(state2.consecutive_failures, 3);
        assert_eq!(state2.last_error.as_deref(), Some("connection refused"));
    }

    #[test]
    fn health_map_insert_and_lookup() {
        let map = new_health_map();
        let key = (ProviderType::Ollama, "http://localhost:11434".to_string());
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 0,
                last_error: None,
            },
        );
        assert!(!map.is_empty());
        let entry = map.get(&key).unwrap();
        assert!(entry.is_healthy);
    }
}
