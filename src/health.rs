//! Background health checker — periodic provider health monitoring with automatic failover.
//!
//! Integrates with majra's heartbeat tracker for Online→Suspect→Offline FSM,
//! GPU telemetry forwarding, fleet statistics, and automatic eviction of
//! persistently offline providers.

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

/// Hardware state handle for the health checker's periodic GPU telemetry refresh.
#[cfg(feature = "hwaccel")]
pub struct HwHandle {
    state: Arc<crate::server::AppState>,
    refresh_interval_secs: u64,
}

#[cfg(feature = "hwaccel")]
impl HwHandle {
    pub fn new(state: Arc<crate::server::AppState>, refresh_interval_secs: u64) -> Self {
        Self {
            state,
            refresh_interval_secs,
        }
    }

    /// Refresh hardware state by re-detecting with disk cache, then return
    /// current GPU telemetry.
    fn refresh_and_collect(&self) -> Vec<majra::heartbeat::GpuTelemetry> {
        let fresh = crate::hardware::HardwareManager::from_cache(self.state.hw_cache_ttl);
        match self.state.hardware.write() {
            Ok(mut hw) => {
                *hw = fresh;
                hw.gpu_telemetry()
            }
            Err(e) => {
                tracing::warn!("hardware state lock poisoned, using fresh detection: {e}");
                fresh.gpu_telemetry()
            }
        }
    }

    /// Collect GPU telemetry from the current (cached) hardware state.
    fn collect(&self) -> Vec<majra::heartbeat::GpuTelemetry> {
        match self.state.hardware.read() {
            Ok(hw) => hw.gpu_telemetry(),
            Err(e) => {
                tracing::warn!("hardware state lock poisoned: {e}");
                Vec::new()
            }
        }
    }
}

/// Spawn a background task that periodically checks all provider health.
/// Returns a JoinHandle that can be used to cancel the task.
#[allow(clippy::too_many_arguments)]
pub fn spawn_health_checker(
    providers: Arc<ProviderRegistry>,
    routes: Vec<ProviderRoute>,
    health_map: HealthMap,
    interval_secs: u64,
    event_bus: Arc<crate::events::EventBus>,
    heartbeat: Arc<majra::heartbeat::ConcurrentHeartbeatTracker>,
    eviction_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    #[cfg(feature = "hwaccel")] hw_handle: HwHandle,
) -> tokio::task::JoinHandle<()> {
    // Spawn eviction handler if configured
    if let Some(mut rx) = eviction_rx {
        let health_map_evict = health_map.clone();
        let event_bus_evict = event_bus.clone();
        tokio::spawn(async move {
            while let Some(node_id) = rx.recv().await {
                tracing::warn!(
                    "heartbeat eviction: {} removed (persistently offline)",
                    node_id
                );
                // Parse node_id back to (ProviderType, base_url) and remove from health map
                if let Some((ptype_str, _base_url)) = node_id.split_once(':') {
                    // Try to find and remove matching entry
                    let keys_to_remove: Vec<_> = health_map_evict
                        .iter()
                        .filter(|entry| {
                            let (pt, url) = entry.key();
                            pt.to_string() == ptype_str || (format!("{}:{}", pt, url) == node_id)
                        })
                        .map(|entry| entry.key().clone())
                        .collect();
                    for key in &keys_to_remove {
                        health_map_evict.remove(key);
                        event_bus_evict.publish(
                            crate::events::topics::HEALTH,
                            crate::events::ProviderEvent::HealthChanged {
                                provider: key.0.to_string(),
                                base_url: key.1.clone(),
                                healthy: false,
                            },
                        );
                    }
                }
            }
        });
    }

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        // Don't run immediately — let providers warm up
        interval.tick().await;

        #[cfg(feature = "hwaccel")]
        let mut last_hw_refresh = std::time::Instant::now();

        loop {
            interval.tick().await;

            // Refresh hardware telemetry at the configured interval
            #[cfg(feature = "hwaccel")]
            let gpu_telemetry = {
                let refresh_due = hw_handle.refresh_interval_secs > 0
                    && last_hw_refresh.elapsed().as_secs() >= hw_handle.refresh_interval_secs;
                if refresh_due {
                    last_hw_refresh = std::time::Instant::now();
                    hw_handle.refresh_and_collect()
                } else {
                    hw_handle.collect()
                }
            };

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
                            // Send heartbeat with GPU telemetry when hwaccel is available
                            #[cfg(feature = "hwaccel")]
                            {
                                if route.provider.is_local() && !gpu_telemetry.is_empty() {
                                    let _ = heartbeat
                                        .heartbeat_with_telemetry(&node_id, gpu_telemetry.clone());
                                } else {
                                    let _ = heartbeat.heartbeat(&node_id);
                                }
                            }
                            #[cfg(not(feature = "hwaccel"))]
                            {
                                let _ = heartbeat.heartbeat(&node_id);
                            }

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

    /// Helper: create a health map and event bus for handle_check_failure tests.
    fn setup_failure_test() -> (
        HealthMap,
        std::sync::Arc<crate::events::EventBus>,
        (ProviderType, String),
    ) {
        let map = new_health_map();
        let bus = std::sync::Arc::new(crate::events::new_event_bus());
        let key = (ProviderType::Ollama, "http://localhost:11434".to_string());
        (map, bus, key)
    }

    #[test]
    fn handle_check_failure_first_failure_still_healthy() {
        let (map, bus, key) = setup_failure_test();

        handle_check_failure(&map, &bus, &key, true, "timeout".into());

        let entry = map.get(&key).unwrap();
        assert!(entry.is_healthy, "should remain healthy after 1 failure");
        assert_eq!(entry.consecutive_failures, 1);
        assert_eq!(entry.last_error.as_deref(), Some("timeout"));
    }

    #[test]
    fn handle_check_failure_second_failure_still_healthy() {
        let (map, bus, key) = setup_failure_test();

        // Simulate first failure already recorded
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 1,
                last_error: Some("first".into()),
            },
        );

        handle_check_failure(&map, &bus, &key, true, "second".into());

        let entry = map.get(&key).unwrap();
        assert!(entry.is_healthy, "should remain healthy after 2 failures");
        assert_eq!(entry.consecutive_failures, 2);
    }

    #[test]
    fn handle_check_failure_third_failure_becomes_unhealthy() {
        let (map, bus, key) = setup_failure_test();

        // Pre-seed with 2 consecutive failures (still healthy)
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 2,
                last_error: Some("prev".into()),
            },
        );

        handle_check_failure(&map, &bus, &key, true, "third strike".into());

        let entry = map.get(&key).unwrap();
        assert!(!entry.is_healthy, "should be unhealthy after 3 failures");
        assert_eq!(entry.consecutive_failures, 3);
        assert_eq!(entry.last_error.as_deref(), Some("third strike"));
    }

    #[test]
    fn handle_check_failure_publishes_event_on_healthy_to_unhealthy_transition() {
        let (map, bus, key) = setup_failure_test();
        let mut rx = bus.subscribe(crate::events::topics::HEALTH);

        // Pre-seed at threshold - 1 failures
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: UNHEALTHY_THRESHOLD - 1,
                last_error: None,
            },
        );

        // This call should transition healthy -> unhealthy and publish an event
        handle_check_failure(&map, &bus, &key, true, "fatal".into());

        let msg = rx
            .try_recv()
            .expect("expected a HealthChanged event to be published");
        match msg.payload {
            crate::events::ProviderEvent::HealthChanged {
                provider,
                base_url,
                healthy,
            } => {
                assert_eq!(provider, "ollama");
                assert_eq!(base_url, "http://localhost:11434");
                assert!(!healthy);
            }
            other => panic!("expected HealthChanged, got {:?}", other),
        }
    }

    #[test]
    fn handle_check_failure_no_event_when_already_unhealthy() {
        let (map, bus, key) = setup_failure_test();
        let mut rx = bus.subscribe(crate::events::topics::HEALTH);

        // Pre-seed as already unhealthy
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: false,
                last_check: std::time::Instant::now(),
                consecutive_failures: 5,
                last_error: Some("old".into()),
            },
        );

        // was_healthy = false, so no transition event should fire
        handle_check_failure(&map, &bus, &key, false, "still broken".into());

        assert!(
            rx.try_recv().is_err(),
            "no event should be published when provider was already unhealthy"
        );

        let entry = map.get(&key).unwrap();
        assert!(!entry.is_healthy);
        assert_eq!(entry.consecutive_failures, 6);
    }

    #[test]
    fn unhealthy_threshold_constant_is_three() {
        assert_eq!(
            UNHEALTHY_THRESHOLD, 3,
            "threshold should be 3 consecutive failures"
        );
    }

    #[test]
    fn handle_check_failure_on_missing_entry_starts_at_one() {
        // When there is no prior entry in the map, consecutive_failures should start at 1
        let (map, bus, key) = setup_failure_test();

        assert!(map.is_empty());
        handle_check_failure(&map, &bus, &key, true, "first ever".into());

        let entry = map.get(&key).unwrap();
        assert_eq!(entry.consecutive_failures, 1);
        assert!(entry.is_healthy, "1 < threshold, should still be healthy");
    }

    #[test]
    fn health_map_recovery_after_failures() {
        // Simulate: 3 failures (unhealthy), then manual recovery, then another failure
        let map = new_health_map();
        let bus = std::sync::Arc::new(crate::events::new_event_bus());
        let key = (ProviderType::LlamaCpp, "http://localhost:8080".to_string());

        // Drive to unhealthy via 3 consecutive failures
        for i in 0..3 {
            let was_healthy = map.get(&key).map(|s| s.is_healthy).unwrap_or(true);
            handle_check_failure(&map, &bus, &key, was_healthy, format!("fail {}", i + 1));
        }
        assert!(!map.get(&key).unwrap().is_healthy);
        assert_eq!(map.get(&key).unwrap().consecutive_failures, 3);

        // Simulate recovery (what spawn_health_checker does on Ok(true))
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 0,
                last_error: None,
            },
        );
        assert!(map.get(&key).unwrap().is_healthy);
        assert_eq!(map.get(&key).unwrap().consecutive_failures, 0);

        // One more failure after recovery — should be healthy again (only 1 failure)
        handle_check_failure(&map, &bus, &key, true, "post-recovery fail".into());
        let entry = map.get(&key).unwrap();
        assert!(entry.is_healthy);
        assert_eq!(entry.consecutive_failures, 1);
    }

    #[test]
    fn handle_check_failure_multiple_providers_independent() {
        let map = new_health_map();
        let bus = std::sync::Arc::new(crate::events::new_event_bus());
        let key_a = (ProviderType::Ollama, "http://host-a:11434".to_string());
        let key_b = (ProviderType::OpenAi, "https://api.openai.com".to_string());

        // Fail provider A 3 times
        for _ in 0..3 {
            let was = map.get(&key_a).map(|s| s.is_healthy).unwrap_or(true);
            handle_check_failure(&map, &bus, &key_a, was, "err".into());
        }

        // Fail provider B once
        handle_check_failure(&map, &bus, &key_b, true, "err".into());

        assert!(
            !map.get(&key_a).unwrap().is_healthy,
            "A should be unhealthy"
        );
        assert!(
            map.get(&key_b).unwrap().is_healthy,
            "B should still be healthy"
        );
        assert_eq!(map.get(&key_a).unwrap().consecutive_failures, 3);
        assert_eq!(map.get(&key_b).unwrap().consecutive_failures, 1);
    }

    #[test]
    fn handle_check_failure_fourth_failure_stays_unhealthy() {
        let (map, bus, key) = setup_failure_test();

        // Pre-seed at 3 failures (already unhealthy)
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: false,
                last_check: std::time::Instant::now(),
                consecutive_failures: 3,
                last_error: Some("third".into()),
            },
        );

        handle_check_failure(&map, &bus, &key, false, "fourth strike".into());

        let entry = map.get(&key).unwrap();
        assert!(!entry.is_healthy, "should remain unhealthy");
        assert_eq!(entry.consecutive_failures, 4);
        assert_eq!(entry.last_error.as_deref(), Some("fourth strike"));
    }

    #[test]
    fn handle_check_failure_error_messages_are_preserved() {
        let (map, bus, key) = setup_failure_test();

        handle_check_failure(&map, &bus, &key, true, "timeout after 5s".into());
        assert_eq!(
            map.get(&key).unwrap().last_error.as_deref(),
            Some("timeout after 5s")
        );

        handle_check_failure(&map, &bus, &key, true, "connection refused".into());
        assert_eq!(
            map.get(&key).unwrap().last_error.as_deref(),
            Some("connection refused")
        );
    }

    #[test]
    fn health_map_multiple_entries_and_removal() {
        let map = new_health_map();
        let key1 = (ProviderType::Ollama, "http://host-a:11434".to_string());
        let key2 = (ProviderType::LlamaCpp, "http://host-b:8080".to_string());
        let key3 = (ProviderType::OpenAi, "https://api.openai.com".to_string());

        for key in [&key1, &key2, &key3] {
            map.insert(
                key.clone(),
                ProviderHealthState {
                    is_healthy: true,
                    last_check: std::time::Instant::now(),
                    consecutive_failures: 0,
                    last_error: None,
                },
            );
        }
        assert_eq!(map.len(), 3);

        map.remove(&key2);
        assert_eq!(map.len(), 2);
        assert!(map.get(&key2).is_none());
        assert!(map.get(&key1).is_some());
        assert!(map.get(&key3).is_some());
    }

    #[test]
    fn health_map_overwrite_entry() {
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
        assert!(map.get(&key).unwrap().is_healthy);

        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: false,
                last_check: std::time::Instant::now(),
                consecutive_failures: 5,
                last_error: Some("down".into()),
            },
        );
        assert!(!map.get(&key).unwrap().is_healthy);
        assert_eq!(map.get(&key).unwrap().consecutive_failures, 5);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn provider_health_state_clone_and_debug() {
        let state = ProviderHealthState {
            is_healthy: true,
            last_check: std::time::Instant::now(),
            consecutive_failures: 2,
            last_error: Some("err".into()),
        };
        let cloned = state.clone();
        assert_eq!(cloned.is_healthy, state.is_healthy);
        assert_eq!(cloned.consecutive_failures, state.consecutive_failures);
        assert_eq!(cloned.last_error, state.last_error);

        // Debug should not panic
        let debug = format!("{:?}", state);
        assert!(debug.contains("ProviderHealthState"));
    }

    #[test]
    fn handle_check_failure_exactly_at_threshold_boundary() {
        // Test that threshold - 1 failures keeps healthy, threshold makes unhealthy
        let (map, bus, key) = setup_failure_test();

        // Seed with UNHEALTHY_THRESHOLD - 2 failures
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: UNHEALTHY_THRESHOLD - 2,
                last_error: None,
            },
        );

        // This brings to UNHEALTHY_THRESHOLD - 1 => still healthy
        handle_check_failure(&map, &bus, &key, true, "still ok".into());
        {
            let entry = map.get(&key).unwrap();
            assert!(entry.is_healthy);
            assert_eq!(entry.consecutive_failures, UNHEALTHY_THRESHOLD - 1);
        }

        // This brings to UNHEALTHY_THRESHOLD => unhealthy
        handle_check_failure(&map, &bus, &key, true, "now bad".into());
        {
            let entry = map.get(&key).unwrap();
            assert!(!entry.is_healthy);
            assert_eq!(entry.consecutive_failures, UNHEALTHY_THRESHOLD);
        }
    }

    #[test]
    fn health_map_concurrent_access_different_keys() {
        let map = new_health_map();
        let bus = std::sync::Arc::new(crate::events::new_event_bus());

        // Simulate multiple provider types independently
        let providers = vec![
            (ProviderType::Ollama, "http://a:11434"),
            (ProviderType::LlamaCpp, "http://b:8080"),
            (ProviderType::OpenAi, "https://c.com"),
        ];

        for (pt, url) in &providers {
            let key = (*pt, url.to_string());
            // Two failures each (still healthy)
            handle_check_failure(&map, &bus, &key, true, "fail1".into());
            handle_check_failure(&map, &bus, &key, true, "fail2".into());
        }

        // All three should still be healthy (2 < 3)
        for (pt, url) in &providers {
            let key = (*pt, url.to_string());
            let entry = map.get(&key).unwrap();
            assert!(entry.is_healthy, "{} should still be healthy", pt);
            assert_eq!(entry.consecutive_failures, 2);
        }
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn handle_check_failure_rapid_recovery_cycle() {
        let map = new_health_map();
        let bus = std::sync::Arc::new(crate::events::new_event_bus());
        let key = (ProviderType::Ollama, "http://localhost:11434".to_string());

        // Cycle 1: fail to unhealthy
        for _ in 0..UNHEALTHY_THRESHOLD {
            let was = map.get(&key).map(|s| s.is_healthy).unwrap_or(true);
            handle_check_failure(&map, &bus, &key, was, "fail".into());
        }
        assert!(!map.get(&key).unwrap().is_healthy);

        // Recovery
        map.insert(
            key.clone(),
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 0,
                last_error: None,
            },
        );

        // Cycle 2: fail to unhealthy again
        for _ in 0..UNHEALTHY_THRESHOLD {
            let was = map.get(&key).map(|s| s.is_healthy).unwrap_or(true);
            handle_check_failure(&map, &bus, &key, was, "fail again".into());
        }
        assert!(!map.get(&key).unwrap().is_healthy);
        assert_eq!(
            map.get(&key).unwrap().consecutive_failures,
            UNHEALTHY_THRESHOLD
        );
    }
}
