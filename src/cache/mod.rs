//! Response caching for inference results.

pub mod semantic;
pub mod warming;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// Configuration for the response cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of cached entries.
    pub max_entries: usize,
    /// Time-to-live for cached entries.
    pub ttl_secs: u64,
    /// Whether caching is enabled.
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_secs: 300,
            enabled: true,
        }
    }
}

struct CacheEntry {
    value: Arc<String>,
    created: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created.elapsed() > self.ttl
    }
}

/// Build a cache key from model + messages hash to avoid collisions.
pub fn cache_key(model: &str, messages: &[crate::inference::Message]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut hasher);
    for m in messages {
        std::mem::discriminant(&m.role).hash(&mut hasher);
        m.content.text().hash(&mut hasher);
    }
    format!("{}:{:016x}", model, hasher.finish())
}

/// Cache statistics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Current number of entries.
    pub entries: usize,
    /// Maximum entries allowed.
    pub max_entries: usize,
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total evictions (expired + capacity).
    pub evictions: u64,
    /// Hit rate (0.0–1.0). NaN if no lookups yet.
    pub hit_rate: f64,
    /// Whether caching is enabled.
    pub enabled: bool,
}

/// Thread-safe response cache with TTL eviction and statistics tracking.
pub struct ResponseCache {
    entries: DashMap<String, CacheEntry>,
    config: CacheConfig,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl ResponseCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: DashMap::new(),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Look up a cached response by key.
    pub fn get(&self, key: &str) -> Option<Arc<String>> {
        if !self.config.enabled {
            return None;
        }
        let entry = match self.entries.get(key) {
            Some(e) => e,
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        if entry.is_expired() {
            drop(entry);
            self.entries.remove(key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let value = entry.value.clone();
        drop(entry);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(value)
    }

    /// Insert a response into the cache.
    pub fn insert(&self, key: String, value: String) {
        if !self.config.enabled {
            return;
        }
        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired();
            // If still at capacity after evicting expired, drop oldest entries
            if self.entries.len() >= self.config.max_entries {
                let to_remove = self.entries.len() - self.config.max_entries + 1;
                let keys: Vec<String> = self
                    .entries
                    .iter()
                    .take(to_remove)
                    .map(|e| e.key().clone())
                    .collect();
                let removed = keys.len() as u64;
                for key in keys {
                    self.entries.remove(&key);
                }
                self.evictions.fetch_add(removed, Ordering::Relaxed);
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                value: Arc::new(value),
                created: Instant::now(),
                ttl: Duration::from_secs(self.config.ttl_secs),
            },
        );
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.entries.clear();
    }

    /// Get a snapshot of cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            entries: self.entries.len(),
            max_entries: self.config.max_entries,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            enabled: self.config.enabled,
        }
    }

    /// Remove expired entries.
    fn evict_expired(&self) {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired());
        let evicted = before.saturating_sub(self.entries.len()) as u64;
        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_insert_get() {
        let cache = ResponseCache::new(CacheConfig::default());
        cache.insert("key1".into(), "value1".into());
        assert_eq!(cache.get("key1").as_deref(), Some(&"value1".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_miss() {
        let cache = ResponseCache::new(CacheConfig::default());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn cache_disabled() {
        let cache = ResponseCache::new(CacheConfig {
            enabled: false,
            ..Default::default()
        });
        cache.insert("key".into(), "value".into());
        assert!(cache.get("key").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_clear() {
        let cache = ResponseCache::new(CacheConfig::default());
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn config_default() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.max_entries, 1000);
        assert_eq!(cfg.ttl_secs, 300);
        assert!(cfg.enabled);
    }

    #[test]
    fn cache_eviction_at_capacity() {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 3,
            ttl_secs: 300,
            enabled: true,
        });
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.insert("c".into(), "3".into());
        assert_eq!(cache.len(), 3);
        // This should trigger eviction
        cache.insert("d".into(), "4".into());
        assert!(cache.len() <= 3);
        // New entry should be present
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn cache_ttl_expiry() {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 100,
            ttl_secs: 0, // expire immediately
            enabled: true,
        });
        cache.insert("key".into(), "value".into());
        // Entry should be expired
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(cache.get("key").is_none());
    }

    #[test]
    fn cache_key_different_models() {
        use crate::inference::{Message, Role};
        let msgs = vec![Message::new(Role::User, "hello")];
        let k1 = super::cache_key("llama3", &msgs);
        let k2 = super::cache_key("gpt-4", &msgs);
        assert_ne!(k1, k2);
        assert!(k1.starts_with("llama3:"));
        assert!(k2.starts_with("gpt-4:"));
    }

    #[test]
    fn cache_key_different_messages() {
        use crate::inference::{Message, Role};
        let msgs1 = vec![Message::new(Role::User, "hello")];
        let msgs2 = vec![Message::new(Role::User, "world")];
        let k1 = super::cache_key("model", &msgs1);
        let k2 = super::cache_key("model", &msgs2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_same_request() {
        use crate::inference::{Message, Role};
        let msgs = vec![Message::new(Role::User, "hello")];
        let k1 = super::cache_key("model", &msgs);
        let k2 = super::cache_key("model", &msgs);
        assert_eq!(k1, k2);
    }

    #[test]
    fn evict_expired_removes_stale() {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 100,
            ttl_secs: 0,
            enabled: true,
        });
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.evict_expired();
        assert!(cache.is_empty());
    }

    #[test]
    fn stats_initial() {
        let cache = ResponseCache::new(CacheConfig::default());
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
        assert!(stats.enabled);
    }

    #[test]
    fn stats_hit_and_miss() {
        let cache = ResponseCache::new(CacheConfig::default());
        cache.insert("key".into(), "value".into());
        let _ = cache.get("key"); // hit
        let _ = cache.get("missing"); // miss
        let _ = cache.get("key"); // hit
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn stats_eviction_counted() {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 2,
            ttl_secs: 300,
            enabled: true,
        });
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.insert("c".into(), "3".into()); // triggers eviction
        let stats = cache.stats();
        assert!(stats.evictions >= 1);
    }

    #[test]
    fn stats_ttl_eviction_counted() {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 100,
            ttl_secs: 0,
            enabled: true,
        });
        cache.insert("key".into(), "value".into());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = cache.get("key"); // should be expired → eviction + miss
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.misses, 1);
    }
}
