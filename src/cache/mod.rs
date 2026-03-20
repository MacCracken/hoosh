//! Response caching for inference results.

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
    value: String,
    created: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created.elapsed() > self.ttl
    }
}

/// Thread-safe response cache with TTL eviction.
pub struct ResponseCache {
    entries: DashMap<String, CacheEntry>,
    config: CacheConfig,
}

impl ResponseCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: DashMap::new(),
            config,
        }
    }

    /// Look up a cached response by key.
    pub fn get(&self, key: &str) -> Option<String> {
        if !self.config.enabled {
            return None;
        }
        let entry = self.entries.get(key)?;
        if entry.is_expired() {
            drop(entry);
            self.entries.remove(key);
            return None;
        }
        Some(entry.value.clone())
    }

    /// Insert a response into the cache.
    pub fn insert(&self, key: String, value: String) {
        if !self.config.enabled {
            return;
        }
        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired();
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
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

    /// Remove expired entries.
    fn evict_expired(&self) {
        self.entries.retain(|_, entry| !entry.is_expired());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_insert_get() {
        let cache = ResponseCache::new(CacheConfig::default());
        cache.insert("key1".into(), "value1".into());
        assert_eq!(cache.get("key1"), Some("value1".into()));
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
}
