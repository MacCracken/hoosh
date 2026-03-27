//! Semantic cache — embedding-similarity-based cache lookups.
//!
//! Augments the exact-match `ResponseCache` with cosine similarity matching.
//! When a query doesn't match exactly, compute its embedding and find the
//! closest cached entry above a configurable threshold.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// Configuration for the semantic cache layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Whether semantic caching is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Cosine similarity threshold (0.0–1.0). Matches above this are cache hits.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Model to use for computing embeddings (e.g. "text-embedding-3-small").
    #[serde(default)]
    pub embedding_model: String,
    /// Maximum entries to search (limits linear scan cost).
    #[serde(default = "default_max_search")]
    pub max_search: usize,
}

fn default_threshold() -> f32 {
    0.92
}

fn default_max_search() -> usize {
    100
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: default_threshold(),
            embedding_model: String::new(),
            max_search: default_max_search(),
        }
    }
}

/// A cached embedding entry.
struct EmbeddingEntry {
    /// Cache key (same key used in the underlying `ResponseCache`).
    cache_key: String,
    /// Embedding vector for the query.
    embedding: Vec<f32>,
}

/// Semantic cache backed by embedding similarity.
pub struct SemanticCache {
    entries: DashMap<String, EmbeddingEntry>,
    config: SemanticCacheConfig,
}

impl SemanticCache {
    /// Create a new semantic cache.
    #[must_use]
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            entries: DashMap::new(),
            config,
        }
    }

    /// Find the best matching cache key for a given embedding.
    ///
    /// Returns the cache key and similarity score if a match exceeds the threshold.
    #[must_use]
    pub fn find_similar(&self, query_embedding: &[f32]) -> Option<(String, f32)> {
        if !self.config.enabled || self.entries.is_empty() {
            return None;
        }

        let mut best_key: Option<String> = None;
        let mut best_score: f32 = self.config.threshold;

        // Full scan — cosine similarity on f32 vecs is fast for reasonable cache sizes.
        // max_search caps the scan if the cache is very large.
        for (searched, entry) in self.entries.iter().enumerate() {
            if self.config.max_search > 0 && searched >= self.config.max_search {
                break;
            }
            let score = cosine_similarity(query_embedding, &entry.embedding);
            if score > best_score {
                best_score = score;
                best_key = Some(entry.cache_key.clone());
            }
        }

        best_key.map(|k| (k, best_score))
    }

    /// Store an embedding for a cache key.
    pub fn insert(&self, cache_key: String, embedding: Vec<f32>) {
        if !self.config.enabled {
            return;
        }
        self.entries.insert(
            cache_key.clone(),
            EmbeddingEntry {
                cache_key,
                embedding,
            },
        );
    }

    /// Remove an entry.
    pub fn remove(&self, cache_key: &str) {
        self.entries.remove(cache_key);
    }

    /// Number of stored embeddings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the semantic cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether semantic caching is enabled.
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configured embedding model.
    #[must_use]
    pub fn embedding_model(&self) -> &str {
        &self.config.embedding_model
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0]. Higher values indicate more similarity.
/// Returns 0.0 if either vector is zero-length or vectors differ in dimension.
#[must_use]
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn semantic_cache_disabled() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        assert!(!cache.is_enabled());
        assert!(cache.find_similar(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn semantic_cache_insert_and_find() {
        let cache = SemanticCache::new(SemanticCacheConfig {
            enabled: true,
            threshold: 0.9,
            max_search: 100,
            ..Default::default()
        });

        cache.insert("key1".into(), vec![1.0, 0.0, 0.0]);
        cache.insert("key2".into(), vec![0.0, 1.0, 0.0]);

        // Query very similar to key1
        let result = cache.find_similar(&[0.99, 0.01, 0.0]);
        assert!(result.is_some());
        let (key, score) = result.unwrap();
        assert_eq!(key, "key1");
        assert!(score > 0.9);
    }

    #[test]
    fn semantic_cache_below_threshold() {
        let cache = SemanticCache::new(SemanticCacheConfig {
            enabled: true,
            threshold: 0.99, // very strict
            max_search: 100,
            ..Default::default()
        });

        cache.insert("key1".into(), vec![1.0, 0.0, 0.0]);

        // Query not similar enough
        let result = cache.find_similar(&[0.7, 0.7, 0.0]);
        assert!(result.is_none());
    }

    #[test]
    fn semantic_cache_remove() {
        let cache = SemanticCache::new(SemanticCacheConfig {
            enabled: true,
            ..Default::default()
        });
        cache.insert("key1".into(), vec![1.0]);
        assert_eq!(cache.len(), 1);
        cache.remove("key1");
        assert!(cache.is_empty());
    }

    #[test]
    fn semantic_cache_max_search_limit() {
        let cache = SemanticCache::new(SemanticCacheConfig {
            enabled: true,
            threshold: 0.5,
            max_search: 2, // only search 2 entries
            ..Default::default()
        });

        // Insert 5 entries
        for i in 0..5 {
            let mut v = vec![0.0; 3];
            v[i % 3] = 1.0;
            cache.insert(format!("key{i}"), v);
        }

        // Should still work but only searches first 2
        let _ = cache.find_similar(&[1.0, 0.0, 0.0]);
        assert_eq!(cache.len(), 5);
    }
}
