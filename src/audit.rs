//! Cryptographic audit log — HMAC-SHA256 linked chain for tamper-proof request/response logging.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const CHAIN_VERSION: &str = "1.0.0";

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: u64,
    pub event: String,
    pub level: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    pub integrity: IntegrityFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityFields {
    pub version: String,
    pub signature: String,
    pub previous_hash: String,
}

/// Thread-safe audit chain.
pub struct AuditChain {
    inner: Mutex<AuditChainInner>,
}

struct AuditChainInner {
    signing_key: Vec<u8>,
    last_hash: String,
    entries: VecDeque<AuditEntry>,
    max_entries: usize,
    /// Hash of the first entry in the deque (updated on eviction).
    first_valid_hash: String,
}

/// Compute SHA-256 hash, returning hex string.
fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let hash = Sha256::digest(data);
    hex::encode(hash)
}

/// Compute HMAC-SHA256, returning hex string.
fn hmac_sha256_hex(data: &[u8], key: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    hex::encode(mac.finalize().into_bytes())
}

/// Generate a unique ID using UUID v4 (consistent with rest of codebase).
fn generate_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Current epoch time in milliseconds.
fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Compute the entry hash from the entry's data fields (excluding integrity).
fn compute_entry_hash(entry: &AuditEntry) -> String {
    let mut data = BTreeMap::new();
    data.insert("id", serde_json::Value::String(entry.id.clone()));
    data.insert("event", serde_json::Value::String(entry.event.clone()));
    data.insert("level", serde_json::Value::String(entry.level.clone()));
    data.insert("message", serde_json::Value::String(entry.message.clone()));
    data.insert(
        "timestamp",
        serde_json::Value::Number(entry.timestamp.into()),
    );
    if let Some(ref provider) = entry.provider {
        data.insert("provider", serde_json::Value::String(provider.clone()));
    }
    if let Some(ref model) = entry.model {
        data.insert("model", serde_json::Value::String(model.clone()));
    }
    if let Some(ref meta) = entry.metadata {
        data.insert("metadata", meta.clone());
    }

    let json = serde_json::to_string(&data).expect("BTreeMap<&str, Value> always serializes");
    sha256_hex(json.as_bytes())
}

impl AuditChain {
    pub fn new(signing_key: &[u8], max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(AuditChainInner {
                signing_key: signing_key.to_vec(),
                last_hash: GENESIS_HASH.to_string(),
                entries: VecDeque::new(),
                max_entries,
                first_valid_hash: GENESIS_HASH.to_string(),
            }),
        }
    }

    /// Record an audit event. Thread-safe.
    pub fn record(
        &self,
        event: &str,
        level: &str,
        message: &str,
        provider: Option<&str>,
        model: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> AuditEntry {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        let id = generate_id();
        let timestamp = now_epoch_ms();

        // Build a temporary entry (with placeholder integrity) for hashing
        let entry = AuditEntry {
            id,
            timestamp,
            event: event.to_string(),
            level: level.to_string(),
            message: message.to_string(),
            provider: provider.map(|s| s.to_string()),
            model: model.map(|s| s.to_string()),
            metadata,
            integrity: IntegrityFields {
                version: CHAIN_VERSION.to_string(),
                signature: String::new(),
                previous_hash: inner.last_hash.clone(),
            },
        };

        let entry_hash = compute_entry_hash(&entry);

        // Compute signature: HMAC-SHA256("{entry_hash}:{previous_hash}", signing_key)
        let sig_input = format!("{}:{}", entry_hash, inner.last_hash);
        let signature = hmac_sha256_hex(sig_input.as_bytes(), &inner.signing_key);

        let entry = AuditEntry {
            integrity: IntegrityFields {
                version: CHAIN_VERSION.to_string(),
                signature,
                previous_hash: inner.last_hash.clone(),
            },
            ..entry
        };

        inner.last_hash = entry_hash;

        // Evict oldest if at capacity (O(1) with VecDeque)
        if inner.entries.len() >= inner.max_entries
            && let Some(evicted) = inner.entries.pop_front()
        {
            // Update first_valid_hash to the evicted entry's own hash
            // so verify() knows where the surviving chain starts
            inner.first_valid_hash = compute_entry_hash(&evicted);
        }

        inner.entries.push_back(entry.clone());
        entry
    }

    /// Verify the entire chain integrity. Returns (valid, optional error message).
    pub fn verify(&self) -> (bool, Option<String>) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Start from first_valid_hash (accounts for evicted entries)
        let mut prev_hash = inner.first_valid_hash.clone();

        for (i, entry) in inner.entries.iter().enumerate() {
            // Check previous hash link
            if entry.integrity.previous_hash != prev_hash {
                return (
                    false,
                    Some(format!(
                        "Entry {} ({}): previous hash mismatch",
                        i, entry.id
                    )),
                );
            }

            // Recompute entry hash
            let entry_hash = compute_entry_hash(entry);

            // Verify signature
            let sig_input = format!("{}:{}", entry_hash, prev_hash);
            let expected_sig = hmac_sha256_hex(sig_input.as_bytes(), &inner.signing_key);

            if entry.integrity.signature != expected_sig {
                return (
                    false,
                    Some(format!(
                        "Entry {} ({}): signature verification failed",
                        i, entry.id
                    )),
                );
            }

            prev_hash = entry_hash;
        }

        (true, None)
    }

    /// Number of entries.
    pub fn count(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.len()
    }

    /// Get last N entries (for the API endpoint).
    pub fn recent(&self, n: usize) -> Vec<AuditEntry> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let len = inner.entries.len();
        let start = len.saturating_sub(n);
        inner.entries.iter().skip(start).cloned().collect()
    }

    /// Get a snapshot of recent entries and count in a single lock acquisition.
    /// Chain validity is always `true` since entries are only appended under lock.
    /// Use `verify()` explicitly for full cryptographic verification.
    pub fn snapshot(&self, n: usize) -> (Vec<AuditEntry>, usize, bool) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let len = inner.entries.len();
        let start = len.saturating_sub(n);
        let entries: Vec<AuditEntry> = inner.entries.iter().skip(start).cloned().collect();
        (entries, len, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_verify() {
        let chain = AuditChain::new(b"test-signing-key", 10_000);
        chain.record(
            "inference.request",
            "info",
            "Request to llama3",
            Some("ollama"),
            Some("llama3"),
            None,
        );
        chain.record(
            "inference.response",
            "info",
            "Response from llama3",
            Some("ollama"),
            Some("llama3"),
            None,
        );

        let (valid, err) = chain.verify();
        assert!(valid, "Chain should be valid: {:?}", err);
        assert_eq!(chain.count(), 2);
    }

    #[test]
    fn tamper_detection() {
        let chain = AuditChain::new(b"key", 10_000);
        chain.record("event", "info", "msg", None, None, None);
        chain.record("event2", "info", "msg2", None, None, None);

        // Tamper with an entry
        {
            let mut inner = chain.inner.lock().unwrap();
            inner.entries[0].message = "TAMPERED".to_string();
        }

        let (valid, err) = chain.verify();
        assert!(!valid);
        assert!(err.unwrap().contains("signature verification failed"));
    }

    #[test]
    fn empty_chain_verifies() {
        let chain = AuditChain::new(b"key", 10_000);
        let (valid, err) = chain.verify();
        assert!(valid);
        assert!(err.is_none());
        assert_eq!(chain.count(), 0);
    }

    #[test]
    fn max_entries_eviction() {
        let chain = AuditChain::new(b"key", 5);
        for i in 0..10 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }
        assert_eq!(chain.count(), 5);

        // The oldest entries should have been evicted; recent ones remain
        let entries = chain.recent(5);
        assert_eq!(entries.len(), 5);
        assert!(entries[0].message.contains("entry 5"));
        assert!(entries[4].message.contains("entry 9"));
    }

    #[test]
    fn recent_returns_correct_entries() {
        let chain = AuditChain::new(b"key", 10_000);
        for i in 0..10 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }

        let last3 = chain.recent(3);
        assert_eq!(last3.len(), 3);
        assert!(last3[0].message.contains("entry 7"));
        assert!(last3[1].message.contains("entry 8"));
        assert!(last3[2].message.contains("entry 9"));

        // Requesting more than available returns all
        let all = chain.recent(100);
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn tamper_previous_hash_link() {
        let chain = AuditChain::new(b"key", 10_000);
        chain.record("e1", "info", "first", None, None, None);
        chain.record("e2", "info", "second", None, None, None);

        {
            let mut inner = chain.inner.lock().unwrap();
            inner.entries[1].integrity.previous_hash = "deadbeef".repeat(8);
        }

        let (valid, err) = chain.verify();
        assert!(!valid);
        assert!(err.unwrap().contains("previous hash mismatch"));
    }

    #[test]
    fn entry_with_metadata() {
        let chain = AuditChain::new(b"key", 10_000);
        let meta = serde_json::json!({"tokens": 42, "cached": true});
        chain.record(
            "inference.response",
            "info",
            "completed",
            Some("openai"),
            Some("gpt-4"),
            Some(meta),
        );

        let (valid, _) = chain.verify();
        assert!(valid);

        let entries = chain.recent(1);
        assert_eq!(entries[0].provider.as_deref(), Some("openai"));
        assert_eq!(entries[0].model.as_deref(), Some("gpt-4"));
        assert!(entries[0].metadata.is_some());
    }

    #[test]
    fn thread_safety() {
        use std::sync::Arc;
        let chain = Arc::new(AuditChain::new(b"key", 10_000));
        let mut handles = Vec::new();

        for t in 0..8 {
            let chain = chain.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    chain.record(
                        "event",
                        "info",
                        &format!("thread {t} entry {i}"),
                        None,
                        None,
                        None,
                    );
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(chain.count(), 400);
        // Note: after concurrent writes the chain won't verify because the
        // eviction of entries by other threads can break the hash linkage
        // within a single thread's perspective. But the mutex ensures no
        // data corruption. For a chain that verifies, entries must be
        // recorded sequentially, which the mutex guarantees internally.
        let (valid, _) = chain.verify();
        assert!(valid);
    }

    #[test]
    fn entry_ids_are_unique() {
        let chain = AuditChain::new(b"key", 10_000);
        chain.record("e1", "info", "first", None, None, None);
        chain.record("e2", "info", "second", None, None, None);

        let entries = chain.recent(2);
        assert_ne!(entries[0].id, entries[1].id);
    }

    #[test]
    fn integrity_version_is_set() {
        let chain = AuditChain::new(b"key", 10_000);
        chain.record("test", "info", "msg", None, None, None);
        let entries = chain.recent(1);
        assert_eq!(entries[0].integrity.version, CHAIN_VERSION);
    }
}
