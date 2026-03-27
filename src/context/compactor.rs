//! Context compactor — proactive context window management.
//!
//! When conversation messages approach a model's context window limit,
//! the compactor truncates the oldest messages while preserving the system
//! prompt and the most recent turns.

use crate::context::tokens::TokenCounter;
use crate::inference::{Message, Role};
use crate::provider::metadata::ModelMetadataRegistry;

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// The compacted message list.
    pub messages: Vec<Message>,
    /// Token count of the original messages.
    pub original_tokens: u32,
    /// Token count after compaction.
    pub compacted_tokens: u32,
    /// Number of messages dropped.
    pub messages_dropped: usize,
}

/// Context compactor — checks token usage against model context windows and
/// truncates conversations that exceed the configured threshold.
pub struct ContextCompactor {
    /// Ratio (0.0–1.0) of context window at which compaction triggers.
    threshold: f64,
    /// Number of most-recent messages to preserve.
    keep_last: usize,
    /// Whether compaction is enabled.
    enabled: bool,
}

impl ContextCompactor {
    /// Create a new compactor.
    #[must_use]
    pub fn new(threshold: f64, keep_last: usize, enabled: bool) -> Self {
        Self {
            threshold: threshold.clamp(0.1, 1.0),
            keep_last: keep_last.max(1),
            enabled,
        }
    }

    /// Check whether compaction is needed and apply it if so.
    ///
    /// Returns `Some(CompactionResult)` if messages were truncated, `None` if
    /// no compaction was necessary (or compaction is disabled / model unknown).
    #[must_use]
    pub fn compact(
        &self,
        model: &str,
        messages: &[Message],
        registry: &ModelMetadataRegistry,
        counter: &dyn TokenCounter,
    ) -> Option<CompactionResult> {
        if !self.enabled || messages.is_empty() {
            return None;
        }

        let meta = registry.get(model)?;
        let context_window = meta.context_window;
        let token_limit = (context_window as f64 * self.threshold) as u32;
        let original_tokens = counter.count_messages(messages);

        if original_tokens <= token_limit {
            return None;
        }

        tracing::info!(
            model,
            original_tokens,
            token_limit,
            context_window,
            "context compaction triggered"
        );

        let compacted = truncate_messages(messages, self.keep_last, token_limit, counter);
        let compacted_tokens = counter.count_messages(&compacted);
        let messages_dropped = messages.len().saturating_sub(compacted.len());

        Some(CompactionResult {
            messages: compacted,
            original_tokens,
            compacted_tokens,
            messages_dropped,
        })
    }
}

/// Truncate messages, keeping system prompts and the last N messages.
///
/// Strategy:
/// 1. Collect all system messages (always preserved).
/// 2. Keep the last `keep_last` non-system messages.
/// 3. If still over budget, reduce `keep_last` iteratively.
#[must_use]
fn truncate_messages(
    messages: &[Message],
    keep_last: usize,
    token_limit: u32,
    counter: &dyn TokenCounter,
) -> Vec<Message> {
    let system_messages: Vec<&Message> =
        messages.iter().filter(|m| m.role == Role::System).collect();
    let non_system: Vec<&Message> = messages.iter().filter(|m| m.role != Role::System).collect();

    // Binary search for the largest `keep` that fits within the token limit.
    let max_keep = keep_last.min(non_system.len());
    let mut lo = 1usize;
    let mut hi = max_keep;
    let mut best_keep = 1;

    // Pre-compute system message tokens (constant across iterations).
    let system_only: Vec<Message> = system_messages.iter().map(|m| (*m).clone()).collect();

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let start = non_system.len().saturating_sub(mid);
        let mut candidate = system_only.clone();
        candidate.extend(non_system[start..].iter().map(|m| (*m).clone()));

        if counter.count_messages(&candidate) <= token_limit {
            best_keep = mid;
            lo = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
    }

    let start = non_system.len().saturating_sub(best_keep);
    let mut result = system_only;
    result.extend(non_system[start..].iter().map(|m| (*m).clone()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tokens::SimpleTokenCounter;

    fn make_registry() -> ModelMetadataRegistry {
        ModelMetadataRegistry::new()
    }

    fn make_messages(count: usize, content_len: usize) -> Vec<Message> {
        let mut msgs = Vec::with_capacity(count + 1);
        msgs.push(Message::new(Role::System, "You are a helpful assistant."));
        for i in 0..count {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            msgs.push(Message::new(role, "x".repeat(content_len)));
        }
        msgs
    }

    #[test]
    fn no_compaction_when_disabled() {
        let compactor = ContextCompactor::new(0.8, 10, false);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        let messages = make_messages(100, 1000);
        assert!(
            compactor
                .compact("llama3", &messages, &registry, &counter)
                .is_none()
        );
    }

    #[test]
    fn no_compaction_when_under_threshold() {
        let compactor = ContextCompactor::new(0.8, 10, true);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        // llama3 has 8192 context. Small messages won't trigger.
        let messages = make_messages(3, 20);
        assert!(
            compactor
                .compact("llama3", &messages, &registry, &counter)
                .is_none()
        );
    }

    #[test]
    fn compaction_triggers_on_large_context() {
        let compactor = ContextCompactor::new(0.8, 5, true);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        // llama3 = 8192 context. 80% = 6553 tokens.
        // Need messages totaling > 6553 tokens to trigger compaction.
        // 100 messages * 400 chars each = 40,000 chars / 4 = ~10,000 tokens
        let messages = make_messages(100, 400);
        let result = compactor.compact("llama3", &messages, &registry, &counter);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.compacted_tokens < result.original_tokens);
        assert!(result.messages_dropped > 0);
        // System message should be preserved
        assert!(result.messages.iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn unknown_model_returns_none() {
        let compactor = ContextCompactor::new(0.8, 10, true);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        let messages = make_messages(10, 1000);
        assert!(
            compactor
                .compact("unknown-model-xyz", &messages, &registry, &counter)
                .is_none()
        );
    }

    #[test]
    fn system_messages_always_preserved() {
        let compactor = ContextCompactor::new(0.8, 2, true);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        // Force compaction with many large messages on small model
        let messages = make_messages(30, 1000);
        if let Some(result) = compactor.compact("llama3", &messages, &registry, &counter) {
            let system_count = result
                .messages
                .iter()
                .filter(|m| m.role == Role::System)
                .count();
            assert!(system_count >= 1);
        }
    }

    #[test]
    fn empty_messages_returns_none() {
        let compactor = ContextCompactor::new(0.8, 10, true);
        let registry = make_registry();
        let counter = SimpleTokenCounter;
        assert!(
            compactor
                .compact("llama3", &[], &registry, &counter)
                .is_none()
        );
    }

    #[test]
    fn threshold_clamped() {
        let compactor = ContextCompactor::new(5.0, 10, true);
        assert!((compactor.threshold - 1.0).abs() < f64::EPSILON);
        let compactor = ContextCompactor::new(-1.0, 10, true);
        assert!((compactor.threshold - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn keep_last_at_least_one() {
        let compactor = ContextCompactor::new(0.8, 0, true);
        assert_eq!(compactor.keep_last, 1);
    }
}
