//! Token counting — provider-aware token estimation without external tokenizer deps.
//!
//! Provides a [`TokenCounter`] trait and two implementations:
//! - [`SimpleTokenCounter`]: rough chars/4 heuristic (always available)
//! - [`ProviderTokenCounter`]: per-provider ratios tuned to known tokenizer behavior

use crate::inference::Message;
use crate::provider::ProviderType;

/// Token counter trait — estimate token counts without calling a tokenizer API.
pub trait TokenCounter: Send + Sync {
    /// Estimate token count for a string.
    fn count(&self, text: &str) -> u32;

    /// Estimate token count for a message sequence.
    ///
    /// Accounts for per-message formatting overhead (role tokens, separators).
    fn count_messages(&self, messages: &[Message]) -> u32;
}

/// Simple token counter using chars/4 heuristic.
///
/// Rough approximation: ~4 characters per token for English text.
/// This is the zero-dependency baseline, always available.
pub struct SimpleTokenCounter;

impl SimpleTokenCounter {
    /// Characters-per-token ratio used by the simple estimator.
    const CHARS_PER_TOKEN: f64 = 4.0;

    /// Per-message overhead: role name + delimiters (~4 tokens).
    const MESSAGE_OVERHEAD: u32 = 4;
}

impl TokenCounter for SimpleTokenCounter {
    #[inline]
    fn count(&self, text: &str) -> u32 {
        (text.len() as f64 / Self::CHARS_PER_TOKEN).ceil() as u32
    }

    fn count_messages(&self, messages: &[Message]) -> u32 {
        let mut total = 0u32;
        for msg in messages {
            total = total.saturating_add(self.count(&msg.content.text()));
            total = total.saturating_add(Self::MESSAGE_OVERHEAD);
            // Tool call arguments contribute tokens
            for tc in &msg.tool_calls {
                total = total.saturating_add(self.count(&tc.name));
                total = total.saturating_add(self.count(&tc.arguments.to_string()));
            }
        }
        // Base overhead for the conversation framing
        total.saturating_add(3)
    }
}

/// Provider-aware token counter with per-provider character ratios.
///
/// Different tokenizers produce different token counts for the same text.
/// This counter adjusts the chars-per-token ratio based on the provider family.
pub struct ProviderTokenCounter {
    chars_per_token: f64,
    message_overhead: u32,
}

impl ProviderTokenCounter {
    /// Create a counter tuned for the given provider type.
    #[must_use]
    pub fn for_provider(provider: ProviderType) -> Self {
        let (cpt, overhead) = match provider {
            // OpenAI cl100k/o200k: ~3.7-4.0 chars/token for English
            ProviderType::OpenAi => (3.8, 4),
            // Anthropic: slightly different tokenizer, ~3.5 chars/token
            ProviderType::Anthropic => (3.5, 4),
            // DeepSeek: similar to OpenAI
            ProviderType::DeepSeek => (3.8, 4),
            // Mistral: SentencePiece-based, ~3.8 chars/token
            ProviderType::Mistral => (3.8, 4),
            // Groq hosts various models, use conservative estimate
            ProviderType::Groq => (3.8, 4),
            // Grok (xAI): similar to OpenAI
            ProviderType::Grok => (3.8, 4),
            // OpenRouter routes to various providers
            ProviderType::OpenRouter => (3.8, 4),
            // Local models: typically LLaMA tokenizer ~3.5-4.0
            ProviderType::Ollama
            | ProviderType::LlamaCpp
            | ProviderType::LmStudio
            | ProviderType::LocalAi => (3.7, 4),
            // Synapse and others: conservative default
            _ => (4.0, 4),
        };
        Self {
            chars_per_token: cpt,
            message_overhead: overhead,
        }
    }
}

impl TokenCounter for ProviderTokenCounter {
    #[inline]
    fn count(&self, text: &str) -> u32 {
        (text.len() as f64 / self.chars_per_token).ceil() as u32
    }

    fn count_messages(&self, messages: &[Message]) -> u32 {
        let mut total = 0u32;
        for msg in messages {
            total = total.saturating_add(self.count(&msg.content.text()));
            total = total.saturating_add(self.message_overhead);
            for tc in &msg.tool_calls {
                total = total.saturating_add(self.count(&tc.name));
                total = total.saturating_add(self.count(&tc.arguments.to_string()));
            }
        }
        total.saturating_add(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Role;

    #[test]
    fn simple_counter_empty_string() {
        let counter = SimpleTokenCounter;
        assert_eq!(counter.count(""), 0);
    }

    #[test]
    fn simple_counter_short_text() {
        let counter = SimpleTokenCounter;
        // "hello" = 5 chars → ceil(5/4) = 2
        assert_eq!(counter.count("hello"), 2);
    }

    #[test]
    fn simple_counter_longer_text() {
        let counter = SimpleTokenCounter;
        // 100 chars → 25 tokens
        let text = "a".repeat(100);
        assert_eq!(counter.count(&text), 25);
    }

    #[test]
    fn simple_counter_messages() {
        let counter = SimpleTokenCounter;
        let messages = vec![
            Message::new(Role::System, "You are helpful."),
            Message::new(Role::User, "Hello there"),
        ];
        let count = counter.count_messages(&messages);
        // "You are helpful." = 16 chars → 4 tokens + 4 overhead = 8
        // "Hello there" = 11 chars → 3 tokens + 4 overhead = 7
        // + 3 base = 18
        assert_eq!(count, 18);
    }

    #[test]
    fn simple_counter_empty_messages() {
        let counter = SimpleTokenCounter;
        assert_eq!(counter.count_messages(&[]), 3);
    }

    #[test]
    fn provider_counter_openai() {
        let counter = ProviderTokenCounter::for_provider(ProviderType::OpenAi);
        // 100 chars / 3.8 = 26.3 → ceil = 27
        let text = "a".repeat(100);
        assert_eq!(counter.count(&text), 27);
    }

    #[test]
    fn provider_counter_anthropic() {
        let counter = ProviderTokenCounter::for_provider(ProviderType::Anthropic);
        // 100 chars / 3.5 = 28.57 → ceil = 29
        let text = "a".repeat(100);
        assert_eq!(counter.count(&text), 29);
    }

    #[test]
    fn provider_counter_local() {
        let counter = ProviderTokenCounter::for_provider(ProviderType::Ollama);
        // 100 chars / 3.7 = 27.03 → ceil = 28
        let text = "a".repeat(100);
        assert_eq!(counter.count(&text), 28);
    }

    #[test]
    fn provider_counter_messages() {
        let counter = ProviderTokenCounter::for_provider(ProviderType::OpenAi);
        let messages = vec![Message::new(Role::User, "What is Rust?")];
        let count = counter.count_messages(&messages);
        // "What is Rust?" = 13 chars / 3.8 = 3.42 → ceil = 4 + 4 overhead + 3 base = 11
        assert_eq!(count, 11);
    }

    #[test]
    fn saturation_on_huge_input() {
        let counter = SimpleTokenCounter;
        // Verify no panic on very large token counts
        let text = "a".repeat(u32::MAX as usize / 2);
        let _ = counter.count(&text);
    }

    #[test]
    fn provider_counter_with_tool_calls() {
        let counter = SimpleTokenCounter;
        let messages = vec![Message {
            role: Role::Assistant,
            content: "Let me check that.".into(),
            tool_call_id: None,
            tool_calls: vec![crate::tools::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "London"}),
            }],
        }];
        let count = counter.count_messages(&messages);
        // content: 18/4=5, overhead: 4, fn name: 11/4=3, args serialized: ~17 chars/4=5, base: 3
        assert!(count > 15, "expected at least 15 tokens, got {count}");
    }
}
