//! Model metadata registry — decoupled from routing dispatch.
//!
//! Stores per-model capability flags and context window sizes for routing
//! decisions and API responses, without coupling to provider selection logic.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Capability flags for a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub chat: bool,
    pub streaming: bool,
    pub tool_use: bool,
    pub vision: bool,
    pub embeddings: bool,
}

/// Metadata for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier (e.g. "gpt-4o", "claude-sonnet-4").
    pub id: String,
    /// Context window size in tokens.
    pub context_window: u32,
    /// Capability flags.
    pub capabilities: ModelCapabilities,
}

/// Registry of model metadata, keyed by model ID prefix.
pub struct ModelMetadataRegistry {
    models: HashMap<String, ModelMetadata>,
}

impl ModelMetadataRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            models: HashMap::new(),
        };
        reg.load_defaults();
        reg
    }

    /// Look up metadata by model ID. Tries exact match, then prefix match.
    pub fn get(&self, model_id: &str) -> Option<&ModelMetadata> {
        if let Some(m) = self.models.get(model_id) {
            return Some(m);
        }
        // Prefix match: "claude-sonnet-4-20250514" matches "claude-sonnet-4"
        self.models
            .iter()
            .filter(|(k, _)| model_id.starts_with(k.as_str()))
            .max_by_key(|(k, _)| k.len())
            .map(|(_, v)| v)
    }

    /// Register or update model metadata.
    pub fn register(&mut self, meta: ModelMetadata) {
        self.models.insert(meta.id.clone(), meta);
    }

    /// All registered models.
    pub fn all(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    fn load_defaults(&mut self) {
        let chat_stream = ModelCapabilities {
            chat: true,
            streaming: true,
            ..Default::default()
        };
        let chat_stream_tools = ModelCapabilities {
            chat: true,
            streaming: true,
            tool_use: true,
            ..Default::default()
        };
        let chat_stream_tools_vision = ModelCapabilities {
            chat: true,
            streaming: true,
            tool_use: true,
            vision: true,
            ..Default::default()
        };

        let defaults = [
            // OpenAI
            ("gpt-4o", 128_000, chat_stream_tools_vision.clone()),
            ("gpt-4o-mini", 128_000, chat_stream_tools_vision.clone()),
            ("o1", 200_000, chat_stream_tools.clone()),
            ("o3-mini", 200_000, chat_stream_tools.clone()),
            // Anthropic
            ("claude-opus-4", 200_000, chat_stream_tools_vision.clone()),
            ("claude-sonnet-4", 200_000, chat_stream_tools_vision.clone()),
            ("claude-3-5-haiku", 200_000, chat_stream_tools.clone()),
            // DeepSeek
            ("deepseek-chat", 64_000, chat_stream_tools.clone()),
            ("deepseek-reasoner", 64_000, chat_stream_tools.clone()),
            // Grok
            ("grok-3", 131_072, chat_stream_tools.clone()),
            ("grok-3-mini", 131_072, chat_stream_tools.clone()),
            // Gemini
            (
                "gemini-2.0-flash",
                1_048_576,
                chat_stream_tools_vision.clone(),
            ),
            // Groq (hosted)
            ("llama-3.3-70b", 128_000, chat_stream.clone()),
            ("llama-3.1-8b", 128_000, chat_stream.clone()),
            // Mistral
            ("mistral-large", 128_000, chat_stream_tools.clone()),
            // Local models (typical defaults)
            ("llama3", 8_192, chat_stream.clone()),
            ("llama3:8b", 8_192, chat_stream.clone()),
            ("llama3:70b", 8_192, chat_stream.clone()),
            ("mistral", 32_768, chat_stream.clone()),
            ("qwen2.5", 32_768, chat_stream),
        ];

        for (id, ctx, caps) in defaults {
            self.register(ModelMetadata {
                id: id.into(),
                context_window: ctx,
                capabilities: caps,
            });
        }
    }
}

impl Default for ModelMetadataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lookup() {
        let reg = ModelMetadataRegistry::new();
        let meta = reg.get("gpt-4o").unwrap();
        assert_eq!(meta.context_window, 128_000);
        assert!(meta.capabilities.tool_use);
        assert!(meta.capabilities.vision);
    }

    #[test]
    fn prefix_lookup() {
        let reg = ModelMetadataRegistry::new();
        let meta = reg.get("claude-sonnet-4-20250514").unwrap();
        assert_eq!(meta.context_window, 200_000);
        assert!(meta.capabilities.tool_use);
    }

    #[test]
    fn unknown_model_returns_none() {
        let reg = ModelMetadataRegistry::new();
        assert!(reg.get("totally-unknown-model").is_none());
    }

    #[test]
    fn register_custom_model() {
        let mut reg = ModelMetadataRegistry::new();
        reg.register(ModelMetadata {
            id: "my-finetuned-model".into(),
            context_window: 4096,
            capabilities: ModelCapabilities {
                chat: true,
                ..Default::default()
            },
        });
        let meta = reg.get("my-finetuned-model").unwrap();
        assert_eq!(meta.context_window, 4096);
        assert!(!meta.capabilities.streaming);
    }

    #[test]
    fn all_returns_defaults() {
        let reg = ModelMetadataRegistry::new();
        assert!(reg.all().len() >= 15);
    }

    #[test]
    fn local_models_no_tool_use() {
        let reg = ModelMetadataRegistry::new();
        let meta = reg.get("llama3").unwrap();
        assert!(meta.capabilities.chat);
        assert!(!meta.capabilities.tool_use);
    }

    #[test]
    fn longest_prefix_wins() {
        let mut reg = ModelMetadataRegistry::new();
        reg.register(ModelMetadata {
            id: "claude".into(),
            context_window: 100,
            capabilities: Default::default(),
        });
        // "claude-sonnet-4" is a longer prefix match than "claude"
        let meta = reg.get("claude-sonnet-4-20250514").unwrap();
        assert_eq!(meta.context_window, 200_000); // should match "claude-sonnet-4", not "claude"
    }
}
