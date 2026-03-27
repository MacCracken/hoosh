//! Model metadata registry — decoupled from routing dispatch.
//!
//! Stores per-model capability flags, context window sizes, performance/cost
//! tiers, and modality support for routing decisions and API responses, without
//! coupling to provider selection logic.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Performance/cost tier for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelTier {
    /// Cheap, fast, lower quality.
    Economy,
    /// Balanced cost/performance.
    Standard,
    /// Expensive, high quality.
    Premium,
    /// Chain-of-thought / reasoning models (o1, o3, deepseek-reasoner).
    Reasoning,
}

/// Input/output modality a model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Embedding,
}

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
    /// Maximum output tokens (if separate from context window).
    pub max_output_tokens: Option<u32>,
    /// Performance/cost tier.
    pub tier: ModelTier,
    /// Supported input modalities.
    pub modalities: Vec<Modality>,
    /// Whether the model accepts a system prompt.
    pub supports_system_prompt: bool,
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

    /// All models in a given tier.
    #[must_use]
    pub fn by_tier(&self, tier: ModelTier) -> Vec<&ModelMetadata> {
        self.models.values().filter(|m| m.tier == tier).collect()
    }

    /// All models supporting a given modality.
    #[must_use]
    pub fn by_modality(&self, modality: Modality) -> Vec<&ModelMetadata> {
        self.models
            .values()
            .filter(|m| m.modalities.contains(&modality))
            .collect()
    }

    /// Number of registered models.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
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
        let embeddings_only = ModelCapabilities {
            embeddings: true,
            ..Default::default()
        };

        type ModelDef<'a> = (
            &'a str,
            u32,
            Option<u32>,
            ModelTier,
            Vec<Modality>,
            bool,
            ModelCapabilities,
        );

        let defaults: Vec<ModelDef<'_>> = vec![
            // ── OpenAI ──────────────────────────────────────────────
            (
                "gpt-4o",
                128_000,
                Some(16_384),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gpt-4o-mini",
                128_000,
                Some(16_384),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gpt-4.1",
                1_047_576,
                Some(32_768),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gpt-4.1-mini",
                1_047_576,
                Some(32_768),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gpt-4.1-nano",
                1_047_576,
                Some(32_768),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "o1",
                200_000,
                Some(100_000),
                ModelTier::Reasoning,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "o1-mini",
                128_000,
                Some(65_536),
                ModelTier::Reasoning,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "o1-pro",
                200_000,
                Some(100_000),
                ModelTier::Reasoning,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "o3",
                200_000,
                Some(100_000),
                ModelTier::Reasoning,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "o3-mini",
                200_000,
                Some(100_000),
                ModelTier::Reasoning,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "o4-mini",
                200_000,
                Some(100_000),
                ModelTier::Reasoning,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "text-embedding-3-large",
                8_191,
                None,
                ModelTier::Standard,
                vec![Modality::Embedding],
                false,
                embeddings_only.clone(),
            ),
            (
                "text-embedding-3-small",
                8_191,
                None,
                ModelTier::Economy,
                vec![Modality::Embedding],
                false,
                embeddings_only.clone(),
            ),
            (
                "text-embedding-ada-002",
                8_191,
                None,
                ModelTier::Economy,
                vec![Modality::Embedding],
                false,
                embeddings_only.clone(),
            ),
            // ── Anthropic ───────────────────────────────────────────
            (
                "claude-opus-4",
                200_000,
                Some(32_000),
                ModelTier::Premium,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "claude-sonnet-4",
                200_000,
                Some(16_000),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "claude-3-5-haiku",
                200_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "claude-3-5-sonnet",
                200_000,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "claude-3-haiku",
                200_000,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            // ── DeepSeek ────────────────────────────────────────────
            (
                "deepseek-chat",
                64_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "deepseek-reasoner",
                64_000,
                Some(8_192),
                ModelTier::Reasoning,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "deepseek-coder",
                64_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            // ── Grok (xAI) ──────────────────────────────────────────
            (
                "grok-3",
                131_072,
                Some(16_384),
                ModelTier::Premium,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "grok-3-mini",
                131_072,
                Some(16_384),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "grok-3-fast",
                131_072,
                Some(16_384),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "grok-2",
                131_072,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            // ── Google Gemini ───────────────────────────────────────
            (
                "gemini-2.5-pro",
                1_048_576,
                Some(65_536),
                ModelTier::Premium,
                vec![Modality::Text, Modality::Vision, Modality::Audio],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gemini-2.5-flash",
                1_048_576,
                Some(65_536),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision, Modality::Audio],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gemini-2.0-flash",
                1_048_576,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision, Modality::Audio],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "gemini-2.0-flash-lite",
                1_048_576,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            // ── Groq (hosted) ───────────────────────────────────────
            (
                "llama-3.3-70b",
                128_000,
                Some(32_768),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "llama-3.1-8b",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "llama-3.1-70b",
                128_000,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "llama-3.1-405b",
                128_000,
                Some(16_384),
                ModelTier::Premium,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "llama-4-scout",
                512_000,
                Some(16_384),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "llama-4-maverick",
                1_048_576,
                Some(32_768),
                ModelTier::Premium,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "mixtral-8x7b",
                32_768,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "gemma-2-9b",
                8_192,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "gemma-2-27b",
                8_192,
                Some(4_096),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            // ── Mistral ─────────────────────────────────────────────
            (
                "mistral-large",
                128_000,
                Some(8_192),
                ModelTier::Premium,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "mistral-small",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "codestral",
                32_768,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "pixtral-large",
                128_000,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream_tools_vision.clone(),
            ),
            (
                "mistral-nemo",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            // ── OpenRouter (aggregated) ─────────────────────────────
            (
                "openrouter/auto",
                200_000,
                None,
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            // ── Local models (Ollama/llama.cpp defaults) ────────────
            (
                "llama3",
                8_192,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "llama3:8b",
                8_192,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "llama3:70b",
                8_192,
                Some(4_096),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "llama3.1",
                128_000,
                Some(8_192),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "llama3.2",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream.clone(),
            ),
            (
                "llama3.3",
                128_000,
                Some(32_768),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "mistral",
                32_768,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "mixtral",
                32_768,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "qwen2.5",
                32_768,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "qwen2.5-coder",
                32_768,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "phi-3",
                128_000,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "phi-4",
                16_384,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "gemma2",
                8_192,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "gemma3",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text, Modality::Vision],
                true,
                chat_stream.clone(),
            ),
            (
                "command-r",
                128_000,
                Some(4_096),
                ModelTier::Standard,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "command-r-plus",
                128_000,
                Some(4_096),
                ModelTier::Premium,
                vec![Modality::Text],
                true,
                chat_stream_tools.clone(),
            ),
            (
                "codellama",
                16_384,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "starcoder2",
                16_384,
                Some(4_096),
                ModelTier::Economy,
                vec![Modality::Text],
                false,
                chat_stream.clone(),
            ),
            (
                "deepseek-coder-v2",
                128_000,
                Some(8_192),
                ModelTier::Economy,
                vec![Modality::Text],
                true,
                chat_stream.clone(),
            ),
            (
                "nomic-embed-text",
                8_192,
                None,
                ModelTier::Economy,
                vec![Modality::Embedding],
                false,
                embeddings_only.clone(),
            ),
            (
                "mxbai-embed-large",
                512,
                None,
                ModelTier::Economy,
                vec![Modality::Embedding],
                false,
                embeddings_only,
            ),
        ];

        for (id, ctx, max_out, tier, modalities, sys, caps) in defaults {
            self.register(ModelMetadata {
                id: id.into(),
                context_window: ctx,
                max_output_tokens: max_out,
                tier,
                modalities,
                supports_system_prompt: sys,
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
            max_output_tokens: None,
            tier: ModelTier::Economy,
            modalities: vec![Modality::Text],
            supports_system_prompt: true,
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
        assert!(reg.all().len() >= 60);
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
            max_output_tokens: None,
            tier: ModelTier::Economy,
            modalities: vec![Modality::Text],
            supports_system_prompt: true,
            capabilities: Default::default(),
        });
        // "claude-sonnet-4" is a longer prefix match than "claude"
        let meta = reg.get("claude-sonnet-4-20250514").unwrap();
        assert_eq!(meta.context_window, 200_000); // should match "claude-sonnet-4", not "claude"
    }

    #[test]
    fn model_tiers_populated() {
        let reg = ModelMetadataRegistry::new();
        assert!(!reg.by_tier(ModelTier::Economy).is_empty());
        assert!(!reg.by_tier(ModelTier::Standard).is_empty());
        assert!(!reg.by_tier(ModelTier::Premium).is_empty());
        assert!(!reg.by_tier(ModelTier::Reasoning).is_empty());
    }

    #[test]
    fn vision_models_found() {
        let reg = ModelMetadataRegistry::new();
        let vision = reg.by_modality(Modality::Vision);
        assert!(vision.len() >= 10);
        for m in &vision {
            assert!(m.modalities.contains(&Modality::Vision));
        }
    }

    #[test]
    fn embedding_models_found() {
        let reg = ModelMetadataRegistry::new();
        let emb = reg.by_modality(Modality::Embedding);
        assert!(emb.len() >= 4);
        for m in &emb {
            assert!(m.capabilities.embeddings);
        }
    }

    #[test]
    fn max_output_tokens_set() {
        let reg = ModelMetadataRegistry::new();
        let meta = reg.get("gpt-4o").unwrap();
        assert_eq!(meta.max_output_tokens, Some(16_384));
    }

    #[test]
    fn reasoning_models_have_correct_tier() {
        let reg = ModelMetadataRegistry::new();
        for id in ["o1", "o3", "o3-mini", "o4-mini", "deepseek-reasoner"] {
            let meta = reg.get(id).unwrap();
            assert_eq!(
                meta.tier,
                ModelTier::Reasoning,
                "{id} should be Reasoning tier"
            );
        }
    }

    #[test]
    fn len_matches_all() {
        let reg = ModelMetadataRegistry::new();
        assert_eq!(reg.len(), reg.all().len());
        assert!(!reg.is_empty());
    }

    #[test]
    fn system_prompt_support() {
        let reg = ModelMetadataRegistry::new();
        // starcoder2 is a completion model, no system prompt
        let meta = reg.get("starcoder2").unwrap();
        assert!(!meta.supports_system_prompt);
        // gpt-4o supports system prompt
        let meta = reg.get("gpt-4o").unwrap();
        assert!(meta.supports_system_prompt);
    }
}
