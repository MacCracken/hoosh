//! LLM provider trait and type registry.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};

/// Trait for LLM inference providers.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Run inference and return the complete response.
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse>;

    /// Stream inference results token by token.
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<String>>>;

    /// List models available from this provider.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>>;

    /// Check if the provider is healthy / reachable.
    async fn health_check(&self) -> anyhow::Result<bool>;

    /// Provider type identifier.
    fn provider_type(&self) -> ProviderType;
}

/// Enumeration of all supported provider backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProviderType {
    // Local backends
    Ollama,
    LlamaCpp,
    Synapse,
    LmStudio,
    LocalAi,
    // Remote APIs
    OpenAi,
    Anthropic,
    DeepSeek,
    Mistral,
    Google,
    Groq,
    Grok,
    OpenRouter,
    // Speech-to-text
    Whisper,
}

impl ProviderType {
    /// Whether this provider runs locally (no data leaves the machine).
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            Self::Ollama
                | Self::LlamaCpp
                | Self::Synapse
                | Self::LmStudio
                | Self::LocalAi
                | Self::Whisper
        )
    }

    /// Whether this provider supports streaming responses.
    pub fn supports_streaming(&self) -> bool {
        !matches!(self, Self::Whisper)
    }
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ollama => write!(f, "ollama"),
            Self::LlamaCpp => write!(f, "llamacpp"),
            Self::Synapse => write!(f, "synapse"),
            Self::LmStudio => write!(f, "lmstudio"),
            Self::LocalAi => write!(f, "localai"),
            Self::OpenAi => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::DeepSeek => write!(f, "deepseek"),
            Self::Mistral => write!(f, "mistral"),
            Self::Google => write!(f, "google"),
            Self::Groq => write!(f, "groq"),
            Self::Grok => write!(f, "grok"),
            Self::OpenRouter => write!(f, "openrouter"),
            Self::Whisper => write!(f, "whisper"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_type_display() {
        assert_eq!(ProviderType::Ollama.to_string(), "ollama");
        assert_eq!(ProviderType::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderType::Whisper.to_string(), "whisper");
    }

    #[test]
    fn local_providers() {
        assert!(ProviderType::Ollama.is_local());
        assert!(ProviderType::LlamaCpp.is_local());
        assert!(ProviderType::Whisper.is_local());
        assert!(!ProviderType::OpenAi.is_local());
        assert!(!ProviderType::Anthropic.is_local());
    }

    #[test]
    fn streaming_support() {
        assert!(ProviderType::Ollama.supports_streaming());
        assert!(ProviderType::OpenAi.supports_streaming());
        assert!(!ProviderType::Whisper.supports_streaming());
    }

    #[test]
    fn serde_roundtrip() {
        let types = [
            ProviderType::Ollama,
            ProviderType::Anthropic,
            ProviderType::Whisper,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: ProviderType = serde_json::from_str(&json).unwrap();
            assert_eq!(*t, back);
        }
    }
}
