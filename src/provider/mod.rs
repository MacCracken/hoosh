//! LLM provider trait and type registry.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::router::ProviderRoute;

// Provider backend modules (feature-gated)
pub mod openai_compat;

// Local providers
#[cfg(feature = "llamacpp")]
pub mod llamacpp;
#[cfg(feature = "lmstudio")]
pub mod lmstudio;
#[cfg(feature = "localai")]
pub mod localai;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "synapse")]
pub mod synapse;

// Speech-to-text
#[cfg(feature = "whisper")]
pub mod whisper;

// Text-to-speech
#[cfg(feature = "piper")]
pub mod tts;

// Remote providers
#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "deepseek")]
pub mod deepseek;
#[cfg(feature = "grok")]
pub mod grok;
#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "mistral")]
pub mod mistral;
#[cfg(feature = "openai")]
pub mod openai_remote;
#[cfg(feature = "openrouter")]
pub mod openrouter;

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

    /// Generate embeddings for input text. Not all providers support this.
    async fn embeddings(
        &self,
        _request: &crate::inference::EmbeddingsRequest,
    ) -> anyhow::Result<crate::inference::EmbeddingsResponse> {
        Err(anyhow::anyhow!("embeddings not supported by this provider"))
    }

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

/// Registry of live provider instances, keyed by (type, base_url).
pub struct ProviderRegistry {
    providers: HashMap<(ProviderType, String), Arc<dyn LlmProvider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Construct and register a provider from a route's type + base_url.
    pub fn register_from_route(&mut self, route: &ProviderRoute) {
        let key = (route.provider, route.base_url.clone());
        if self.providers.contains_key(&key) {
            return;
        }

        #[allow(unused_variables)]
        let api_key = route.api_key.clone();
        let provider: Option<Arc<dyn LlmProvider>> = match route.provider {
            // Local providers
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => Some(Arc::new(ollama::OllamaProvider::new(&route.base_url))),
            #[cfg(feature = "llamacpp")]
            ProviderType::LlamaCpp => {
                Some(Arc::new(llamacpp::LlamaCppProvider::new(&route.base_url)))
            }
            #[cfg(feature = "synapse")]
            ProviderType::Synapse => Some(Arc::new(synapse::SynapseProvider::new(&route.base_url))),
            #[cfg(feature = "lmstudio")]
            ProviderType::LmStudio => {
                Some(Arc::new(lmstudio::LmStudioProvider::new(&route.base_url)))
            }
            #[cfg(feature = "localai")]
            ProviderType::LocalAi => Some(Arc::new(localai::LocalAiProvider::new(&route.base_url))),
            // Remote providers
            #[cfg(feature = "openai")]
            ProviderType::OpenAi => Some(Arc::new(openai_remote::OpenAiProvider::new(
                &route.base_url,
                api_key,
            ))),
            #[cfg(feature = "anthropic")]
            ProviderType::Anthropic => Some(Arc::new(anthropic::AnthropicProvider::new(
                &route.base_url,
                api_key,
            ))),
            #[cfg(feature = "deepseek")]
            ProviderType::DeepSeek => Some(Arc::new(deepseek::DeepSeekProvider::new(
                &route.base_url,
                api_key,
            ))),
            #[cfg(feature = "mistral")]
            ProviderType::Mistral => Some(Arc::new(mistral::MistralProvider::new(
                &route.base_url,
                api_key,
            ))),
            #[cfg(feature = "groq")]
            ProviderType::Groq => Some(Arc::new(groq::GroqProvider::new(&route.base_url, api_key))),
            #[cfg(feature = "openrouter")]
            ProviderType::OpenRouter => Some(Arc::new(openrouter::OpenRouterProvider::new(
                &route.base_url,
                api_key,
            ))),
            #[cfg(feature = "grok")]
            ProviderType::Grok => Some(Arc::new(grok::GrokProvider::new(&route.base_url, api_key))),
            _ => None,
        };

        if let Some(p) = provider {
            self.providers.insert(key, p);
        }
    }

    /// Look up a provider by type and base_url.
    pub fn get(&self, provider_type: ProviderType, base_url: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers
            .get(&(provider_type, base_url.to_string()))
            .cloned()
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// All registered providers.
    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn LlmProvider>> {
        self.providers.values()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
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
