//! Groq provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType};

/// Groq API provider (default: `https://api.groq.com/openai`).
pub struct GroqProvider {
    inner: OpenAiCompatibleProvider,
}

impl GroqProvider {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "https://api.groq.com/openai".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, api_key, ProviderType::Groq),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for GroqProvider {
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse> {
        self.inner.infer(request).await
    }
    async fn infer_stream(&self, request: InferenceRequest) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<String>>> {
        self.inner.infer_stream(request).await
    }
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        self.inner.list_models().await
    }
    async fn health_check(&self) -> anyhow::Result<bool> {
        self.inner.health_check().await
    }
    fn provider_type(&self) -> ProviderType {
        ProviderType::Groq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url() {
        let p = GroqProvider::new("", Some("gsk-test".into()));
        assert_eq!(p.inner.base_url(), "https://api.groq.com/openai");
    }

    #[test]
    fn provider_type_is_groq() {
        let p = GroqProvider::new("", None);
        assert_eq!(p.provider_type(), ProviderType::Groq);
    }
}
