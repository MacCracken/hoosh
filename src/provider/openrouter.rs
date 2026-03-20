//! OpenRouter provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType};

/// OpenRouter API provider (default: `https://openrouter.ai/api`).
pub struct OpenRouterProvider {
    inner: OpenAiCompatibleProvider,
}

impl OpenRouterProvider {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "https://openrouter.ai/api".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, api_key, ProviderType::OpenRouter),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse> {
        self.inner.infer(request).await
    }
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<String>>> {
        self.inner.infer_stream(request).await
    }
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        self.inner.list_models().await
    }
    async fn health_check(&self) -> anyhow::Result<bool> {
        self.inner.health_check().await
    }
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenRouter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url() {
        let p = OpenRouterProvider::new("", Some("sk-or-test".into()));
        assert_eq!(p.inner.base_url(), "https://openrouter.ai/api");
    }

    #[test]
    fn provider_type_is_openrouter() {
        let p = OpenRouterProvider::new("", None);
        assert_eq!(p.provider_type(), ProviderType::OpenRouter);
    }
}
