//! OpenAI remote provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType, TlsConfig};

/// OpenAI API provider (default: `https://api.openai.com`).
pub struct OpenAiProvider {
    inner: OpenAiCompatibleProvider,
}

impl OpenAiProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        tls_config: Option<&TlsConfig>,
    ) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "https://api.openai.com".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, api_key, ProviderType::OpenAi, tls_config),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
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
        ProviderType::OpenAi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url() {
        let p = OpenAiProvider::new("", Some("sk-test".into()), None);
        assert_eq!(p.inner.base_url(), "https://api.openai.com");
    }

    #[test]
    fn custom_url() {
        let p = OpenAiProvider::new("https://custom.openai.azure.com", Some("key".into()), None);
        assert_eq!(p.inner.base_url(), "https://custom.openai.azure.com");
    }

    #[test]
    fn provider_type_is_openai() {
        let p = OpenAiProvider::new("", None, None);
        assert_eq!(p.provider_type(), ProviderType::OpenAi);
    }
}
