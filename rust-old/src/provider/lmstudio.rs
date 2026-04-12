//! LM Studio provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType, TlsConfig};

/// LM Studio provider (default: `http://localhost:1234`).
pub struct LmStudioProvider {
    inner: OpenAiCompatibleProvider,
}

impl LmStudioProvider {
    pub fn new(base_url: impl Into<String>, tls_config: Option<&TlsConfig>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "http://localhost:1234".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, None, ProviderType::LmStudio, tls_config),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for LmStudioProvider {
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
        ProviderType::LmStudio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url() {
        let p = LmStudioProvider::new("", None);
        assert_eq!(p.inner.base_url(), "http://localhost:1234");
    }

    #[test]
    fn custom_url() {
        let p = LmStudioProvider::new("http://workstation:5555", None);
        assert_eq!(p.inner.base_url(), "http://workstation:5555");
    }

    #[test]
    fn provider_type_is_lmstudio() {
        let p = LmStudioProvider::new("", None);
        assert_eq!(p.provider_type(), ProviderType::LmStudio);
    }
}
