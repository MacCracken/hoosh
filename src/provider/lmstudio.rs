//! LM Studio provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType};

/// LM Studio provider (default: `http://localhost:1234`).
pub struct LmStudioProvider {
    inner: OpenAiCompatibleProvider,
}

impl LmStudioProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "http://localhost:1234".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, None, ProviderType::LmStudio),
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
