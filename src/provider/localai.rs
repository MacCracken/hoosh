//! LocalAI provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType};

/// LocalAI provider (default: `http://localhost:8080`).
pub struct LocalAiProvider {
    inner: OpenAiCompatibleProvider,
}

impl LocalAiProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "http://localhost:8080".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, None, ProviderType::LocalAi),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for LocalAiProvider {
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
        ProviderType::LocalAi
    }
}
