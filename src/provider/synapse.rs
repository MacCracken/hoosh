//! Synapse provider — thin wrapper over OpenAI-compatible API.

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};
use crate::provider::openai_compat::OpenAiCompatibleProvider;
use crate::provider::{LlmProvider, ProviderType};

/// Synapse provider (default: `http://localhost:5000`).
pub struct SynapseProvider {
    inner: OpenAiCompatibleProvider,
}

impl SynapseProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "http://localhost:5000".to_string()
        } else {
            url
        };
        Self {
            inner: OpenAiCompatibleProvider::new(url, None, ProviderType::Synapse),
        }
    }

    /// Check training job status.
    pub async fn training_status(&self, job_id: &str) -> anyhow::Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let url = format!("{}/v1/training/{}", self.inner.base_url(), job_id);
        let resp = client.get(&url).send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Sync the model catalog.
    pub async fn sync_catalog(&self) -> anyhow::Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let url = format!("{}/v1/catalog/sync", self.inner.base_url());
        let resp = client.post(&url).send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }
}

#[async_trait::async_trait]
impl LlmProvider for SynapseProvider {
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
        ProviderType::Synapse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url() {
        let p = SynapseProvider::new("");
        assert_eq!(p.inner.base_url(), "http://localhost:5000");
    }

    #[test]
    fn custom_url() {
        let p = SynapseProvider::new("http://synapse:7000");
        assert_eq!(p.inner.base_url(), "http://synapse:7000");
    }

    #[test]
    fn provider_type_is_synapse() {
        let p = SynapseProvider::new("");
        assert_eq!(p.provider_type(), ProviderType::Synapse);
    }
}
