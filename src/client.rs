//! HTTP client for talking to a hoosh server.
//!
//! This is what downstream crates (tarang, daimon, consumer apps) use to
//! call hoosh over the network. OpenAI-compatible API.

use crate::error::{HooshError, Result};
use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo};

/// HTTP client for the hoosh inference gateway.
#[derive(Debug, Clone)]
pub struct HooshClient {
    base_url: String,
    client: reqwest::Client,
}

impl HooshClient {
    /// Create a new client pointing at the given hoosh server.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Run inference via the hoosh server.
    pub async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(request)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        let response: InferenceResponse = resp
            .json()
            .await
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        Ok(response)
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        let models: Vec<ModelInfo> = resp
            .json()
            .await
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        Ok(models)
    }

    /// Health check.
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/v1/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Base URL of the hoosh server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_creation() {
        let client = HooshClient::new("http://localhost:8088");
        assert_eq!(client.base_url(), "http://localhost:8088");
    }

    #[test]
    fn client_strips_trailing_slash() {
        let client = HooshClient::new("http://localhost:8088/");
        assert_eq!(client.base_url(), "http://localhost:8088");
    }
}
