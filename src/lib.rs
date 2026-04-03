//! Hoosh — AI inference gateway for Rust.
//!
//! Multi-provider LLM routing, local model serving, speech-to-text, and
//! token budget management. OpenAI-compatible HTTP API.
//!
//! > **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI.
//!
//! # Architecture
//!
//! ```text
//! Clients (tarang, daimon, agnoshi, consumer apps)
//!     │
//!     ▼
//! Router (provider selection, load balancing, fallback)
//!     │
//!     ├──▶ Local backends (Ollama, llama.cpp, Synapse, whisper.cpp)
//!     │
//!     └──▶ Remote APIs (OpenAI, Anthropic, DeepSeek, Mistral, Groq, ...)
//!           │
//!           ▼
//!     Cache ◀── Rate Limiter ◀── Token Budget
//! ```
//!
//! # Quick start
//!
//! ```rust,no_run
//! use hoosh::{InferenceRequest, HooshClient};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = HooshClient::new("http://localhost:8088");
//! let response = client.infer(&InferenceRequest {
//!     model: "llama3".into(),
//!     prompt: "Explain Rust ownership in one sentence.".into(),
//!     ..Default::default()
//! }).await?;
//! println!("{}", response.text);
//! # Ok(())
//! # }
//! ```

pub mod audit;
pub mod budget;
pub mod cache;
pub mod client;
pub mod config;
pub mod context;
pub mod cost;
#[cfg(feature = "dlp")]
pub mod dlp;
pub mod error;
pub mod events;
#[cfg(feature = "hwaccel")]
pub mod hardware;
pub mod health;
pub mod inference;
pub mod metrics;
pub mod middleware;
pub mod provider;
pub mod queue;
pub mod router;
pub mod server;
#[cfg(feature = "otel")]
pub mod telemetry;
pub mod tools;

pub use budget::{TokenBudget, TokenPool};
pub use cache::ResponseCache;
pub use client::HooshClient;
pub use error::HooshError;
pub use inference::{InferenceRequest, InferenceResponse, ModelInfo};
pub use provider::{LlmProvider, ProviderRegistry, ProviderType};
pub use router::Router;
pub use tools::{ToolCall, ToolChoice, ToolDefinition, ToolResult};

/// Install the rustls ring crypto provider for tests (called once per process).
#[cfg(test)]
#[doc(hidden)]
pub fn install_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[cfg(test)]
mod tests;
