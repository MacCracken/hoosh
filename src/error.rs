//! Error types for hoosh.

use thiserror::Error;

/// Top-level error type.
#[derive(Debug, Error)]
pub enum HooshError {
    #[error("provider error: {0}")]
    Provider(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("rate limited: retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("token budget exceeded: pool '{pool}' has {remaining} tokens remaining")]
    BudgetExceeded { pool: String, remaining: u64 },

    #[error("no provider available for model '{0}'")]
    NoProvider(String),

    #[error("inference timeout after {0}ms")]
    Timeout(u64),

    #[error("cache error: {0}")]
    Cache(String),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, HooshError>;
