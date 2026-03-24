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

impl HooshError {
    /// Map to an OpenAI-compatible HTTP status code.
    pub fn http_status_code(&self) -> u16 {
        match self {
            Self::ModelNotFound(_) | Self::NoProvider(_) => 404,
            Self::RateLimited { .. } | Self::BudgetExceeded { .. } => 429,
            Self::Timeout(_) => 408,
            Self::Cache(_) | Self::Provider(_) => 500,
            Self::Http(e) => e.status().map(|s| s.as_u16()).unwrap_or(502),
            Self::Other(_) => 500,
        }
    }

    /// Map to an OpenAI-compatible error code string.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::ModelNotFound(_) => "model_not_found",
            Self::NoProvider(_) => "no_provider",
            Self::RateLimited { .. } => "rate_limit_exceeded",
            Self::BudgetExceeded { .. } => "budget_exceeded",
            Self::Timeout(_) => "timeout",
            Self::Cache(_) => "cache_error",
            Self::Provider(_) => "provider_error",
            Self::Http(_) => "upstream_error",
            Self::Other(_) => "internal_error",
        }
    }
}

pub type Result<T> = std::result::Result<T, HooshError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_codes() {
        assert_eq!(HooshError::ModelNotFound("x".into()).http_status_code(), 404);
        assert_eq!(HooshError::NoProvider("x".into()).http_status_code(), 404);
        assert_eq!(
            HooshError::RateLimited { retry_after_ms: 1000 }.http_status_code(),
            429
        );
        assert_eq!(
            HooshError::BudgetExceeded {
                pool: "default".into(),
                remaining: 0,
            }
            .http_status_code(),
            429
        );
        assert_eq!(HooshError::Timeout(5000).http_status_code(), 408);
        assert_eq!(HooshError::Provider("err".into()).http_status_code(), 500);
        assert_eq!(HooshError::Cache("err".into()).http_status_code(), 500);
        assert_eq!(
            HooshError::Other(anyhow::anyhow!("err")).http_status_code(),
            500
        );
    }

    #[test]
    fn error_codes() {
        assert_eq!(
            HooshError::ModelNotFound("x".into()).error_code(),
            "model_not_found"
        );
        assert_eq!(HooshError::NoProvider("x".into()).error_code(), "no_provider");
        assert_eq!(
            HooshError::RateLimited { retry_after_ms: 0 }.error_code(),
            "rate_limit_exceeded"
        );
        assert_eq!(
            HooshError::BudgetExceeded {
                pool: "p".into(),
                remaining: 0,
            }
            .error_code(),
            "budget_exceeded"
        );
        assert_eq!(HooshError::Timeout(0).error_code(), "timeout");
        assert_eq!(HooshError::Cache("c".into()).error_code(), "cache_error");
        assert_eq!(
            HooshError::Provider("p".into()).error_code(),
            "provider_error"
        );
        assert_eq!(
            HooshError::Other(anyhow::anyhow!("o")).error_code(),
            "internal_error"
        );
    }

    #[test]
    fn error_display() {
        let e = HooshError::ModelNotFound("llama99".into());
        assert_eq!(e.to_string(), "model not found: llama99");

        let e = HooshError::RateLimited {
            retry_after_ms: 5000,
        };
        assert!(e.to_string().contains("5000"));

        let e = HooshError::BudgetExceeded {
            pool: "default".into(),
            remaining: 42,
        };
        assert!(e.to_string().contains("default"));
        assert!(e.to_string().contains("42"));
    }
}
