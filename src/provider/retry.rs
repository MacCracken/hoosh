//! Retry manager — jittered exponential backoff with error classification.
//!
//! Wraps provider inference calls with automatic retry for transient failures.
//! Distinguishes retryable errors (429, 5xx, timeouts) from permanent errors
//! (400, 401, 404, budget exceeded).

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Retry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts. 0 = no retries.
    pub max_retries: u32,
    /// Base delay between retries in milliseconds.
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Jitter factor (0.0–1.0). Randomizes delay to prevent thundering herd.
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
            max_delay_ms: 30_000,
            jitter_factor: 0.5,
        }
    }
}

/// Retry manager wrapping async operations with backoff.
pub struct RetryManager {
    config: RetryConfig,
}

impl RetryManager {
    /// Create a new retry manager with the given configuration.
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute an async operation with retry logic.
    ///
    /// The closure `f` is called for each attempt. If it returns an error
    /// that is classified as retryable, the manager waits with exponential
    /// backoff before retrying.
    pub async fn with_retry<F, Fut, T>(&self, mut f: F) -> anyhow::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let mut attempt = 0u32;
        loop {
            match f().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    let is_retryable = e
                        .downcast_ref::<crate::error::HooshError>()
                        .map(|he| he.is_retryable())
                        .unwrap_or(false); // unknown errors are NOT retried

                    if !is_retryable || attempt >= self.config.max_retries {
                        if attempt > 0 {
                            tracing::warn!(
                                attempt,
                                max_retries = self.config.max_retries,
                                "retry exhausted or permanent error"
                            );
                        }
                        return Err(e);
                    }

                    let delay = self.compute_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = self.config.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "retrying after transient error"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Compute delay for a given attempt using jittered exponential backoff.
    ///
    /// delay = min(base * 2^attempt, max) * (1 + random(0, jitter))
    #[must_use]
    fn compute_delay(&self, attempt: u32) -> Duration {
        let base = self.config.base_delay_ms as f64;
        let exp_delay = base * 2.0f64.powi(attempt as i32);
        let capped = exp_delay.min(self.config.max_delay_ms as f64);

        let jitter = if self.config.jitter_factor > 0.0 {
            let j: f64 = rand::random::<f64>() * self.config.jitter_factor;
            1.0 + j
        } else {
            1.0
        };

        Duration::from_millis((capped * jitter) as u64)
    }

    /// Whether retry is enabled (max_retries > 0).
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.max_retries > 0
    }
}

/// Execute an operation with retry, using a pinned future factory.
///
/// Convenience function for use in handlers where the closure returns
/// a pinned boxed future.
pub async fn retry_with<F, T>(manager: &RetryManager, f: F) -> anyhow::Result<T>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send>>,
{
    manager.with_retry(f).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn default_config() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_delay_ms, 500);
        assert_eq!(cfg.max_delay_ms, 30_000);
        assert!((cfg.jitter_factor - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn delay_exponential() {
        let manager = RetryManager::new(RetryConfig {
            jitter_factor: 0.0, // no jitter for deterministic testing
            ..Default::default()
        });
        let d0 = manager.compute_delay(0);
        let d1 = manager.compute_delay(1);
        let d2 = manager.compute_delay(2);
        assert_eq!(d0.as_millis(), 500);
        assert_eq!(d1.as_millis(), 1000);
        assert_eq!(d2.as_millis(), 2000);
    }

    #[test]
    fn delay_capped_at_max() {
        let manager = RetryManager::new(RetryConfig {
            base_delay_ms: 10_000,
            max_delay_ms: 15_000,
            jitter_factor: 0.0,
            ..Default::default()
        });
        // 10000 * 2^2 = 40000, capped to 15000
        let d = manager.compute_delay(2);
        assert_eq!(d.as_millis(), 15_000);
    }

    #[test]
    fn is_enabled() {
        let enabled = RetryManager::new(RetryConfig::default());
        assert!(enabled.is_enabled());

        let disabled = RetryManager::new(RetryConfig {
            max_retries: 0,
            ..Default::default()
        });
        assert!(!disabled.is_enabled());
    }

    #[tokio::test]
    async fn retry_succeeds_first_try() {
        let manager = RetryManager::new(RetryConfig::default());
        let result = manager
            .with_retry(|| async { Ok::<_, anyhow::Error>(42) })
            .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retry_succeeds_after_failures() {
        let manager = RetryManager::new(RetryConfig {
            max_retries: 3,
            base_delay_ms: 1, // minimal delay for test speed
            max_delay_ms: 10,
            jitter_factor: 0.0,
        });
        let attempts = AtomicU32::new(0);
        let result = manager
            .with_retry(|| {
                let count = attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err(anyhow::anyhow!(crate::error::HooshError::Timeout(1000)))
                    } else {
                        Ok(99)
                    }
                }
            })
            .await;
        assert_eq!(result.unwrap(), 99);
        assert_eq!(attempts.load(Ordering::SeqCst), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn retry_stops_on_permanent_error() {
        let manager = RetryManager::new(RetryConfig {
            max_retries: 3,
            base_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        });
        let attempts = AtomicU32::new(0);
        let result: anyhow::Result<i32> = manager
            .with_retry(|| {
                attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(anyhow::anyhow!(crate::error::HooshError::ModelNotFound(
                        "no".into()
                    )))
                }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1); // no retries
    }

    #[tokio::test]
    async fn retry_exhausts_max_retries() {
        let manager = RetryManager::new(RetryConfig {
            max_retries: 2,
            base_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        });
        let attempts = AtomicU32::new(0);
        let result: anyhow::Result<i32> = manager
            .with_retry(|| {
                attempts.fetch_add(1, Ordering::SeqCst);
                async { Err(anyhow::anyhow!(crate::error::HooshError::Timeout(1000))) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[test]
    fn compute_delay_with_jitter() {
        let manager = RetryManager::new(RetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            jitter_factor: 0.5,
            ..Default::default()
        });
        // With jitter factor 0.5, delay should be in range [100, 150] for attempt 0
        let delay = manager.compute_delay(0);
        assert!(delay.as_millis() >= 100);
        assert!(delay.as_millis() <= 150);
    }

    #[test]
    fn compute_delay_zero_jitter() {
        let manager = RetryManager::new(RetryConfig {
            base_delay_ms: 200,
            max_delay_ms: 10_000,
            jitter_factor: 0.0,
            ..Default::default()
        });
        let d0 = manager.compute_delay(0);
        let d1 = manager.compute_delay(1);
        assert_eq!(d0.as_millis(), 200);
        assert_eq!(d1.as_millis(), 400);
    }

    #[test]
    fn compute_delay_large_attempt_capped() {
        let manager = RetryManager::new(RetryConfig {
            base_delay_ms: 500,
            max_delay_ms: 5_000,
            jitter_factor: 0.0,
            ..Default::default()
        });
        // 500 * 2^10 = 512000, should be capped to 5000
        let d = manager.compute_delay(10);
        assert_eq!(d.as_millis(), 5000);
    }

    #[tokio::test]
    async fn retry_with_convenience_fn() {
        let manager = RetryManager::new(RetryConfig {
            max_retries: 1,
            base_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        });
        let result = retry_with(&manager, || Box::pin(async { Ok::<_, anyhow::Error>(42) })).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retry_unknown_error_not_retried() {
        // Unknown errors (not HooshError) should NOT be retried
        let manager = RetryManager::new(RetryConfig {
            max_retries: 3,
            base_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        });
        let attempts = AtomicU32::new(0);
        let result: anyhow::Result<i32> = manager
            .with_retry(|| {
                attempts.fetch_add(1, Ordering::SeqCst);
                async { Err(anyhow::anyhow!("generic error, not HooshError")) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1); // no retries
    }

    #[test]
    fn retry_config_serde_roundtrip() {
        let cfg = RetryConfig {
            max_retries: 5,
            base_delay_ms: 1000,
            max_delay_ms: 60_000,
            jitter_factor: 0.3,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RetryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_retries, 5);
        assert_eq!(back.base_delay_ms, 1000);
        assert_eq!(back.max_delay_ms, 60_000);
        assert!((back.jitter_factor - 0.3).abs() < f64::EPSILON);
    }
}
