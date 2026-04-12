//! Per-provider sliding-window rate limiting.

use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Registry of per-provider rate limiters.
pub struct RateLimitRegistry {
    limits: DashMap<String, RateLimiter>,
}

struct RateLimiter {
    max_rpm: u32,
    window: Mutex<VecDeque<Instant>>,
}

impl RateLimitRegistry {
    pub fn new() -> Self {
        Self {
            limits: DashMap::new(),
        }
    }

    /// Configure (or reconfigure) the RPM limit for a given provider key.
    pub fn configure(&self, provider: &str, max_rpm: u32) {
        self.limits.insert(
            provider.to_string(),
            RateLimiter {
                max_rpm,
                window: Mutex::new(VecDeque::new()),
            },
        );
    }

    /// Check whether a request to `provider` is allowed under its rate limit.
    ///
    /// Returns `true` if the request is allowed (and records it), `false` if
    /// rate-limited. Providers with no configured limit are always allowed.
    pub fn check(&self, provider: &str) -> bool {
        let entry = match self.limits.get(provider) {
            Some(e) => e,
            None => return true, // unconfigured → no limit
        };

        let limiter = entry.value();
        let now = Instant::now();
        let window_start = now - Duration::from_secs(60);

        let mut window = limiter.window.lock().unwrap_or_else(|e| e.into_inner());

        // Evict timestamps older than 60 seconds
        while window.front().is_some_and(|&t| t < window_start) {
            window.pop_front();
        }

        if (window.len() as u32) < limiter.max_rpm {
            window.push_back(now);
            true
        } else {
            false
        }
    }
}

impl Default for RateLimitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconfigured_provider_always_allowed() {
        let reg = RateLimitRegistry::new();
        for _ in 0..1000 {
            assert!(reg.check("unknown-provider"));
        }
    }

    #[test]
    fn requests_within_limit_allowed() {
        let reg = RateLimitRegistry::new();
        reg.configure("test", 5);
        for _ in 0..5 {
            assert!(reg.check("test"));
        }
    }

    #[test]
    fn requests_exceeding_limit_rejected() {
        let reg = RateLimitRegistry::new();
        reg.configure("test", 3);
        assert!(reg.check("test"));
        assert!(reg.check("test"));
        assert!(reg.check("test"));
        assert!(!reg.check("test")); // 4th should fail
        assert!(!reg.check("test")); // 5th should fail
    }

    #[test]
    fn different_providers_independent() {
        let reg = RateLimitRegistry::new();
        reg.configure("a", 1);
        reg.configure("b", 1);
        assert!(reg.check("a"));
        assert!(!reg.check("a"));
        // Provider b is independent
        assert!(reg.check("b"));
        assert!(!reg.check("b"));
    }
}
