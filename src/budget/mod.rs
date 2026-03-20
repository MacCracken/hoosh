//! Token budget management: per-agent and per-pool token accounting.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A named token pool with a capacity limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPool {
    /// Pool name (e.g. "default", "agent-123", "batch-jobs").
    pub name: String,
    /// Maximum tokens allowed in this pool.
    pub capacity: u64,
    /// Tokens consumed so far.
    pub used: u64,
    /// Tokens currently reserved (pending completion).
    pub reserved: u64,
}

impl TokenPool {
    /// Create a new pool with the given capacity.
    pub fn new(name: impl Into<String>, capacity: u64) -> Self {
        Self {
            name: name.into(),
            capacity,
            used: 0,
            reserved: 0,
        }
    }

    /// Available tokens (capacity - used - reserved).
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.used + self.reserved)
    }

    /// Whether the pool can accommodate `tokens` more.
    pub fn can_reserve(&self, tokens: u64) -> bool {
        self.available() >= tokens
    }

    /// Reserve tokens (before inference).
    pub fn reserve(&mut self, tokens: u64) -> bool {
        if !self.can_reserve(tokens) {
            return false;
        }
        self.reserved += tokens;
        true
    }

    /// Commit reserved tokens as used (after inference completes).
    pub fn commit(&mut self, reserved: u64, actual: u64) {
        self.reserved = self.reserved.saturating_sub(reserved);
        self.used = self.used.saturating_add(actual);
    }

    /// Release reserved tokens without using them (on failure/cancel).
    pub fn release(&mut self, tokens: u64) {
        self.reserved = self.reserved.saturating_sub(tokens);
    }

    /// Utilisation ratio (0.0–1.0).
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.used as f64 / self.capacity as f64
    }
}

/// Token budget manager: tracks multiple named pools.
pub struct TokenBudget {
    pools: HashMap<String, TokenPool>,
}

impl TokenBudget {
    /// Create a new budget manager.
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Add or replace a pool.
    pub fn add_pool(&mut self, pool: TokenPool) {
        self.pools.insert(pool.name.clone(), pool);
    }

    /// Get a pool by name.
    pub fn get_pool(&self, name: &str) -> Option<&TokenPool> {
        self.pools.get(name)
    }

    /// Get a mutable pool by name.
    pub fn get_pool_mut(&mut self, name: &str) -> Option<&mut TokenPool> {
        self.pools.get_mut(name)
    }

    /// All pools.
    pub fn pools(&self) -> &HashMap<String, TokenPool> {
        &self.pools
    }

    /// Check if a pool can accommodate a request.
    pub fn check(&self, pool_name: &str, tokens: u64) -> bool {
        self.pools
            .get(pool_name)
            .map(|p| p.can_reserve(tokens))
            .unwrap_or(false)
    }

    /// Reserve tokens in a pool.
    pub fn reserve(&mut self, pool_name: &str, tokens: u64) -> bool {
        self.pools
            .get_mut(pool_name)
            .map(|p| p.reserve(tokens))
            .unwrap_or(false)
    }

    /// Report actual usage and release the reservation.
    pub fn report(&mut self, pool_name: &str, reserved: u64, actual: u64) {
        if let Some(pool) = self.pools.get_mut(pool_name) {
            pool.commit(reserved, actual);
        }
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_basic() {
        let mut pool = TokenPool::new("test", 1000);
        assert_eq!(pool.available(), 1000);
        assert!(pool.reserve(400));
        assert_eq!(pool.available(), 600);
        pool.commit(400, 350);
        assert_eq!(pool.used, 350);
        assert_eq!(pool.reserved, 0);
        assert_eq!(pool.available(), 650);
    }

    #[test]
    fn pool_over_budget() {
        let pool = TokenPool::new("test", 100);
        assert!(!pool.can_reserve(200));
    }

    #[test]
    fn pool_release() {
        let mut pool = TokenPool::new("test", 1000);
        pool.reserve(500);
        pool.release(500);
        assert_eq!(pool.reserved, 0);
        assert_eq!(pool.available(), 1000);
    }

    #[test]
    fn pool_utilization() {
        let mut pool = TokenPool::new("test", 1000);
        pool.used = 750;
        assert!((pool.utilization() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_multi_pool() {
        let mut budget = TokenBudget::new();
        budget.add_pool(TokenPool::new("default", 10000));
        budget.add_pool(TokenPool::new("agent-1", 5000));

        assert!(budget.check("default", 8000));
        assert!(!budget.check("agent-1", 8000));
        assert!(!budget.check("nonexistent", 1));
    }

    #[test]
    fn budget_reserve_report() {
        let mut budget = TokenBudget::new();
        budget.add_pool(TokenPool::new("pool", 1000));
        assert!(budget.reserve("pool", 500));
        budget.report("pool", 500, 420);
        let pool = budget.get_pool("pool").unwrap();
        assert_eq!(pool.used, 420);
        assert_eq!(pool.reserved, 0);
    }
}
