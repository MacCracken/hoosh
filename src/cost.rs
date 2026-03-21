//! Per-provider cost tracking — pricing table and cost accumulation.

use std::collections::HashMap;
use std::sync::LazyLock;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::inference::TokenUsage;
use crate::provider::ProviderType;

/// Pricing for a single model (USD per million tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    #[serde(default)]
    pub cached_input_per_million: Option<f64>,
}

impl ModelPricing {
    const fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            cached_input_per_million: None,
        }
    }

    const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Static pricing table — sourced from secureyeoman's cost-calculator.
static PRICING: LazyLock<HashMap<&'static str, ModelPricing>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    // Anthropic
    m.insert("claude-opus-4", ModelPricing::new(15.0, 75.0));
    m.insert("claude-sonnet-4", ModelPricing::new(3.0, 15.0));
    m.insert("claude-3.5-haiku", ModelPricing::new(0.8, 4.0));
    m.insert("claude-3-5-haiku", ModelPricing::new(0.8, 4.0));

    // OpenAI
    m.insert("gpt-4o", ModelPricing::new(2.5, 10.0));
    m.insert("gpt-4o-mini", ModelPricing::new(0.15, 0.6));
    m.insert("o1", ModelPricing::new(15.0, 60.0));
    m.insert("o3-mini", ModelPricing::new(1.1, 4.4));

    // DeepSeek
    m.insert("deepseek-chat", ModelPricing::new(0.27, 1.1));
    m.insert("deepseek-coder", ModelPricing::new(0.14, 0.28));
    m.insert("deepseek-reasoner", ModelPricing::new(0.55, 2.19));

    // Grok
    m.insert("grok-3", ModelPricing::new(3.0, 15.0));
    m.insert("grok-3-mini", ModelPricing::new(0.3, 0.5));

    // Groq (hosted models)
    m.insert("llama-3.3-70b", ModelPricing::new(0.59, 0.79));
    m.insert("llama-3.1-8b", ModelPricing::new(0.05, 0.08));

    // Google Gemini
    m.insert("gemini-2.0-flash", ModelPricing::new(0.1, 0.4));

    m
});

/// Fallback pricing when a model is not found in the pricing table.
fn fallback_pricing(provider: ProviderType) -> ModelPricing {
    if provider.is_local() {
        return ModelPricing::zero();
    }
    // For remote providers with unknown models, use a conservative middle-ground.
    match provider {
        ProviderType::Anthropic => ModelPricing::new(3.0, 15.0),
        ProviderType::OpenAi => ModelPricing::new(2.5, 10.0),
        ProviderType::DeepSeek => ModelPricing::new(0.27, 1.1),
        ProviderType::Groq => ModelPricing::new(0.59, 0.79),
        ProviderType::Grok => ModelPricing::new(3.0, 15.0),
        ProviderType::Google => ModelPricing::new(0.1, 0.4),
        ProviderType::Mistral => ModelPricing::new(2.0, 6.0),
        ProviderType::OpenRouter => ModelPricing::new(2.5, 10.0),
        _ => ModelPricing::zero(),
    }
}

/// Look up pricing for a model.  Falls back to provider-level pricing.
///
/// 1. Try exact match in PRICING table.
/// 2. Try prefix match (e.g. "claude-sonnet-4-20250514" matches "claude-sonnet-4").
/// 3. Fall back to provider-level default.
fn lookup_pricing(model: &str, provider: ProviderType) -> ModelPricing {
    // Exact match
    if let Some(p) = PRICING.get(model) {
        return p.clone();
    }
    // Prefix match — find the longest key that is a prefix of the model name.
    let mut best: Option<(&str, &ModelPricing)> = None;
    for (key, pricing) in PRICING.iter() {
        if model.starts_with(key) && best.is_none_or(|(k, _)| key.len() > k.len()) {
            best = Some((key, pricing));
        }
    }
    if let Some((_, p)) = best {
        return p.clone();
    }
    fallback_pricing(provider)
}

/// Per-provider cumulative cost record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCostRecord {
    pub provider: String,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost_usd: f64,
    pub request_count: u64,
}

/// Thread-safe cost accumulator keyed by `"provider:base_url"`.
pub struct CostTracker {
    records: DashMap<String, ProviderCostRecord>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            records: DashMap::new(),
        }
    }

    /// Calculate cost for a single request and add to the running total.
    /// Returns the cost (USD) of this individual request.
    pub fn record(
        &self,
        provider: ProviderType,
        base_url: &str,
        model: &str,
        usage: &TokenUsage,
    ) -> f64 {
        let pricing = lookup_pricing(model, provider);
        let cost = (usage.prompt_tokens as f64 * pricing.input_per_million / 1_000_000.0)
            + (usage.completion_tokens as f64 * pricing.output_per_million / 1_000_000.0);

        let key = format!("{provider}:{base_url}");
        let mut entry = self
            .records
            .entry(key)
            .or_insert_with(|| ProviderCostRecord {
                provider: provider.to_string(),
                ..Default::default()
            });
        entry.total_input_tokens += usage.prompt_tokens as u64;
        entry.total_output_tokens += usage.completion_tokens as u64;
        entry.total_cost_usd += cost;
        entry.request_count += 1;

        cost
    }

    /// Get all cost records.
    pub fn all(&self) -> Vec<ProviderCostRecord> {
        self.records.iter().map(|r| r.value().clone()).collect()
    }

    /// Get all records and total cost in a single pass.
    pub fn all_with_total(&self) -> (Vec<ProviderCostRecord>, f64) {
        let mut records = Vec::new();
        let mut total = 0.0;
        for entry in self.records.iter() {
            total += entry.value().total_cost_usd;
            records.push(entry.value().clone());
        }
        (records, total)
    }

    /// Get total cost across all providers.
    pub fn total_cost(&self) -> f64 {
        self.records.iter().map(|r| r.value().total_cost_usd).sum()
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.records.clear();
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_lookup_exact_match() {
        let p = lookup_pricing("gpt-4o", ProviderType::OpenAi);
        assert!((p.input_per_million - 2.5).abs() < f64::EPSILON);
        assert!((p.output_per_million - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pricing_lookup_prefix_match() {
        // Versioned model name should match the base entry
        let p = lookup_pricing("claude-sonnet-4-20250514", ProviderType::Anthropic);
        assert!((p.input_per_million - 3.0).abs() < f64::EPSILON);
        assert!((p.output_per_million - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pricing_fallback_to_provider() {
        // An unknown model for a known remote provider should use provider defaults
        let p = lookup_pricing("some-unknown-model", ProviderType::Anthropic);
        assert!((p.input_per_million - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pricing_local_providers_are_free() {
        for provider in [
            ProviderType::Ollama,
            ProviderType::LlamaCpp,
            ProviderType::LmStudio,
            ProviderType::LocalAi,
            ProviderType::Synapse,
        ] {
            let p = lookup_pricing("anything", provider);
            assert!(
                p.input_per_million == 0.0 && p.output_per_million == 0.0,
                "local provider {provider} should be free"
            );
        }
    }

    #[test]
    fn cost_calculation_math() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            total_tokens: 1_500,
        };
        // gpt-4o: $2.5 / 1M input, $10 / 1M output
        let cost = tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );
        let expected = (1_000.0 * 2.5 / 1_000_000.0) + (500.0 * 10.0 / 1_000_000.0);
        assert!(
            (cost - expected).abs() < 1e-12,
            "cost={cost}, expected={expected}"
        );
    }

    #[test]
    fn tracker_accumulates() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );
        tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );

        let records = tracker.all();
        assert_eq!(records.len(), 1);
        let rec = &records[0];
        assert_eq!(rec.request_count, 2);
        assert_eq!(rec.total_input_tokens, 200);
        assert_eq!(rec.total_output_tokens, 100);
        assert!(rec.total_cost_usd > 0.0);
    }

    #[test]
    fn tracker_separates_providers() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );
        tracker.record(
            ProviderType::Anthropic,
            "https://api.anthropic.com",
            "claude-sonnet-4",
            &usage,
        );

        assert_eq!(tracker.all().len(), 2);
    }

    #[test]
    fn tracker_total_cost() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            total_tokens: 1_000_000,
        };
        // gpt-4o input: $2.5 per million
        tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );
        assert!((tracker.total_cost() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn tracker_reset() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        tracker.record(
            ProviderType::OpenAi,
            "https://api.openai.com",
            "gpt-4o",
            &usage,
        );
        assert!(!tracker.all().is_empty());

        tracker.reset();
        assert!(tracker.all().is_empty());
        assert!(tracker.total_cost() == 0.0);
    }
}
