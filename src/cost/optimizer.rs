//! Cost optimizer — dynamic model selection based on workload complexity.
//!
//! Given a request's characteristics (token count, tool use, vision), recommends
//! the cheapest model from available routes that meets capability requirements.

use crate::provider::metadata::{Modality, ModelMetadataRegistry, ModelTier};
use crate::router::ProviderRoute;

use serde::{Deserialize, Serialize};

/// A model recommendation from the cost optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// Recommended model ID.
    pub model: String,
    /// Model tier.
    pub tier: ModelTier,
    /// Estimated cost in USD for the request.
    pub estimated_cost: f64,
    /// Reason for the recommendation.
    pub reason: String,
}

/// Characteristics of a request used for optimization.
#[derive(Debug, Clone)]
pub struct RequestProfile {
    /// Estimated input tokens.
    pub input_tokens: u32,
    /// Requested max output tokens.
    pub max_output_tokens: u32,
    /// Whether the request uses tool/function calling.
    pub uses_tools: bool,
    /// Whether the request contains images.
    pub has_vision: bool,
    /// Whether the request has a system prompt.
    pub has_system_prompt: bool,
}

/// Cost optimizer that recommends models based on workload complexity.
pub struct CostOptimizer {
    enabled: bool,
}

impl CostOptimizer {
    /// Create a new cost optimizer.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Recommend a model for the given request profile.
    ///
    /// Returns `None` if optimization is disabled or no suitable model is found.
    #[must_use]
    pub fn recommend(
        &self,
        profile: &RequestProfile,
        routes: &[ProviderRoute],
        registry: &ModelMetadataRegistry,
    ) -> Option<ModelRecommendation> {
        if !self.enabled {
            return None;
        }

        let required_tier = self.classify_complexity(profile);
        let required_modalities = self.required_modalities(profile);

        // Collect all models from routes that are available in the registry
        let mut candidates: Vec<(&str, &crate::provider::metadata::ModelMetadata, f64)> =
            Vec::new();

        for route in routes {
            if !route.enabled {
                continue;
            }
            for pattern in &route.model_patterns {
                // For exact model names (no wildcards), check the registry
                if !pattern.contains('*')
                    && let Some(meta) = registry.get(pattern)
                    && self.meets_requirements(meta, required_tier, &required_modalities, profile)
                {
                    let cost = estimate_cost(
                        pattern,
                        route.provider,
                        profile.input_tokens,
                        profile.max_output_tokens,
                    );
                    candidates.push((pattern, meta, cost));
                }
            }
        }

        // Sort by cost (cheapest first)
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .first()
            .map(|(model, meta, cost)| ModelRecommendation {
                model: (*model).to_string(),
                tier: meta.tier,
                estimated_cost: *cost,
                reason: format!("{:?} tier, est. ${:.6}", meta.tier, cost),
            })
    }

    /// Classify request complexity into a minimum required model tier.
    #[must_use]
    fn classify_complexity(&self, profile: &RequestProfile) -> ModelTier {
        if profile.has_vision {
            return ModelTier::Standard;
        }
        if profile.uses_tools && profile.input_tokens > 2000 {
            return ModelTier::Standard;
        }
        if profile.input_tokens > 10_000 {
            return ModelTier::Standard;
        }
        ModelTier::Economy
    }

    /// Determine required modalities.
    #[must_use]
    fn required_modalities(&self, profile: &RequestProfile) -> Vec<Modality> {
        let mut mods = vec![Modality::Text];
        if profile.has_vision {
            mods.push(Modality::Vision);
        }
        mods
    }

    /// Check if a model meets the requirements.
    #[must_use]
    fn meets_requirements(
        &self,
        meta: &crate::provider::metadata::ModelMetadata,
        _min_tier: ModelTier,
        required_modalities: &[Modality],
        profile: &RequestProfile,
    ) -> bool {
        // Check modalities
        for modality in required_modalities {
            if !meta.modalities.contains(modality) {
                return false;
            }
        }
        // Check tool support
        if profile.uses_tools && !meta.capabilities.tool_use {
            return false;
        }
        // Check system prompt support
        if profile.has_system_prompt && !meta.supports_system_prompt {
            return false;
        }
        // Check context window is large enough
        let total_tokens = profile
            .input_tokens
            .saturating_add(profile.max_output_tokens);
        if total_tokens > meta.context_window {
            return false;
        }
        true
    }

    /// Whether the optimizer is enabled.
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Estimate cost for a request on a given model/provider.
fn estimate_cost(
    model: &str,
    provider: crate::provider::ProviderType,
    input_tokens: u32,
    output_tokens: u32,
) -> f64 {
    let pricing = super::lookup_pricing(model, provider);
    let input_cost = (input_tokens as f64 / 1_000_000.0) * pricing.input_per_million;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * pricing.output_per_million;
    input_cost + output_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_disabled_returns_none() {
        let optimizer = CostOptimizer::new(false);
        let profile = RequestProfile {
            input_tokens: 100,
            max_output_tokens: 100,
            uses_tools: false,
            has_vision: false,
            has_system_prompt: false,
        };
        assert!(
            optimizer
                .recommend(&profile, &[], &ModelMetadataRegistry::new())
                .is_none()
        );
    }

    #[test]
    fn complexity_small_request_economy() {
        let optimizer = CostOptimizer::new(true);
        let profile = RequestProfile {
            input_tokens: 50,
            max_output_tokens: 100,
            uses_tools: false,
            has_vision: false,
            has_system_prompt: false,
        };
        assert_eq!(optimizer.classify_complexity(&profile), ModelTier::Economy);
    }

    #[test]
    fn complexity_vision_requires_standard() {
        let optimizer = CostOptimizer::new(true);
        let profile = RequestProfile {
            input_tokens: 50,
            max_output_tokens: 100,
            uses_tools: false,
            has_vision: true,
            has_system_prompt: false,
        };
        assert_eq!(optimizer.classify_complexity(&profile), ModelTier::Standard);
    }

    #[test]
    fn complexity_large_input_requires_standard() {
        let optimizer = CostOptimizer::new(true);
        let profile = RequestProfile {
            input_tokens: 15_000,
            max_output_tokens: 1000,
            uses_tools: false,
            has_vision: false,
            has_system_prompt: false,
        };
        assert_eq!(optimizer.classify_complexity(&profile), ModelTier::Standard);
    }

    #[test]
    fn estimate_cost_basic() {
        let cost = estimate_cost(
            "gpt-4o-mini",
            crate::provider::ProviderType::OpenAi,
            1000,
            500,
        );
        assert!(cost > 0.0);
        assert!(cost < 0.01); // should be very cheap for gpt-4o-mini
    }

    #[test]
    fn is_enabled() {
        assert!(CostOptimizer::new(true).is_enabled());
        assert!(!CostOptimizer::new(false).is_enabled());
    }
}
