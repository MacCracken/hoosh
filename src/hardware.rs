//! Hardware-aware model placement using ai-hwaccel.

use ai_hwaccel::AcceleratorRegistry;

/// Manages hardware detection and model placement recommendations.
pub struct HardwareManager {
    registry: AcceleratorRegistry,
}

/// Recommendation for where and how to run a model.
#[derive(Debug, Clone)]
pub struct PlacementRecommendation {
    /// Suggested provider type name.
    pub provider: String,
    /// Suggested quantization level (e.g. "Q4_K_M").
    pub quantization: Option<String>,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: u64,
    /// Whether the model fits in accelerator VRAM.
    pub fits_in_vram: bool,
}

impl HardwareManager {
    /// Run hardware discovery and create a new manager.
    pub fn detect() -> Self {
        Self {
            registry: AcceleratorRegistry::builder().detect(),
        }
    }

    /// Check if any hardware accelerator is available.
    pub fn has_accelerator(&self) -> bool {
        self.registry.has_accelerator()
    }

    /// Total accelerator memory in bytes.
    pub fn total_accelerator_memory(&self) -> u64 {
        self.registry.total_accelerator_memory()
    }

    /// Recommend placement for a model with the given parameter count.
    pub fn recommend_placement(
        &self,
        model_params: u64,
        available_providers: &[String],
    ) -> PlacementRecommendation {
        let quant = self.registry.suggest_quantization(model_params);
        let estimated = AcceleratorRegistry::estimate_memory(model_params, &quant);
        let total_vram = self.registry.total_accelerator_memory();
        let fits = estimated <= total_vram && total_vram > 0;

        let quantization_str = format!("{quant}");
        let quantization = if quantization_str == "None" || quantization_str.is_empty() {
            None
        } else {
            Some(quantization_str)
        };

        // Prefer GPU-capable local provider if it fits
        let provider = if fits {
            available_providers
                .iter()
                .find(|p| *p == "llamacpp" || *p == "ollama")
                .cloned()
                .unwrap_or_else(|| {
                    available_providers
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "ollama".into())
                })
        } else {
            available_providers
                .first()
                .cloned()
                .unwrap_or_else(|| "ollama".into())
        };

        PlacementRecommendation {
            provider,
            quantization,
            estimated_memory_bytes: estimated,
            fits_in_vram: fits,
        }
    }

    /// Human-readable hardware summary for `hoosh info`.
    pub fn summary(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for p in self.registry.all_profiles() {
            let mem_gb = p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let mut detail = format!("  {} ({:.1} GB", p.accelerator, mem_gb);

            // Show free VRAM if available
            if let Some(free) = p.memory_free_bytes
                && free > 0
            {
                let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
                detail.push_str(&format!(", {free_gb:.1} GB free"));
            }

            // Show bandwidth if available
            if let Some(bw) = p.memory_bandwidth_gbps
                && bw > 0.0
            {
                detail.push_str(&format!(", {bw:.0} GB/s"));
            }

            detail.push(')');

            // Show power/thermal if available
            let temp = p.temperature_c.unwrap_or(0);
            let power = p.power_watts.unwrap_or(0.0);
            let util = p.gpu_utilization_percent.unwrap_or(0);
            if temp > 0 || power > 0.0 {
                let mut extras = Vec::new();
                if temp > 0 {
                    extras.push(format!("{temp}°C"));
                }
                if power > 0.0 {
                    extras.push(format!("{power:.0}W"));
                }
                if util > 0 {
                    extras.push(format!("{util}% util"));
                }
                detail.push_str(&format!(" [{}]", extras.join(", ")));
            }

            lines.push(detail);
        }
        if lines.is_empty() {
            lines.push("  No hardware accelerators detected".into());
        }

        // Show warnings from detection
        for w in self.registry.warnings() {
            lines.push(format!("  warning: {w}"));
        }

        lines
    }

    /// Access the underlying registry for advanced queries.
    pub fn registry(&self) -> &AcceleratorRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_creates_manager() {
        let hw = HardwareManager::detect();
        // Should not panic — detection works even without GPUs
        let _ = hw.has_accelerator();
        let _ = hw.total_accelerator_memory();
    }

    #[test]
    fn summary_is_nonempty() {
        let hw = HardwareManager::detect();
        let lines = hw.summary();
        assert!(!lines.is_empty());
        // Without GPUs in CI, should say "No hardware accelerators detected"
        // With GPUs, should list them — either way, at least one line
    }

    #[test]
    fn recommend_placement_defaults_to_ollama() {
        let hw = HardwareManager::detect();
        let rec = hw.recommend_placement(7_000_000_000, &[]);
        assert_eq!(rec.provider, "ollama");
        assert!(rec.estimated_memory_bytes > 0);
    }

    #[test]
    fn recommend_placement_uses_first_available() {
        let hw = HardwareManager::detect();
        let providers = vec!["localai".to_string(), "llamacpp".to_string()];
        let rec = hw.recommend_placement(7_000_000_000, &providers);
        // Without VRAM it should pick the first available
        if !hw.has_accelerator() {
            assert_eq!(rec.provider, "localai");
            assert!(!rec.fits_in_vram);
        }
    }

    #[test]
    fn placement_recommendation_fields() {
        let rec = PlacementRecommendation {
            provider: "ollama".into(),
            quantization: Some("Q4_K_M".into()),
            estimated_memory_bytes: 4_000_000_000,
            fits_in_vram: true,
        };
        assert_eq!(rec.provider, "ollama");
        assert_eq!(rec.quantization.as_deref(), Some("Q4_K_M"));
        assert!(rec.fits_in_vram);

        // Clone works
        let rec2 = rec.clone();
        assert_eq!(rec2.estimated_memory_bytes, 4_000_000_000);
    }
}
