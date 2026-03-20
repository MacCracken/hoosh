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
            lines.push(format!("  {} ({:.1} GB)", p.accelerator, mem_gb));
        }
        if lines.is_empty() {
            lines.push("  No hardware accelerators detected".into());
        }
        lines
    }
}
