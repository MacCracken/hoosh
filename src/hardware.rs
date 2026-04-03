//! Hardware-aware model placement using ai-hwaccel.
//!
//! Wraps the `ai-hwaccel` crate to provide:
//! - Hardware detection with optional disk caching
//! - Model placement recommendations (provider + quantization)
//! - Multi-GPU sharding plans
//! - System I/O topology for throughput estimation
//! - Device family filtering (GPU / NPU / TPU)

use std::sync::Arc;
use std::time::Duration;

use ai_hwaccel::{
    AcceleratorFamily, AcceleratorProfile, AcceleratorRegistry, Backend, DiskCachedRegistry,
    RuntimeEnvironment, ShardingStrategy, SystemIo, TimedDetection,
    cost::{self, CloudProvider, InstanceRecommendation},
};

use serde::{Deserialize, Serialize};

// ─── Detection ──────────────────────────────────────────────────────────────

/// Manages hardware detection and model placement recommendations.
pub struct HardwareManager {
    registry: AcceleratorRegistry,
    /// Per-backend detection timings (populated only via `detect_with_timing`).
    detection_timings: Option<TimedDetection>,
}

impl HardwareManager {
    /// Run hardware discovery and create a new manager.
    pub fn detect() -> Self {
        Self {
            registry: AcceleratorRegistry::builder().detect(),
            detection_timings: None,
        }
    }

    /// Run hardware discovery with per-backend timing information.
    ///
    /// The timings are available via [`HardwareManager::detection_timing_summary`].
    pub fn detect_with_timing() -> Self {
        let timed = AcceleratorRegistry::detect_with_timing();
        Self {
            registry: timed.registry.clone(),
            detection_timings: Some(timed),
        }
    }

    /// Create a manager from a disk-cached registry.
    ///
    /// Re-uses a cached detection result if it is younger than `ttl`,
    /// avoiding expensive re-probing on every startup. Cache lives at
    /// `$XDG_CACHE_HOME/ai-hwaccel/registry.json`.
    pub fn from_cache(ttl: Duration) -> Self {
        let cache = DiskCachedRegistry::new(ttl);
        tracing::info!(
            cache_path = %cache.cache_path().display(),
            ttl_secs = ttl.as_secs(),
            "hardware detection using disk cache"
        );
        let registry_arc: Arc<AcceleratorRegistry> = cache.get();
        // Clone out of Arc — the cache keeps its own copy.
        let registry = (*registry_arc).clone();
        Self {
            registry,
            detection_timings: None,
        }
    }

    /// Run hardware discovery with selective backend control.
    ///
    /// Disables backends whose names appear in `disabled` (case-insensitive).
    /// Unknown names are logged as warnings and ignored.
    pub fn detect_selective(disabled: &[String]) -> Self {
        if disabled.is_empty() {
            return Self::detect();
        }
        let mut builder = AcceleratorRegistry::builder();
        for name in disabled {
            match parse_backend(name) {
                Some(b) => {
                    builder = builder.without(b);
                    tracing::info!(backend = %name, "hardware backend disabled by config");
                }
                None => {
                    tracing::warn!(backend = %name, "unknown hardware backend in disabled list");
                }
            }
        }
        Self {
            registry: builder.detect(),
            detection_timings: None,
        }
    }

    /// Create a manager from an existing registry (for periodic refresh).
    pub fn from_registry(registry: AcceleratorRegistry) -> Self {
        Self {
            registry,
            detection_timings: None,
        }
    }

    // ─── Basic queries ──────────────────────────────────────────────────

    /// Check if any hardware accelerator is available.
    #[must_use]
    #[inline]
    pub fn has_accelerator(&self) -> bool {
        self.registry.has_accelerator()
    }

    /// Total accelerator memory in bytes (excludes CPU).
    #[must_use]
    #[inline]
    pub fn total_accelerator_memory(&self) -> u64 {
        self.registry.total_accelerator_memory()
    }

    /// All detected accelerator profiles (including CPU).
    #[must_use]
    #[inline]
    pub fn all_profiles(&self) -> &[AcceleratorProfile] {
        self.registry.all_profiles()
    }

    /// Only available (non-errored) accelerator profiles.
    #[must_use]
    pub fn available_profiles(&self) -> Vec<&AcceleratorProfile> {
        self.registry.available().collect()
    }

    // ─── Device filtering ───────────────────────────────────────────────

    /// The single best accelerator (by memory × throughput), if any.
    #[must_use]
    pub fn best_device(&self) -> Option<&AcceleratorProfile> {
        self.registry.best_available()
    }

    /// All devices in a given family (GPU, NPU, TPU, etc.).
    #[must_use]
    pub fn devices_by_family(&self, family: AcceleratorFamily) -> Vec<&AcceleratorProfile> {
        self.registry.by_family(family).collect()
    }

    /// All GPUs.
    #[must_use]
    pub fn gpus(&self) -> Vec<&AcceleratorProfile> {
        self.registry.by_family(AcceleratorFamily::Gpu).collect()
    }

    /// All NPUs (neural processing units).
    #[must_use]
    pub fn npus(&self) -> Vec<&AcceleratorProfile> {
        self.registry.by_family(AcceleratorFamily::Npu).collect()
    }

    /// All TPUs (tensor processing units).
    #[must_use]
    pub fn tpus(&self) -> Vec<&AcceleratorProfile> {
        self.registry.by_family(AcceleratorFamily::Tpu).collect()
    }

    // ─── Model placement ────────────────────────────────────────────────

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

    // ─── Sharding ───────────────────────────────────────────────────────

    /// Generate a multi-device sharding plan for a model.
    ///
    /// Returns a plan describing how to split the model across available
    /// accelerators (pipeline parallel, tensor parallel, or single-device).
    #[must_use]
    pub fn plan_sharding(&self, model_params: u64) -> ShardingSummary {
        let quant = self.registry.suggest_quantization(model_params);
        let plan = self.registry.plan_sharding(model_params, &quant);

        let device_count = plan.shards().len();
        let strategy_name = format!("{}", plan.strategy);

        ShardingSummary {
            strategy: plan.strategy.clone(),
            strategy_name,
            device_count,
            total_memory_bytes: plan.total_memory_bytes,
            estimated_tokens_per_sec: plan.estimated_tokens_per_sec,
            quantization: format!("{quant}"),
            shards: plan
                .shards()
                .iter()
                .map(|s| ShardInfo {
                    shard_id: s.shard_id,
                    layer_range: s.layer_range,
                    device: format!("{}", s.device),
                    memory_bytes: s.memory_bytes,
                })
                .collect(),
        }
    }

    // ─── System I/O ─────────────────────────────────────────────────────

    /// System I/O topology: interconnects, storage, runtime environment.
    #[must_use]
    pub fn system_io(&self) -> &SystemIo {
        self.registry.system_io()
    }

    /// Whether high-speed interconnects (NVLink, InfiniBand, etc.) are present.
    #[must_use]
    pub fn has_fast_interconnect(&self) -> bool {
        self.registry.system_io().has_interconnect()
    }

    /// Estimate seconds to load a dataset of `bytes` size from storage.
    #[must_use]
    pub fn estimate_data_load_secs(&self, bytes: u64) -> Option<f64> {
        self.registry.system_io().estimate_ingestion_secs(bytes)
    }

    // ─── VRAM-aware placement ─────────────────────────────────────────

    /// Available accelerator VRAM in bytes after accounting for current usage and
    /// a reservation for non-inference GPU workloads (mabda compute, etc.).
    ///
    /// Returns `total_accelerator_memory - used - reserved`, floored at 0.
    #[must_use]
    pub fn available_vram(&self, reserved: u64) -> u64 {
        let total = self.registry.total_accelerator_memory();
        let used: u64 = self
            .registry
            .all_profiles()
            .iter()
            .filter(|p| p.available && !matches!(p.accelerator, ai_hwaccel::AcceleratorType::Cpu))
            .filter_map(|p| p.memory_used_bytes)
            .sum();
        total.saturating_sub(used).saturating_sub(reserved)
    }

    /// Whether a model with `model_params` parameters fits in available VRAM
    /// (after reserving `reserved` bytes for other GPU workloads).
    #[must_use]
    #[inline]
    pub fn fits_model(&self, model_params: u64, reserved: u64) -> bool {
        let quant = self.registry.suggest_quantization(model_params);
        let estimated = AcceleratorRegistry::estimate_memory(model_params, &quant);
        estimated <= self.available_vram(reserved)
    }

    /// Build GPU telemetry snapshots for the heartbeat tracker.
    ///
    /// Maps real runtime fields from `AcceleratorProfile` into `majra`'s
    /// `GpuTelemetry` struct — utilization, memory, temperature.
    #[must_use]
    pub fn gpu_telemetry(&self) -> Vec<majra::heartbeat::GpuTelemetry> {
        self.gpus()
            .iter()
            .map(|gpu| majra::heartbeat::GpuTelemetry {
                utilization_pct: gpu.gpu_utilization_percent.unwrap_or(0) as f32,
                memory_used_mb: gpu.memory_used_bytes.unwrap_or(0) / (1024 * 1024),
                memory_total_mb: gpu.memory_bytes / (1024 * 1024),
                temperature_c: gpu.temperature_c.map(|t| t as f32),
            })
            .collect()
    }

    /// Runtime environment: Docker, Kubernetes, cloud instance metadata.
    #[must_use]
    #[inline]
    pub fn runtime_environment(&self) -> Option<&RuntimeEnvironment> {
        self.registry.system_io().environment.as_ref()
    }

    // ─── Cloud cost ─────────────────────────────────────────────────────

    /// Recommend the cheapest viable cloud GPU instance(s) for a model.
    ///
    /// Returns instances sorted by price. Optionally filter to a specific
    /// cloud provider (AWS, GCP, Azure).
    #[must_use]
    pub fn recommend_cloud_instance(
        &self,
        model_params: u64,
        provider: Option<CloudProvider>,
    ) -> Vec<InstanceRecommendation> {
        let quant = self.registry.suggest_quantization(model_params);
        cost::recommend_instance(model_params, &quant, provider)
    }

    // ─── Diagnostics ────────────────────────────────────────────────────

    /// Per-backend detection timing summary (only if created with `detect_with_timing`).
    #[must_use]
    pub fn detection_timing_summary(&self) -> Option<Vec<(String, Duration)>> {
        self.detection_timings.as_ref().map(|t| {
            let mut timings: Vec<(String, Duration)> = t.timings.clone().into_iter().collect();
            timings.sort_by(|a, b| b.1.cmp(&a.1)); // slowest first
            timings
        })
    }

    /// Total detection time (only if created with `detect_with_timing`).
    #[must_use]
    pub fn total_detection_time(&self) -> Option<Duration> {
        self.detection_timings.as_ref().map(|t| t.total)
    }

    /// Human-readable hardware summary for `hoosh info`.
    pub fn summary(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for p in self.registry.all_profiles() {
            let mem_gb = p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            // Prefer device_name (e.g. "RTX 4090") over accelerator type enum
            let fallback = p.accelerator.to_string();
            let name = p.device_name.as_deref().unwrap_or(&fallback);
            let mut detail = format!("  {name} ({:.1} GB", mem_gb);

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

        // Show interconnects
        let sio = self.registry.system_io();
        for ic in &sio.interconnects {
            lines.push(format!(
                "  {} {}: {:.0} GB/s",
                ic.kind, ic.name, ic.bandwidth_gbps
            ));
        }

        // Show warnings from detection
        for w in self.registry.warnings() {
            lines.push(format!("  warning: {w}"));
        }

        // Show detection timing if available
        if let Some(total) = self.total_detection_time() {
            lines.push(format!("  detection: {:.0}ms", total.as_millis()));
        }

        lines
    }

    /// Access the underlying registry for advanced queries.
    #[must_use]
    pub fn registry(&self) -> &AcceleratorRegistry {
        &self.registry
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a backend name string into a `Backend` variant (case-insensitive).
fn parse_backend(name: &str) -> Option<Backend> {
    match name.to_ascii_lowercase().as_str() {
        "cuda" => Some(Backend::Cuda),
        "rocm" => Some(Backend::Rocm),
        "apple" => Some(Backend::Apple),
        "vulkan" => Some(Backend::Vulkan),
        "intel_npu" | "intel-npu" => Some(Backend::IntelNpu),
        "amd_xdna" | "amd-xdna" => Some(Backend::AmdXdna),
        "tpu" => Some(Backend::Tpu),
        "gaudi" => Some(Backend::Gaudi),
        "aws_neuron" | "aws-neuron" => Some(Backend::AwsNeuron),
        "intel_oneapi" | "intel-oneapi" => Some(Backend::IntelOneApi),
        "qualcomm" => Some(Backend::Qualcomm),
        "cerebras" => Some(Backend::Cerebras),
        "graphcore" => Some(Backend::Graphcore),
        "groq" => Some(Backend::Groq),
        "samsung_npu" | "samsung-npu" => Some(Backend::SamsungNpu),
        "mediatek_apu" | "mediatek-apu" => Some(Backend::MediaTekApu),
        "windows_wmi" | "windows-wmi" => Some(Backend::WindowsWmi),
        _ => None,
    }
}

// ─── Output types ───────────────────────────────────────────────────────────

/// Recommendation for where and how to run a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Summary of a multi-device sharding plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingSummary {
    /// Sharding strategy (None, PipelineParallel, TensorParallel, DataParallel).
    pub strategy: ShardingStrategy,
    /// Human-readable strategy name.
    pub strategy_name: String,
    /// Number of devices used.
    pub device_count: usize,
    /// Total memory required across all shards.
    pub total_memory_bytes: u64,
    /// Estimated throughput in tokens/sec, if calculable.
    pub estimated_tokens_per_sec: Option<f64>,
    /// Quantization level used for the plan.
    pub quantization: String,
    /// Per-shard breakdown.
    pub shards: Vec<ShardInfo>,
}

/// Information about a single model shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard identifier.
    pub shard_id: u32,
    /// Layer range (start, end) assigned to this shard.
    pub layer_range: (u32, u32),
    /// Device this shard runs on.
    pub device: String,
    /// Memory required by this shard.
    pub memory_bytes: u64,
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
    fn detect_with_timing_has_timings() {
        let hw = HardwareManager::detect_with_timing();
        assert!(hw.total_detection_time().is_some());
        let timings = hw.detection_timing_summary().unwrap();
        // At least one backend was probed
        assert!(!timings.is_empty() || hw.all_profiles().is_empty());
    }

    #[test]
    fn summary_is_nonempty() {
        let hw = HardwareManager::detect();
        let lines = hw.summary();
        assert!(!lines.is_empty());
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

        let rec2 = rec.clone();
        assert_eq!(rec2.estimated_memory_bytes, 4_000_000_000);
    }

    #[test]
    fn device_family_filtering() {
        let hw = HardwareManager::detect();
        // These should not panic even without hardware
        let _ = hw.gpus();
        let _ = hw.npus();
        let _ = hw.tpus();
        let _ = hw.best_device();
        let _ = hw.available_profiles();
    }

    #[test]
    fn sharding_plan_generation() {
        let hw = HardwareManager::detect();
        let plan = hw.plan_sharding(7_000_000_000);
        assert!(plan.total_memory_bytes > 0);
        assert!(!plan.quantization.is_empty());
        assert!(!plan.strategy_name.is_empty());
    }

    #[test]
    fn system_io_accessible() {
        let hw = HardwareManager::detect();
        let sio = hw.system_io();
        // Should not panic — returns empty on systems without interconnects
        let _ = sio.interconnects.len();
        let _ = sio.storage.len();
        let _ = hw.has_fast_interconnect();
    }

    #[test]
    fn data_load_estimate() {
        let hw = HardwareManager::detect();
        // May return None if no storage info detected, that's fine
        let _ = hw.estimate_data_load_secs(10_000_000_000);
    }

    #[test]
    fn cached_detection() {
        let hw = HardwareManager::from_cache(Duration::from_secs(300));
        // Should work the same as direct detection
        let _ = hw.has_accelerator();
        let _ = hw.summary();
    }

    #[test]
    fn selective_detection_disables_backends() {
        let disabled = vec!["vulkan".into(), "tpu".into()];
        let hw = HardwareManager::detect_selective(&disabled);
        // Should not panic — CPU is always available
        let _ = hw.has_accelerator();
        let _ = hw.summary();
    }

    #[test]
    fn selective_detection_empty_is_full() {
        let hw = HardwareManager::detect_selective(&[]);
        let _ = hw.has_accelerator();
    }

    #[test]
    fn selective_detection_unknown_backend_ignored() {
        let disabled = vec!["nonexistent_backend".into()];
        let hw = HardwareManager::detect_selective(&disabled);
        let _ = hw.has_accelerator();
    }

    #[test]
    fn available_vram_without_gpus() {
        let hw = HardwareManager::detect();
        if !hw.has_accelerator() {
            assert_eq!(hw.available_vram(0), 0);
            assert_eq!(hw.available_vram(1024), 0);
        }
    }

    #[test]
    fn available_vram_reservation_saturates() {
        let hw = HardwareManager::detect();
        // Reserving more than total should not underflow
        let result = hw.available_vram(u64::MAX);
        assert_eq!(result, 0);
    }

    #[test]
    fn fits_model_without_gpus() {
        let hw = HardwareManager::detect();
        if !hw.has_accelerator() {
            // Without accelerators, nothing fits in VRAM
            assert!(!hw.fits_model(7_000_000_000, 0));
        }
    }

    #[test]
    fn gpu_telemetry_no_panic() {
        let hw = HardwareManager::detect();
        let telemetry = hw.gpu_telemetry();
        // On systems without GPUs, returns empty vec
        if !hw.has_accelerator() {
            assert!(telemetry.is_empty());
        }
        for t in &telemetry {
            assert!(t.memory_total_mb > 0);
        }
    }

    #[test]
    fn runtime_environment_accessible() {
        let hw = HardwareManager::detect();
        // May be None if not in container/cloud — that's expected
        let _ = hw.runtime_environment();
    }

    #[test]
    fn cloud_cost_recommendations() {
        let hw = HardwareManager::detect();
        let recs = hw.recommend_cloud_instance(70_000_000_000, None);
        // Should return at least some instances for a 70B model
        assert!(!recs.is_empty());
        // Should be sorted by price (cheapest first)
        for window in recs.windows(2) {
            assert!(window[0].instance.price_per_hour <= window[1].instance.price_per_hour);
        }
    }

    #[test]
    fn cloud_cost_provider_filter() {
        let hw = HardwareManager::detect();
        let aws_recs = hw.recommend_cloud_instance(7_000_000_000, Some(CloudProvider::Aws));
        for rec in &aws_recs {
            assert_eq!(rec.instance.provider, "aws");
        }
    }

    #[test]
    fn from_registry_constructor() {
        let registry = AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(
            16 * 1024 * 1024 * 1024,
        )]);
        let hw = HardwareManager::from_registry(registry);
        assert!(!hw.has_accelerator()); // CPU only
        assert_eq!(hw.available_vram(0), 0);
    }

    #[test]
    fn placement_recommendation_serializes() {
        let rec = PlacementRecommendation {
            provider: "ollama".into(),
            quantization: Some("Q4_K_M".into()),
            estimated_memory_bytes: 4_000_000_000,
            fits_in_vram: true,
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("ollama"));
        assert!(json.contains("Q4_K_M"));
    }

    #[test]
    fn parse_backend_known() {
        assert_eq!(parse_backend("cuda"), Some(Backend::Cuda));
        assert_eq!(parse_backend("ROCM"), Some(Backend::Rocm));
        assert_eq!(parse_backend("Vulkan"), Some(Backend::Vulkan));
        assert_eq!(parse_backend("intel-npu"), Some(Backend::IntelNpu));
        assert_eq!(parse_backend("intel_npu"), Some(Backend::IntelNpu));
        assert_eq!(parse_backend("amd-xdna"), Some(Backend::AmdXdna));
    }

    #[test]
    fn parse_backend_unknown() {
        assert_eq!(parse_backend("nonexistent"), None);
        assert_eq!(parse_backend(""), None);
    }
}
