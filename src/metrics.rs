//! Prometheus metrics for the hoosh inference gateway.

use prometheus::{
    Encoder, HistogramVec, IntCounterVec, IntGauge, TextEncoder, histogram_opts, opts,
    register_histogram_vec, register_int_counter_vec, register_int_gauge,
};
use std::sync::LazyLock;

// Request counters
static REQUESTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        opts!("hoosh_requests_total", "Total inference requests"),
        &["provider", "model", "status"]
    )
    .unwrap()
});

// Request duration histogram (seconds)
static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        histogram_opts!(
            "hoosh_request_duration_seconds",
            "Inference request duration in seconds",
            vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        ),
        &["provider", "model"]
    )
    .unwrap()
});

// Active providers gauge
static PROVIDERS_CONFIGURED: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(opts!(
        "hoosh_providers_configured",
        "Number of configured providers"
    ))
    .unwrap()
});

// Token usage counters
static TOKENS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        opts!("hoosh_tokens_total", "Total tokens processed"),
        &["provider", "type"] // type = "prompt" or "completion"
    )
    .unwrap()
});

pub fn record_request(
    provider: &str,
    model: &str,
    status: &str,
    duration_secs: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
) {
    REQUESTS_TOTAL
        .with_label_values(&[provider, model, status])
        .inc();
    REQUEST_DURATION
        .with_label_values(&[provider, model])
        .observe(duration_secs);
    if prompt_tokens > 0 {
        TOKENS_TOTAL
            .with_label_values(&[provider, "prompt"])
            .inc_by(prompt_tokens as u64);
    }
    if completion_tokens > 0 {
        TOKENS_TOTAL
            .with_label_values(&[provider, "completion"])
            .inc_by(completion_tokens as u64);
    }
}

// Workflow step metrics
static WORKFLOW_STEPS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        opts!(
            "hoosh_workflow_steps_total",
            "Total workflow steps executed"
        ),
        &["step_type", "status"]
    )
    .unwrap()
});

static WORKFLOW_STEP_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        histogram_opts!(
            "hoosh_workflow_step_duration_ms",
            "Workflow step duration in milliseconds",
            vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 30000.0]
        ),
        &["step_type"]
    )
    .unwrap()
});

/// Record a workflow step execution for Prometheus.
pub fn record_workflow_step(step_type: &str, status: &str, duration_ms: u64) {
    WORKFLOW_STEPS.with_label_values(&[step_type, status]).inc();
    WORKFLOW_STEP_DURATION
        .with_label_values(&[step_type])
        .observe(duration_ms as f64);
}

pub fn set_providers_configured(count: i64) {
    PROVIDERS_CONFIGURED.set(count);
}

/// Render all metrics in Prometheus text format.
pub fn gather() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .unwrap_or_default();
    String::from_utf8(buffer).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_returns_valid_prometheus_text() {
        let output = gather();
        // Prometheus text format: lines are either comments (# ...) or metric lines
        // An empty output is also valid (no metrics recorded yet).
        for line in output.lines() {
            assert!(
                line.starts_with('#') || line.contains(' '),
                "unexpected line format: {line}"
            );
        }
    }

    #[test]
    fn record_request_increments_counters() {
        record_request("test_prov", "test_model", "success", 1.5, 100, 50);

        let output = gather();
        assert!(
            output.contains("hoosh_requests_total"),
            "should contain requests_total metric"
        );
        assert!(
            output.contains("hoosh_request_duration_seconds"),
            "should contain request_duration metric"
        );
        assert!(
            output.contains("hoosh_tokens_total"),
            "should contain tokens_total metric"
        );
    }

    #[test]
    fn set_providers_configured_updates_gauge() {
        set_providers_configured(42);
        let output = gather();
        assert!(
            output.contains("hoosh_providers_configured"),
            "should contain providers_configured metric"
        );
        assert!(output.contains("42"), "gauge should reflect the set value");
    }
}
