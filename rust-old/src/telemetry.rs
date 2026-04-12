//! OpenTelemetry integration — OTLP export and trace context propagation.
//! Only compiled when the `otel` feature is enabled.

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::TracerProvider;

/// Guard that shuts down the OpenTelemetry tracer provider on drop.
pub struct OtelGuard {
    provider: TracerProvider,
    layer: Option<
        tracing_opentelemetry::OpenTelemetryLayer<
            tracing_subscriber::Registry,
            opentelemetry_sdk::trace::Tracer,
        >,
    >,
}

impl OtelGuard {
    /// Take the tracing layer out of the guard (can only be called once).
    pub fn layer(
        &mut self,
    ) -> Option<
        tracing_opentelemetry::OpenTelemetryLayer<
            tracing_subscriber::Registry,
            opentelemetry_sdk::trace::Tracer,
        >,
    > {
        self.layer.take()
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(e) = self.provider.shutdown() {
            eprintln!("OpenTelemetry shutdown error: {e}");
        }
    }
}

/// Initialize OpenTelemetry with an OTLP exporter.
///
/// Returns a guard whose `.layer()` method yields the tracing layer to
/// install into the subscriber. The guard must be kept alive for the
/// lifetime of the application so traces are flushed on shutdown.
pub fn init_otel(endpoint: &str, service_name: &str) -> anyhow::Result<OtelGuard> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(opentelemetry_sdk::Resource::new(vec![KeyValue::new(
            "service.name",
            service_name.to_string(),
        )]))
        .build();

    let tracer = provider.tracer("hoosh");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);

    Ok(OtelGuard {
        provider,
        layer: Some(layer),
    })
}
