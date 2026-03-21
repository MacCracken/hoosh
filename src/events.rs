//! Provider event bus — pub/sub for internal provider events.

use majra::pubsub::TypedPubSub;
use serde::{Deserialize, Serialize};

/// Events emitted by the hoosh runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderEvent {
    /// Provider health changed.
    HealthChanged {
        provider: String,
        base_url: String,
        healthy: bool,
    },
    /// Inference completed.
    InferenceCompleted {
        provider: String,
        model: String,
        latency_ms: u64,
        tokens: u32,
    },
    /// Inference failed.
    InferenceFailed {
        provider: String,
        model: String,
        error: String,
    },
    /// Provider rate limited.
    RateLimited { provider: String },
}

/// The event bus for hoosh provider events.
pub type EventBus = TypedPubSub<ProviderEvent>;

/// Create a new event bus.
pub fn new_event_bus() -> EventBus {
    TypedPubSub::new()
}

/// Topic constants for structured event routing.
pub mod topics {
    pub const HEALTH: &str = "providers/health";
    pub const INFERENCE: &str = "providers/inference";
    pub const ERRORS: &str = "providers/errors";
    pub const RATE_LIMIT: &str = "providers/rate_limit";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_bus_creation() {
        let bus = new_event_bus();
        // Should be able to create without panicking
        drop(bus);
    }

    #[test]
    fn publish_and_subscribe_roundtrip() {
        let bus = new_event_bus();
        let mut rx = bus.subscribe(topics::INFERENCE);

        let event = ProviderEvent::InferenceCompleted {
            provider: "ollama".into(),
            model: "llama3".into(),
            latency_ms: 42,
            tokens: 100,
        };
        bus.publish(topics::INFERENCE, event);

        let msg = rx.try_recv().unwrap();
        match msg.payload {
            ProviderEvent::InferenceCompleted {
                provider,
                model,
                latency_ms,
                tokens,
            } => {
                assert_eq!(provider, "ollama");
                assert_eq!(model, "llama3");
                assert_eq!(latency_ms, 42);
                assert_eq!(tokens, 100);
            }
            _ => panic!("expected InferenceCompleted event"),
        }
    }
}
