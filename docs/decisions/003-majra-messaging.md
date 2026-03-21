# ADR-003: Majra for Messaging Primitives

**Status:** Accepted
**Date:** 2026-03-21

## Context

Hoosh needed three messaging primitives: a priority queue for inference requests, a pub/sub event bus for provider events, and a heartbeat tracker for provider health. Building these from scratch would duplicate work already done in the AGNOS ecosystem.

## Decision

Use the [majra](https://crates.io/crates/majra) crate (shared across AgnosAI, SecureYeoman, and other AGNOS projects) for all three:

1. **`majra::queue::PriorityQueue`** — 5-tier priority queue (Background → Critical) for `InferenceQueue`
2. **`majra::pubsub::TypedPubSub<ProviderEvent>`** — typed event bus with MQTT-style topic matching for health changes, inference events, errors
3. **`majra::heartbeat::ConcurrentHeartbeatTracker`** — Online/Suspect/Offline FSM with configurable timeouts for provider health

## Rationale

- **Reuse over rebuild**: majra is already battle-tested across the ecosystem
- **Type safety**: `TypedPubSub<T>` provides compile-time payload type checking
- **Consistent patterns**: Same abstractions used in AgnosAI's agent orchestration and SecureYeoman's node management
- **Thread-safe by default**: All majra types are `Send + Sync`, backed by DashMap and tokio broadcast channels

## Consequences

- Adds a dependency on `majra 0.21.3` (pulls in `chrono`, `dashmap`, `tokio`, `uuid` — all already transitive deps)
- The heartbeat tracker integrates with the existing `HealthMap` — both run in parallel, with the heartbeat providing the Online/Suspect/Offline state machine and the health map tracking per-check failure counts
- Event bus publishing is fire-and-forget on the inference hot path (broadcast channel, no backpressure blocking)
