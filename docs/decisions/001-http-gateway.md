# ADR-001: HTTP Gateway Design

**Status:** Accepted
**Date:** 2026-03-20

## Context

Hoosh needs to route LLM requests across 14+ provider backends (local and remote).
The question is whether hoosh should be:

1. A library with in-process provider bindings (like litellm/CrewAI)
2. An HTTP gateway server with an OpenAI-compatible API

## Decision

Hoosh is an HTTP gateway server.

## Rationale

- **Provider isolation**: Backend crashes don't take down the consumer process
- **Language agnostic**: Any language can call the HTTP API, not just Rust
- **Deployment flexibility**: Run hoosh on a GPU box, consumers on CPU-only machines
- **Observability**: Centralized logging, metrics, and token budget tracking
- **Configuration**: Hot-reloadable TOML config without recompiling consumers

## Consequences

- **Latency overhead**: Extra HTTP hop between consumer and backend. Mitigated with:
  - TCP_NODELAY (disables Nagle's 40ms batching)
  - Connection pooling (52us warm vs 151us cold)
  - HTTP/2 multiplexing
  - Gateway overhead measured at ~0ms (within noise of direct Ollama calls)
- **Deployment complexity**: Consumers need a running hoosh instance. Mitigated by `HooshClient` crate.
- **Not in-process**: Cannot avoid serialization overhead. Acceptable because inference time (200ms+) dominates serialization (820ns).

## Alternatives Considered

- **In-process library**: Would eliminate HTTP overhead but couples consumers to Rust, prevents backend isolation, and complicates multi-language support.
- **Unix socket**: Faster than TCP for local communication but less portable. May be added later as an optimization.
