# Hoosh Documentation

> AI inference gateway — multi-provider LLM routing, local model serving, and
> token-budget management, written in Cyrius.

---

## Architecture

- [Architecture Overview](architecture/overview.md) — system design, module map, dependencies, endpoints

## Development

- [Roadmap](development/roadmap.md) — shipped summary + open/planned work
- [Current State](development/state.md) — volatile state snapshot (version, sizes, counts), per release
- [Performance & Benchmarks](development/performance.md) — the bench system + how to run it
- [Doc Health](doc-health.md) — documentation currency tracker

## Decisions

- [ADR-001: HTTP Gateway Design](decisions/001-http-gateway.md) — why hoosh is an HTTP server, not a library
- [ADR-002: Cryptographic Audit Chain](decisions/002-cryptographic-audit-chain.md) — HMAC-SHA256 linked chain for tamper-proof logging
- [ADR-003: Majra Messaging](decisions/003-majra-messaging.md) — priority queues, pub/sub events, heartbeat tracking
- [ADR-004: Auth and Security](decisions/004-auth-and-security.md) — bearer tokens, rate limiting, TLS pinning, secret management
- [ADR-005: MCP via Bote + Szál](decisions/005-mcp-via-bote.md) — MCP tool use via bote (protocol) + szál (58 tools)
- [ADR-006: Kavach Tool Sandbox](decisions/006-kavach-tool-sandbox.md) — sandboxed tool execution with secret scanning
- [ADR-007: Cyrius 6 Modernization](decisions/007-cyrius-6-modernization.md) — toolchain/scaffolding to Cyrius 6.x
- [ADR-008: Persistence via Patra](decisions/008-persistence-via-patra.md) — opt-in audit-chain + budget persistence
- [ADR-009: Concurrent Batch Inference](decisions/009-concurrent-batch-inference.md) — worker/crypto-lane pools, thread-safety
- [ADR-010: Observability](decisions/010-observability.md) — latency histograms, event bus, traceparent, OTLP export
- [ADR-011: Multi-threaded accept loop](decisions/011-multithreaded-accept-loop.md) — unified 7-worker pool, crypto-bank budget, synchronization pass

## Reference

- [benchmarks.md](../benchmarks.md) — raw benchmark results with hardware specs
- [CHANGELOG.md](../CHANGELOG.md) — version history
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development workflow
