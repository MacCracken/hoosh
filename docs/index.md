# Hoosh Documentation

> AI inference gateway — multi-provider LLM routing, local model serving,
> speech-to-text, and token budget management.

---

## Architecture

- [Architecture Overview](architecture/overview.md) — system design, module structure, key types

## Development

- [Performance & Benchmarks](development/performance.md) — benchmark results, connection tuning, hot-path analysis
- [Roadmap](development/roadmap.md) — planned features, v1.0 criteria

## Decisions

- [ADR-001: HTTP Gateway Design](decisions/001-http-gateway.md) — why hoosh is an HTTP server, not a library
- [ADR-002: Cryptographic Audit Chain](decisions/002-cryptographic-audit-chain.md) — HMAC-SHA256 linked chain for tamper-proof logging
- [ADR-003: Majra Messaging](decisions/003-majra-messaging.md) — priority queues, pub/sub events, heartbeat tracking
- [ADR-004: Auth and Security](decisions/004-auth-and-security.md) — bearer tokens, rate limiting, TLS pinning, secret management

## Reference

- [BENCHMARKS.md](../BENCHMARKS.md) — raw benchmark results with hardware specs
- [CHANGELOG.md](../CHANGELOG.md) — version history
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development workflow
