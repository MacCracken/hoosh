# Engineering Backlog

Items identified during Cyrius port audit (2026-04-13). Sorted by priority.

## Open Items

| ID | Priority | Issue | Notes |
|----|----------|-------|-------|
| P1 | High | Tool calling / MCP bridge | Needs bote/szál Cyrius port |
| P2 | High | DLP content filtering | PII patterns, classification levels, privacy-aware routing |
| P3 | High | TLS for remote providers | Blocked on Cyrius TLS stdlib |
| P4 | Medium | Multi-threaded accept loop | Single-threaded limits throughput |
| P5 | Medium | Connection pooling | New socket per backend request |
| P6 | Medium | OpenTelemetry traces | Distributed tracing propagation |
| P7 | Low | Semantic cache | Cosine similarity over embeddings |
| P8 | Low | Batch inference manager | Concurrent requests with semaphore |
| P9 | Low | Cost optimizer | Cheapest capable model recommendation |

## Deferred

| ID | Issue | Reason |
|----|-------|--------|
| D1 | Audio endpoints (STT/TTS) | Migrating to svara — still pending |
| D2 | DNS resolution | Cyrius stdlib doesn't have DNS yet |
| D3 | WASM target | Cyrius doesn't target WASM |

## Resolved (Rust-era)

See `rust-old/` BACKLOG for historical items (M1–M10, L1–L13). All cleared before Cyrius port.
