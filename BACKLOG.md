# Engineering Backlog

Items identified during the Cyrius port audit (2026-04-13). The parity arc
(v2.2.x–v2.3.x) cleared all but two; remaining open work is tracked in the
[roadmap](docs/development/roadmap.md). This file is the audit-trail of the
original backlog.

## Remaining open

| ID | Priority | Issue | Status |
|----|----------|-------|--------|
| P4 | Medium | Multi-threaded accept loop | Open → roadmap **v2.4.0** (unblocked; allocator thread-safe) |
| P5 | Medium | Connection pooling | Deferred — local loopback is cheap; remote TLS-reuse gated on sandhi keep-alive |

## Cleared during the parity arc

| ID | Issue | Shipped |
|----|-------|---------|
| P1 | Tool calling / MCP bridge | Tool calling **2.2.4**; MCP server (bote) **2.3.0** |
| P2 | DLP content filtering | **2.2.2** (PII scanner + privacy-aware routing) |
| P3 | TLS for remote providers | **2.2.0** (sandhi native TLS) |
| P6 | OpenTelemetry traces | traceparent **2.3.4**; OTLP/JSON export **2.3.5** |
| P7 | Semantic cache | **2.2.3** (embedding cosine) |
| P8 | Batch inference manager | **2.3.1** sync → **2.3.2** async → **2.3.3** concurrent |
| P9 | Cost optimizer | **2.2.3** (cheapest capable model) |

## Deferred (external)

| ID | Issue | Reason |
|----|-------|--------|
| D1 | Audio endpoints (STT/TTS) | Migrating to **svara** — still pending |
| D3 | WASM target | Cyrius doesn't target WASM (non-goal) |

(D2 "DNS resolution" is **resolved** — the cyrius `sandhi`/`net` stack ships DNS;
remote provider transport landed in 2.2.0.)

## Resolved (Rust-era)

See `rust-old/` BACKLOG for historical items (M1–M10, L1–L13). All cleared before
the Cyrius port.
