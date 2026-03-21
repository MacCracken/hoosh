# ADR-002: Cryptographic Audit Chain

**Status:** Accepted
**Date:** 2026-03-21

## Context

Hoosh proxies inference requests containing user data and API keys to paid providers. Production deployments need tamper-proof audit trails for compliance (HIPAA, SOC2) and incident investigation.

## Decision

Implement an HMAC-SHA256 linked audit chain (ported from secureyeoman's `sy-audit` pattern). Each entry is signed with `HMAC-SHA256(entry_hash:previous_hash, signing_key)`, creating an append-only chain where tampering with any entry invalidates all subsequent signatures.

## Key Design Choices

- **In-memory with bounded eviction**: `VecDeque` with configurable `max_entries` (default 10,000). `first_valid_hash` tracks the chain start after eviction so `verify()` works correctly on the surviving window.
- **Mutex with minimized critical section**: Entry construction and crypto (SHA-256, HMAC) happen outside the lock. The mutex is only held to read `last_hash` and append the entry.
- **Signing key supports `$ENV_VAR`**: Resolved via the same `resolve_api_key()` used for provider API keys. Auto-generated 32-byte key if not configured.
- **UUID v4 for entry IDs**: Consistent with the rest of the codebase (not custom hex random).

## Consequences

- Audit is opt-in (`[audit] enabled = true` in config). No overhead when disabled.
- The `/v1/audit` endpoint returns a `snapshot()` (single lock acquisition) rather than calling `verify()` on every GET.
- Full chain verification is available via `verify()` for explicit admin checks.

## Alternatives Considered

- **External audit service** (e.g., write to a log aggregator): Adds a network dependency on the hot path.
- **Database-backed audit**: Adds a persistence dependency. Can be added later on top of the chain.
- **No cryptographic integrity**: Simpler but provides no tamper detection.
