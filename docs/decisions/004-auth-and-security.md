# ADR-004: Authentication and Security Design

**Status:** Accepted
**Date:** 2026-03-21

## Context

Hoosh proxies API keys to paid providers and handles user inference data. It needs authentication, but the threat model varies: local development (no auth needed) vs production (mandatory auth).

## Decision

### Bearer Token Authentication

- Static bearer tokens configured in `[auth] tokens` in `hoosh.toml`
- Tokens are SHA-256 hashed at startup and stored as digests — raw tokens never kept in memory
- Per-request comparison: hash provided token once, compare against stored digests with constant-time XOR (no length leak)
- Empty token list = auth disabled (with startup warning)

### Per-Provider Rate Limiting

- Sliding window RPM per provider (not per client)
- Configured via `rate_limit_rpm` on each provider section
- Checked after route selection (inside the handler, not as Tower middleware) because the provider identity is only known post-routing

### TLS Security

- Certificate pinning: `tls_pinned_certs` disables built-in root certs and adds only specified PEM certificates
- mTLS: `client_cert` + `client_key` for local provider mutual authentication
- Loud failure: if all pinned certs fail to load, logs `error!` (not silent degradation)
- Shared `build_provider_client()` utility across all 12+ providers

### Secret Management

- Config `Debug` impls redact `api_key`, `signing_key`, and auth `tokens`
- API keys support `$ENV_VAR` resolution (not hardcoded in config files)
- Inference error messages sanitized to clients — internal details logged server-side only
- Health endpoint returns generic "error" / "health check failed" — no internal hostnames/IPs

## Consequences

- Auth is disabled by default for development convenience. Production deployments MUST configure tokens.
- No role-based access control yet — all tokens grant equal access including admin endpoints. Tracked for future work.
- Rate limiting is per-provider, not per-client. A single client can consume the entire rate budget.
