# Security Policy

## Scope

`hoosh` is an AI inference gateway that proxies requests to LLM providers,
manages token budgets, and optionally scans content for PII (DLP feature).
Security-sensitive areas include:

- **API key handling**: Provider API keys are resolved from environment variables
  and passed to HTTP backends. Keys are redacted in config Debug output.
- **Authentication**: Bearer token auth with constant-time SHA-256 comparison.
- **Audit chain**: HMAC-SHA256 linked chain for tamper-proof request logging.
- **DLP scanning**: Regex-based PII detection with classification-driven routing.
- **Content proxying**: User-supplied prompts are forwarded to LLM providers.
  Hoosh does not execute arbitrary code from prompts.

## Supported versions

Only the latest released version receives security fixes.

| Version | Supported |
|---|---|
| Latest | Yes |
| Older | No |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/MacCracken/hoosh/security/advisories/new)
or by emailing the maintainer directly.

Include:

- A description of the vulnerability.
- Steps to reproduce or a proof of concept.
- The potential impact.

You should receive an acknowledgement within 72 hours. We aim to release a fix
within 14 days of confirmation.

## Security considerations

- **API keys**: Keys are stored in memory and passed via HTTPS to providers.
  Never log or expose keys. Config Debug impls redact all sensitive fields.
- **Authentication**: Token comparison uses constant-time SHA-256 digest
  comparison to prevent timing attacks. Tokens are pre-hashed at startup.
- **DLP patterns**: Regex patterns are compiled at startup. Custom patterns
  from config are validated for syntax but not for ReDoS resistance — use
  bounded quantifiers in custom patterns.
- **Serialization**: `InferenceRequest`, `MessageContent`, and related types
  derive `Serialize` and `Deserialize`. If you deserialize untrusted input,
  apply your own validation layer.
- **Audit chain**: The signing key should be kept secret. If auto-generated,
  it is random per process and not persisted.
