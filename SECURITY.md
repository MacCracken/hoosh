# Security Policy

## Scope

`hoosh` is an AI inference gateway that proxies requests to LLM providers,
manages token budgets, and optionally scans content for PII (DLP feature).
Security-sensitive areas include:

- **API key handling**: provider API keys come from `[[providers]] api_key`, which
  takes `$NAME` so a key can live in the environment, and go only to their own
  provider.
- **Authentication**: bearer tokens from `[auth] tokens`, compared in constant time.
- **Audit chain**: HMAC-SHA256 linked chain for tamper-evident request logging.
- **DLP scanning**: hand-written detectors for PII and secrets, plus the
  operator's literal `[[dlp_pattern]]` strings, with classification-driven
  routing (restricted content refused, confidential content kept on local
  providers).
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

- **API keys**: a key is sent to its provider in that provider's auth header
  (`x-api-key` for Anthropic, `Authorization: Bearer` for the rest) or, for
  Gemini, in the `key` query parameter Google's API takes. Keys are not written
  to the log or into any response. A route whose `base_url` is plain `http://`
  sends its key unencrypted; give keys only to `https://` routes or to local
  backends you trust.
- **Authentication**: each configured token is compared with the presented one
  in constant time (stdlib `ct_eq_bytes_lens`), so a wrong token costs the same
  whether it differs early or late; the bench's `auth_verify_wrong_*` rows watch
  this. (Through 2.9.23 this line said tokens were pre-hashed and compared as
  SHA-256 digests; the port has never done that.)
- **Token contents** (2.9.23): `[auth] tokens` is read as TOML, so a token may
  hold any characters. Through 2.9.22 a `,`, `]` or `"` inside a token split it,
  and each piece authenticated on its own; rotate any token holding those.
- **DLP patterns**: the built-in detectors (SSN, card numbers, API keys, AWS and
  GitHub tokens, phone numbers, IPv4 addresses, email) are byte scanners, and
  `[[dlp_pattern]]` entries are literal, case-insensitive substrings, not
  regular expressions. hoosh has no regex engine, so a pattern cannot be
  written to backtrack catastrophically (ReDoS). DLP is off unless `[dlp]`
  enables it.
- **Request parsing**: request bodies are read by hand-written byte scanners
  (`jsonlite.cyr`, `reqparse.cyr`), not a general JSON parser. Four fuzz
  targets (`fuzz/*.fcyr`) run in CI, and a body is capped at 1 MiB (a larger
  `Content-Length` is answered 413 before anything is allocated for it).
- **Audit chain**: the key comes from `[audit] signing_key`, which takes
  `$NAME`; keep it secret. Without one, a random 32-byte key is generated per
  process and not persisted, which detects tampering within a process's
  lifetime; a chain persisted across restarts with `[[storage]]` needs a
  configured key.
- (Through 2.9.25 this file still described rust-old in places: config `Debug`
  impls, serde `Serialize`/`Deserialize` types, regex DLP patterns compiled at
  startup, and a SHA-256 token comparison. The port has none of those; 2.9.24
  and 2.9.26 corrected them.)
