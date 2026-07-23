# rust-old → Cyrius Port Parity Review

**Date**: 2026-07-22 · **Reviewed**: `rust-old/` (22,956 LOC Rust, v1.3.0) vs `src/` (13,585 LOC Cyrius, v2.5.0)

Method: 24 agents mapped every behavior in the Rust tree and searched for a Cyrius
counterpart — 1,007 behaviors catalogued, 602 flagged as `partial`/`missing`.
The high-impact subset below was then **hand-verified against source**; items in
[Unverified](#unverified-agent-reported) are agent-reported but not personally confirmed.

The roadmap's claim that the port "reached parity with the `rust-old` reference and then
extended it" is **broadly true for surface area** — the Cyrius route table is a strict superset
minus the deliberately-deferred audio pair, and it adds vLLM/TensorRT/ONNX providers, batch
inference, MCP, OTLP, and hardware planning that Rust never had. It is **not true for
request-path fidelity**: several per-request knobs and the entire health/failover feedback
loop did not survive the port.

---

## Verified findings

### 1. `bind` is ignored — the gateway listens on `0.0.0.0` — **security**

`hoosh.cyml:5` ships `bind = "127.0.0.1"`, but [config.cyr](../../src/lib/config.cyr) never
reads the key. [main.cyr:294](../../src/main.cyr:294) binds `INADDR_ANY()` and
[main.cyr:310](../../src/main.cyr:310) prints `listening on 0.0.0.0:`.

Rust bound `self.server.bind`, defaulting to `127.0.0.1` (`config.rs:496`).

Combined with [auth.cyr:5-6](../../src/lib/auth.cyr:5) — *"If no auth tokens configured, allow
all"* — a default-config hoosh is an **unauthenticated LLM gateway reachable from the whole
network**, while its own config file claims loopback. This is the most serious item in the review.

### 2. `temperature` and `top_p` are silently dropped; `max_tokens` is hardcoded

`temperature` and `top_p` appear **nowhere** in `src/lib/` or `src/main.cyr`. `max_tokens` is
read once ([handlers.cyr:1778](../../src/lib/handlers.cyr:1778)) purely to size the *token-budget
reservation* — it is never forwarded to a provider. Anthropic bodies hardcode `4096`
([provider.cyr:558](../../src/lib/provider.cyr:558)) or `16384` when thinking is on
([provider.cyr:551](../../src/lib/provider.cyr:551)).

Rust forwarded all three and range-validated them (`server/handlers.rs:245-267`,
`provider/openai_compat.rs:71-84`, `client.rs:49-57`).

Impact: an OpenAI-compatible client asking for `temperature: 0` gets nondeterministic output
with no error. This is the loudest correctness gap for API consumers.

### 3. No health probing, no circuit breaker, no failover

Rust ran a background health checker (`health.rs:123-269`) with a
`UNHEALTHY_THRESHOLD = 3` consecutive-failure state machine, and the router **filtered
unhealthy providers out of the candidate set** (`router.rs:76-96`).

In the port, `health_check`, `consecutive_fail`, `unhealthy`, and `circuit` match nothing in
`src/lib/*.cyr`. `_health_map` is written and read **only** inside `handle_health_providers`
([handlers.cyr:558-567](../../src/lib/handlers.cyr:558)) — it is observational telemetry for the
endpoint. [router.cyr](../../src/lib/router.cyr) never consults it.

Consequence: a dead provider stays in rotation and keeps taking traffic until an operator
polls `/v1/health/providers` and intervenes. Retry ([retry.cyr](../../src/lib/retry.cyr)) is the
only resilience left, and see #4.

### 4. Retry has no retryability gate

[retry.cyr](../../src/lib/retry.cyr) contains no `retryable`/`is_retry`/`status` logic. Rust
gated retries on `HooshError::is_retryable()` (`provider/retry.rs:64-78`, `error.rs:70-84`), so
a 400/401/404 failed fast. The port re-attempts permanent errors, burning the full backoff
schedule on requests that can never succeed.

### 5. `[cache] ttl_secs` is shipped but inert — stale responses served forever

`hoosh.cyml:33` sets `ttl_secs = 300`. The string `ttl` appears **zero times** in
[cache.cyr](../../src/lib/cache.cyr) and [config.cyr](../../src/lib/config.cyr).

The cache entry layout is `{0: value, 8: last_access_ms}`; that timestamp is an LRU recency key
(rewritten on every `cache_get`), never a deadline. Rust had `CacheEntry.ttl`, `is_expired()`,
an expired-on-get removal path that bumped both `evictions` and `misses`, and an
`evict_expired()` sweep (`cache/mod.rs:40-44,123-129,207-214`).

Eviction is capacity-only, so a cached response is returned as a hit indefinitely.

### 6. Groq's default base URL is wrong

[types.cyr:150](../../src/lib/types.cyr:150) returns `https://api.groq.com`.
`_provider_url` ([provider.cyr:749](../../src/lib/provider.cyr:749)) is a plain concatenation,
producing `https://api.groq.com/v1/chat/completions`.

Groq's OpenAI-compatible endpoint lives under `/openai` — Rust defaulted to
`https://api.groq.com/openai` (`provider/groq.rs:21`). **The default Groq route 404s.**

### 7. No request body size limit

No `1048576`, `body_limit`, `MAX_BODY`, `413`, or `content_length` cap in
[http_server.cyr](../../src/lib/http_server.cyr) or [handlers.cyr](../../src/lib/handlers.cyr).
Rust applied `DefaultBodyLimit::max(1 MiB)` (`server/mod.rs:335`). Unbounded request bodies on
a never-freeing bump allocator is a memory-exhaustion path.

### 8. `[auth] tokens` accepts only one token

[config.cyr:264-266](../../src/lib/config.cyr:264) does a single `toml_get(auth_p, "tokens")`
followed by one `vec_push`. The consumer ([auth.cyr:24](../../src/lib/auth.cyr:24)) loops a vec
correctly, so this is purely a parse-side limitation. Rust took `tokens = ["a","b"]`
(`config.rs:166-171`). Multi-token setups (per-consumer keys, key rotation) can't be expressed.

### 9. `POST /v1/hardware/simulate` was not ported

`handlers.rs:1104-1183` + `hardware.rs:389-415` — what-if capacity planning: add/remove/replace
devices, re-run sharding, return `{original, simulated}` snapshots, with validation
(`model_params > 0`, ≤64 devices, non-zero `memory_bytes`). No counterpart, and it is **not** in
the roadmap's deferred list.

Related hardware regressions: `available_vram(reserved)`, `fits_model(...)` against *available*
rather than *total* VRAM, and `Router::select_with_hardware` (deprioritize local providers when
a model won't fit) — `hardware.rs:171-296`, `router.rs:218-295`.

Note also `POST /v1/hardware/format` → `POST /v1/hardware/model-format`: deliberately
redesigned (path arg → raw bytes, which is safer), but it is an unannounced **breaking URL
change** for any consumer of the Rust API.

### 10. CLI surface narrowed

[main.cyr:623](../../src/main.cyr:623) takes a bare positional port. Missing versus Rust
(`main.rs:26-86`): `serve --bind`, `serve -c/--config`, `models/infer/health --server <url>`,
`infer --stream`. Also absent: `RUST_LOG`-equivalent log-level env var, graceful
shutdown on SIGINT, and SIGHUP config reload (`server/mod.rs:390-401`).

`transcribe` / `speak` are correctly gone (deferred to svara). The port adds `bench`, `help`,
and `version`.

### 11. `/v1/costs` reports aggregates, not per-provider cost

[handlers.cyr:383-403](../../src/lib/handlers.cyr:383) emits totals plus a route listing. Rust
returned `{records: [ProviderCostRecord], total_cost_usd}`, where each record carried
`total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `request_count`, keyed by
`{provider}:{base_url}` (`cost/mod.rs:116-127`). Per-token pricing survives
([pricing.cyr](../../src/lib/pricing.cyr)) but nothing **accumulates** it per provider, so
`/v1/costs` can't answer "what has Anthropic cost me today".

### 12. CI dropped two gates

`.github/workflows/ci.yml` runs vet/fmt/lint/build/test/security-scan/fuzz/bench. It does **not**
run `cyrius deny` — even though CLAUDE.md lists it in the cleanliness check — and there is no
coverage equivalent for rust-old's `codecov.yml` (project 85% / patch 80%).

---

## Unverified (agent-reported)

Reported by the mapping pass but **not** hand-checked. Treat as leads.

**Request validation** (`handlers.rs` `validate_chat_request`): empty `messages` → 400;
`messages.len() > 256` → 400; temperature/top_p range errors; `max_tokens` clamped to a
per-provider `max_tokens_limit`; per-route HTTP method enforcement → 405.

**Budget**: unknown `pool` name → `400 "Token pool 'X' does not exist"` (port reportedly falls
through silently); budget pool state surviving `/v1/admin/reload`; the 5-tier priority queue
(Critical→Background) behind `/v1/queue/status`.

**Audit**: `[audit]` config section (`enabled`, `signing_key` with `$ENV` expansion, random
32-byte fallback, `max_entries`); chain-link verification
(`entry.previous_hash == prior.hash`, `audit.rs:186-195`); `admin.costs_reset` and
`inference.error` audit entries.

**Config keys with no reader**: `[context]` (`compaction_threshold` 0.8, `keep_last_messages` 10,
`enabled`), `[hardware]` (`cache_ttl_secs`, `disabled_backends`, `vram_reserve_bytes`,
`refresh_interval_secs`), `[server] health_check_interval_secs`, `[retry] jitter_factor`,
`[[providers]] enabled = false` / `max_tokens_limit`.

**Model catalog**: Rust shipped 66 default entries with a `>= 60` test invariant; the Mistral
family (`mistral-large`, `mistral-small`, `codestral`, `pixtral-large`, `mistral-nemo`) and the
local Ollama/llama.cpp defaults (`llama3*`, etc.) are reported absent
(`provider/metadata.rs:523,579`).

**Ollama**: `options.temperature` / `options.num_predict` passthrough; embeddings normalized to
the OpenAI `{object:"list", data:[...]}` envelope (`ollama.rs:125-133,314-340`).

**Timeouts**: `connect_timeout(10s)` and the 300s total provider timeout
(`client.rs:127`, `provider/mod.rs:26-27`).

**DLP**: `custom_patterns` config, and `PatternMatch` records (name/level/offset/length) —
the port is presence-only, returning just the highest classification level. The 8 built-in
patterns themselves **are** all present.

**Other**: SSE keep-alive pings every 15s; `created` unix timestamp in the chat.completion
envelope; hardware `system_io()` / `has_fast_interconnect()` / `gpu_telemetry()` /
`runtime_environment()` accessors; both fuzz targets (`fuzz_inference_request`,
`fuzz_message_content`) have no `.fcyr` counterpart; `bench` suites `e2e.rs` and
`live_providers.rs` unported.

---

## Confirmed clean

- **Routes**: Cyrius is a strict superset of Rust's table apart from `/v1/audio/*` (deferred)
  and `/v1/hardware/simulate` (#9). Adds `/v1/batch*`, `/v1/events/recent`, `/v1/tools/*`,
  `/v1/cost/estimate|recommend`, `/v1/models/catalog|pull|delete`, `/v1/training/*`,
  `/v1/hardware/{cost,compatible-models,requirement-match,training-estimate}`, `/api/*`.
- **Providers**: 17 kinds ([types.cyr:51-70](../../src/lib/types.cyr:51)) vs Rust's 14 — adds
  vLLM, TensorRT, ONNX. CLAUDE.md's "17 LLM providers" is accurate.
- **Routing strategies**: all four (priority, round-robin, lowest-latency, direct) present and
  config-selectable ([config.cyr:7-11](../../src/lib/config.cyr:7)).
- **DLP patterns**: all 8 (email, ipv4, phone_us, ssn, credit_card, api_key, aws_key,
  github_token) hand-ported as byte matchers with correct classification levels.
- **Audio**: cleanly deferred — no half-ported remnants.

---

## Suggested order

1. **`bind`** (#1) — security, ~10 lines.
2. **temperature / top_p / max_tokens** (#2) — the visible API-conformance break.
3. **Groq base URL** (#6) — one string.
4. **Body size limit** (#7) and **retryability gate** (#4) — small, bound real risk.
5. **Cache TTL** (#5) — either implement it or drop `ttl_secs` from `hoosh.cyml`; shipping an
   inert key is worse than shipping neither.
6. **Health/failover loop** (#3) — the largest piece of genuinely missing engineering.
7. Then triage the Unverified list, and update the roadmap's parity claim to name what is
   knowingly not ported.
