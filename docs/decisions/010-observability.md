# ADR-010: Observability — latency histograms, event bus, traceparent

**Status:** Accepted
**Date:** 2026-06-10 (2.3.4)

## Context

The 2.3.x observability item (roadmap) had three parts: per-provider latency
histograms, a provider event bus (rust-old `events.rs`, majra pubsub), and
OpenTelemetry (rust-old `telemetry.rs`). Full OTel is an OTLP gRPC/protobuf
exporter with no Cyrius SDK — too heavy for this release.

## Decisions

### 1. Per-provider latency histograms (concrete, self-contained)

`hoosh_provider_latency_ms` — a Prometheus histogram per provider: 11 finite
`le` buckets (5..10000 ms) + sum + count, stored on the heap
(`metrics_latency_init` at startup), recorded around each `provider_forward`.
The counters are bumped with the same atomic CAS adds as the token counters,
because batch workers record concurrently. Emitted in `/metrics` only for
providers with traffic.

### 2. Event bus = majra pubsub + a recent-events ring

Faithful to `events.rs`: a majra `TypedPubSub` carrying the four `ProviderEvent`
kinds (HealthChanged / InferenceCompleted / InferenceFailed / RateLimited) as
JSON, published from the forward path, rate-limit check, and health poll.

**The impedance mismatch:** majra's `pubsub_subscribe` returns a bounded channel
(64). hoosh's accept loop is synchronous — a subscriber that is never drained
fills the channel and the next `pubsub_publish` blocks, hanging the gateway. So
hoosh does **not** keep an internal subscriber. majra is the pub/sub substrate
(`pubsub_total_published` → `hoosh_events_published_total`, ready for external
consumers, matching rust-old whose bus also had no in-process subscriber), and
hoosh's own observability is a **bounded recent-events ring** (128) exposed at
`GET /v1/events/recent` — events append to the ring (under a small mutex; batch
workers emit concurrently) as they publish. No channel, no fill, no hang.

majra is vendored at `src/vendor/majra.cyr` (committed, not `[deps]` — same
transitive-dep rationale as bote-core). Its `ratelimit_new`/`ratelimit_check`
collide by name with hoosh's (different arity); harmless — dead code in majra's
core pubsub, so hoosh's definitions win and work (one benign build note).

### 3. OTel = `traceparent` propagation now, OTLP export later (2.3.5)

The valuable, lightweight slice of OTel for a gateway is **W3C Trace Context
propagation**: forward the incoming `traceparent` to the backend so the
gateway→backend hop joins the caller's trace, generating one when absent.

The traceparent lives in the request headers, but the forward path is deep
(`handle_chat` → prep/finish → `provider_forward` → header builders) and runs on
batch worker threads. Rather than thread the header through every signature, it
is carried in a **thread-local** slot: `http_route` sets it (incoming or none),
the outgoing header builders read it (generating a fresh id from `clock_now_ns`
+ an atomic counter if absent). Each batch worker is a fresh thread with a zeroed
TLS slot, so it generates a per-item traceparent automatically.

**Critical gotcha:** `thread_local_init()` is **not idempotent** — it allocates
and installs a fresh *zeroed* TLS block (arch_prctl ARCH_SET_FS / TPIDR_EL0),
wiping every slot, including sigil's crypto bank at slot 0. It must be called
exactly once per thread: the main/accept thread via `crypto_tls_main_init()` at
startup, worker threads via `CLONE_SETTLS` in `thread_create`. `trace_*` only
get/set, never init. (The first build called init per-access and silently lost
every traceparent.)

## Consequences

- Latency, event, and trace visibility with no heavy dependency.
- Full OTLP span export deferred to **2.3.5**.
- `traceparent` ids are unique (clock + counter), not crypto-random — uniqueness
  is what correlation needs; a CSPRNG id is a later refinement.
- The event ring is lossy beyond 128 entries (recent-only), and `health_changed`
  fires on a health *poll* flip (no background health checker yet).

## Alternatives considered

- **Lightweight in-process bus (no majra)** — simpler, but the maintainer chose
  the majra substrate to match rust-old and keep the door open for real consumers.
- **Full OTLP export now** — gRPC/protobuf stack with no Cyrius SDK; its own
  multi-release effort (2.3.5).
- **Thread the traceparent through every signature** — invasive across ~6
  functions and the batch worker; the thread-local carrier is far smaller.

## Update (2.3.5): OTLP span export

The deferred OTel half. One span per inference exported to a collector over
**OTLP/HTTP + JSON** (`src/lib/otlp.cyr`).

**JSON, not protobuf.** Full OTLP/gRPC+protobuf needs a protobuf wire encoder
with no Cyrius lib — a large, error-prone effort. OTLP/HTTP also accepts JSON
(`Content-Type: application/json` on the collector's `/v1/traces`), buildable
with `str_builder`. A Cyrius stdlib protobuf lib is proposed upstream
(`cyrius/docs/development/proposals/2026-06-10-protobuf-lib.md`); an
OTLP/protobuf content-type is a follow-up once it lands.

**Non-blocking background exporter.** A synchronous per-request POST would add
collector-RTT to every request. Instead each inference enqueues a span fragment
into a bounded ring (under a mutex — batch workers enqueue concurrently); a
background thread wakes every `OTLP_BATCH_MS`, drains the ring, wraps the
fragments in one `resourceSpans` document, and POSTs it. The exporter does no
crypto (plain HTTP), so it needs no sigil bank; the POST uses a small response
buffer to limit the bump-allocator's per-cycle leak.

**Correlation + timing.** The span's traceId + spanId come from the request's
traceparent (2.3.4), so the gateway span joins the caller's trace. OTLP
timestamps are wall-clock epoch ns — `clock_now_ns` is `CLOCK_MONOTONIC`, so a
`CLOCK_REALTIME` `clock_gettime` is used; `start = end − latency` from the
monotonic-measured duration (one realtime read per span).

**Scope / deferred:** localhost `http://` collector only (the common sidecar) —
remote + `https://` (via sandhi) and OTLP/protobuf are follow-ups; spans are
inference-only (no nested provider/cache spans yet).
