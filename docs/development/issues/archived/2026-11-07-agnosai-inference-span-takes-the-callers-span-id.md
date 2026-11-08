# An inference span takes the caller's span id, and the forwarded traceparent repeats it — RESOLVED

**Status:** ✅ **RESOLVED** in hoosh 2.10.3 ([CHANGELOG](../../../../CHANGELOG.md), *Fixed — trace context*). Both asks
shipped. A request with an inbound `traceparent` gets a span id hoosh mints, the exported span names the caller's id as
its `parentSpanId`, and every upstream request (buffered, streamed, local and remote) carries hoosh's own:
`00-<the caller's trace id>-<hoosh's span id>-<the caller's flags>`. With no inbound header the span is a root with no
`parentSpanId`, as before. `scripts/wire-smoke.py`'s `trace` run is this issue's repro against a stub provider and a
stub collector; 8 of its 10 checks fail on 2.10.2. Two more things this found, fixed in the same release:
- **A batch item took an unrelated request's context.** Since 2.4.0's pool a worker kept its last connection's trace,
  so batch items, catalog refreshes and OTLP exports went out under it. A batch item is now a span of its own under the
  request that submitted it (sync and async), and every other job begins as a root.
- **A minted id told the provider the gateway's uptime and request count.** A root's trace id was the monotonic clock
  and a counter in plain hex. Ids come from a keyed permutation of the counter now, with no clock read.
- Version `ff` and all-zero trace and parent ids, which W3C calls invalid, start a root instead of being continued.

**agnosai can add its outbound `traceparent`** (`agnosai_hoosh_chat_in`), and move this off *Waiting on another project*.

**Discovered:** 2026-11-07, by agnosai, wiring its `chat` span to hoosh's. **Severity:** Low. Nothing fails, but every
trace that crosses hoosh is drawn wrong, so a consumer cannot hand hoosh its context.
**Affects:** hoosh 2.3.4 (when the `traceparent` arrived) through 2.10.2: `src/lib/otlp.cyr`'s
`otlp_enqueue_inference_span`, and `src/lib/provider.cyr:938`.
**Placement:** 2.10.3.

A request that carries `traceparent: 00-<trace-id>-<parent-id>-01` is the caller saying "my span is `<parent-id>`, and
yours is its child". hoosh does two things with it:

1. **Its span reuses the caller's id.** `otlp_enqueue_inference_span` writes `spanId` from `tp + 36`, 16 hex digits,
   which is the inbound `<parent-id>`. It writes no `parentSpanId`. So the exported trace holds two spans with one id,
   the caller's and hoosh's, and hoosh's span is the child of nothing.
2. **The provider is told the same thing.** `provider.cyr:938` adds `trace_current_or_new()` to the upstream request
   unchanged, so a provider that traces names the *caller's* span as its parent and skips hoosh.

With no inbound header, `trace_generate` mints both ids and hoosh's span is a root, which is right.

Because of (1), agnosai sends hoosh no `traceparent` at all (`src/llm/hoosh.cyr`, `agnosai_hoosh_chat_in`): its
`chat` span and hoosh's `inference` span sit in two unrelated traces, and an operator cannot follow one request from a
crew to the provider.

## Repro

```sh
# hoosh with OTLP export on, and any provider route
curl -s localhost:8088/v1/chat/completions \
  -H 'traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' \
  -d '{"model":"<a listed model>","messages":[{"role":"user","content":"hi"}]}'
# The exported span reads "spanId":"00f067aa0ba902b7" and has no "parentSpanId".
```

## The ask

For a request with an inbound `traceparent`:

- **Mint a span id** for the inference span, and export the inbound `<parent-id>` as its `parentSpanId`.
- **Forward hoosh's own:** the upstream `traceparent` carries the same trace id and hoosh's new span id, so a provider's
  span is hoosh's child.

With no inbound header, keep today's behaviour: a root span with no `parentSpanId`.

## What agnosai does once this ships

`agnosai_hoosh_chat_in` adds an outbound `traceparent` from its `chat` span's context (ADR 023's explicit context, so
no thread-local is involved), and hoosh's span becomes that span's child. agnosai's roadmap lists this under
*Waiting on another project*.
