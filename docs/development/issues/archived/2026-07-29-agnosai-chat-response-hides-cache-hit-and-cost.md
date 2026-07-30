# `/v1/chat/completions` hides two facts hoosh already knows: whether it was a cache hit, and what it cost — RESOLVED

**Status:** ✅ **RESOLVED in 2.5.12** (2026-07-30). Both halves shipped: `X-Hoosh-Cache:
HIT|SEMANTIC|MISS` as a response header, and `cost_micro_usd` + `provider` inside the existing
`usage` object. `cost_record` additionally now *returns* the figure it accumulated, so the value in
the response and the value in `/v1/costs` are the same value by construction rather than two
computations that ought to agree — which is stronger than what this issue asked for. See
CHANGELOG [2.5.12]. Originally filed 2026-07-29 against hoosh 2.5.11.

**Not addressed, and deliberately so:** a cached response still replays the original call's
`cost_micro_usd` in its body. That is correct — it is what the original call cost — and it is why
the signal is a *header*: a consumer bills on `X-Hoosh-Cache: MISS`, not on the presence of a cost
field. The `usage` block of a HIT describes the inference that was cached, not the request that
just hit it.

**Original report:** filed 2026-07-29 against hoosh 2.5.11. Verified by reading
`src/lib/handlers.cyr` (`_chat_prep` :2074-2103, `_chat_assemble` :2282-2295, `_chat_completion_body`
:1896-1922) and `src/lib/pricing.cyr` (`cost_record` :188, `estimate_cost_micro`). Both facts are
computed inside the request and then not surfaced.
**Placement:** unpinned — backlog. Two independent one-field additions; either is useful alone.
**Discovered:** 2026-07-29 while porting agnosai's `orchestrator/crew_runner` to Cyrius, at the point
where the Rust original's in-process `ResponseCache` and `CostTracker` (which were
`pub use hoosh::…` re-exports) had to become HTTP-seam behaviour.
**Severity:** Medium — no crash and no wrong answer from hoosh, but it makes a consumer's cost
accounting provably wrong and forces it to re-implement pricing hoosh already owns.
**Affects:** hoosh 2.5.11 and every earlier version with the server-side response cache.

## Summary

A client POSTing `/v1/chat/completions` gets back:

```json
{"id":"chatcmpl-hoosh","object":"chat.completion","model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M}}
```

That is `_chat_completion_body` (`src/lib/handlers.cyr:1896-1922`) in full. Two things hoosh knew
while building it are absent:

1. **Whether the response came from the cache.** `_chat_prep` (`:2081-2085`) returns a cached body
   verbatim as a terminal 200. There is no `X-Cache` header, no body field, and `"id"` is the
   constant `"chatcmpl-hoosh"` on both paths, so **a cache hit is byte-indistinguishable from a
   fresh inference.**
2. **What the call cost.** `_chat_assemble` (`:2282`) calls
   `cost_record(route_provider(route), route_base_url(route), resp_ptok, resp_ctok, model_cstr)`,
   which computes `estimate_cost_micro(...)` and folds it into a global counter — and then the
   response carries only `usage`.

Neither is recoverable client-side. `/v1/costs` is a cumulative global counter with no
request-identity dimension, so differencing it around a call is racy the moment two requests overlap
— which for agnosai is by construction, since its parallel and DAG process modes issue concurrent
task inferences.

## Why it matters to a consumer

agnosai's Rust original ran the cache and the cost tracker **in-process**, as `hoosh::cache` and
`hoosh::cost` re-exports (`agnosai/rust-old/src/llm/mod.rs:14-20`). The Cyrius port reaches hoosh
over HTTP instead, and both capabilities correctly move server-side — hoosh does the work already,
automatically, on every call. That part is a clean simplification and the port deletes a lot of code
because of it.

The gap is only that the *result* is invisible. Three concrete consequences:

**A cached response gets costed twice.** The Rust original returned early on a cache hit, before the
cost-record call, so a hit cost nothing. Over the seam agnosai sees an ordinary 200 carrying the
original call's `usage` block and has no way to know it was a hit, so it prices it again. hoosh's own
`/v1/costs` correctly skips hits. **The two views diverge by exactly the server-cached traffic**, and
neither side is wrong — the information to reconcile them simply is not on the wire.

**The consumer must re-implement pricing.** To put a per-task cost in its own result metadata,
agnosai has to port hoosh's `pricing.cyr` — the 16-row table, the per-provider fallbacks, the
`provider_is_local` zero-cost short-circuit, and the exact truncating expression
`in_tok * in_scaled / 1000 + out_tok * out_scaled / 1000` with each term truncated separately —
purely so the numbers reconcile against `/v1/costs`. That is a copy of hoosh's pricing table living
in a consumer, guaranteed to drift the first time hoosh updates a price.

**Provider attribution is a guess.** Lacking any provider field on the response, agnosai infers it
from the model name by prefix (`gpt-` → OpenAI, `claude` → Anthropic, …). hoosh bills against
`route_provider(route)` — the actual serving route, after `router_select_hw` and after the DLP
confidential re-route to a local provider (`handlers.cyr:2058-2071`). A DLP re-route means hoosh
records $0 while the consumer prices it as remote, and nothing on the wire can correct that.

## Proposed fix

Two independent additions to `_chat_completion_body`. Either alone is worth having.

**(a) A cache-hit signal.** Cheapest correct form is a response header set on the terminal-200 path
in `_chat_prep`:

```
X-Hoosh-Cache: HIT        # exact-key hit
X-Hoosh-Cache: SEMANTIC   # semantic-similarity hit
X-Hoosh-Cache: MISS       # forwarded to a provider
```

A header keeps the body OpenAI-compatible, which matters — consumers parse this with
OpenAI-shaped clients. A body field (`"cached":true`) would work too but is a compatibility
question; the header is not.

**(b) Per-response cost.** hoosh has already computed it by the time the body is built:

```json
"usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M,
         "cost_micro_usd":C,"provider":"ollama"}
```

`cost_micro_usd` as an integer, matching `estimate_cost_micro`'s own unit exactly, so a consumer
never converts and never rounds. `provider` is the real `route_provider(route)`, which also closes
the DLP-re-route attribution gap. Both are additive inside the existing `usage` object; an
OpenAI-shaped client ignores unknown fields.

Together they let a consumer delete its pricing copy entirely, stop double-costing cached responses,
and report a per-task cost that reconciles with `/v1/costs` exactly.

## Consumer-side workaround

agnosai is proceeding without either, deliberately:

- No client-side cache at all — hoosh's is strictly better, and the original's client-side cache had
  a real key-collision bug that the seam removes. Nothing to work around here.
- A local `src/llm_pricing.cyr` porting hoosh's table and its truncating arithmetic verbatim. This is
  the code that (b) would delete.
- The double-costing on cache hits and the inferred-provider attribution are **documented as known
  divergences** in the port's module header rather than papered over, with `/v1/costs` named as the
  billing truth.

Filed rather than patched: hoosh is a separate project, and this is a surface decision that is
hoosh's to make.
