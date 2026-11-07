# ADR-012: The model catalog is asked of the providers

**Status:** Accepted
**Date:** 2026-10-09 (2.8.0)

## Context

`GET /v1/models/catalog` is what a client's model picker reads (thoth's Ctrl-P picker and `/models <provider>`
are the consumers). Through 2.7.1 it was the compiled table in `metadata.cyr`, ported from rust-old in mid-2025,
filtered by which enabled route's patterns matched each row. On 2026-10-09 a gateway with an Anthropic route
offered six Claude ids — `claude-opus-4`, `claude-sonnet-4`, `claude-3.5-haiku`, `claude-3-5-haiku`,
`claude-3-5-sonnet`, `claude-3-haiku` — and the live `GET /v1/models` showed fourteen others. Not one of the six was
servable: two are family prefixes (the alias is `claude-opus-4-0`), one never existed, three are retired. A pick from
the catalog was a 404 every time, and no model released after the table was written could appear without a source
edit and a new hoosh.

The same table fed context windows to compaction (every Claude it knew was 200K; the current ones are 1M), and its
neighbour in `pricing.cyr` billed every Opus 4.x through one `claude-opus-4` row at $15/$75 — three times what
Opus 4.5–4.8 cost — while Claude 5.x models fell to a $3/$15 provider default (thirty times Haiku 5.5's rate, a
third of Fable 5.1's).

And the request builder could not know what each model accepts. It sent `{"type":"adaptive"}` + an effort level to
every Claude model when the client asked for reasoning: Claude Haiku 4.5 answers that with a 400 ("adaptive
thinking is not supported on this model"), Claude Sonnet 4.6 400s on effort `xhigh`, and Claude Opus 4.7 and later
return thinking blocks with EMPTY text unless the request says `display: "summarized"`.

## Decision

**The catalog is asked of the providers, with each route's own credentials, and kept until the next refresh.**
Anthropic's Models API answers per model with `max_input_tokens`, `max_tokens` and a capability tree (which
thinking modes and which effort levels the model takes); Gemini's `models.list` with token limits and generation
methods; an OpenAI-compatible `/v1/models` with ids (and, from OpenRouter and Mistral, context lengths, prices and
vision); Ollama's `/api/tags` with what the box has.

- **Where it runs.** Remote lists are HTTPS, and a TLS request needs a sigil crypto bank, which only the pool
  workers own (ADR 011). A refresh is therefore a POOL JOB (`JOB_CATALOG_REFRESH`), enqueued at startup (the
  gateway never waits on a provider to start serving), after `/v1/admin/reload`, and by a bankless timer every
  `[catalog] refresh_secs` (six hours by default); `POST /v1/models/refresh` runs one inline on the worker serving
  it. Single-flight: a refresh asked for while one runs is skipped. (2.9.1: any thread may do TLS now, see
  ADR 011's 2.9.1 update. The refresh stays a pool job.)
- **Publication.** A refresh builds a fresh vec of per-route records and publishes it with one aligned pointer
  store — the same lock-free old-or-new reasoning as the `_router` swap (ADR 011): readers on the workers load it
  once, and the allocator never frees, so an old snapshot stays valid. A route whose fetch fails keeps its previous
  good list; a provider blip does not empty a picker.
- **Memory.** hoosh's heap never frees, and a model list is parsed into a full JSON tree before the few fields kept
  are copied out. The fetch and the parse run in one growable arena allocator, reset after every route; it settles
  at the largest single answer. Only the kept entries reach the heap.
- **What is listed.** A live route is offered for what its provider listed, filtered by the route's own patterns
  (the operator's routing rules still decide what the gateway serves). A static route — `catalog = "static"`, the
  default for OpenRouter (a marketplace of several hundred models) and Whisper — or a live one that has not answered
  yet is offered for the compiled table's rows, which now carry a `listed` bit so a family prefix is never offered
  as an id. Each entry gains `source`, `routes` (every enabled route matching the id, in router order — what
  `router_select` chooses among) and whatever is known: `display_name`, `context_window`, `max_output_tokens`,
  `capabilities` and `pricing` (micro-USD per 1K tokens, written only when known).
- **The operator has the last word.** `[[models]]` blocks in `hoosh.cyml` set a model's price, context window,
  tier, vision and (for OpenAI-compatible routes) whether it takes `reasoning_effort`, over anything a provider or
  the table says — the way to price a model released after this hoosh, or a fine-tune. A block that sets only some
  fields is laid over the provider's entry, never in place of it.
- **The table stays, as the fallback.** It still supplies tiers (no provider publishes one) and prices (few
  providers publish one), and it was refreshed to the current Claude lineup from the Models API itself.

The request builder reads the model's live entry on the serving route (`catalog_thinking_plan`): adaptive +
effort for models that take it, a thinking budget for those that take only that, the effort level clamped to the
ones the model accepts, and `display: "summarized"` whenever thinking is on. A model the catalog has not listed gets
what the client asked for, exactly as before.

**2.8.1:** the entry's output ceiling also bounds the request. `max_tokens` is trimmed to the operator's
`[[models]]` `max_output_tokens`, else the provider's listed ceiling (`catalog_out_cap`), because a provider refuses a
request over it with a 400 rather than trimming it. The thinking budget is planned against the trimmed figure.

**2.10.2:** an entry says `tier`, `kind` and `capabilities.tools`, because a client choosing a model needs them and was
reading them from prices and a list of words. See the addendum below.

## Consequences

- A new model appears in every client's picker at the next refresh, priced and shaped correctly, with no hoosh
  release — unless its provider publishes no price, in which case it is listed unpriced until the operator adds a
  `[[models]]` block or a hoosh release adds a row.
- Each refresh is one unbilled, authenticated GET per live route (plus pages). That also makes the catalog the
  first place a revoked or mistyped key shows: `/v1/health/providers` now carries each route's `catalog_status`,
  where a 401 or 403 is visible before any completion fails on it — the remote health probe is a TCP connect and
  cannot see a key.
- An OpenAI-compatible list carries non-chat models; they are dropped by a documented substring filter
  (embeddings, speech in and out, images, moderation, realtime/audio, the retired completion engines). The filter
  is a heuristic, which is why it is narrow and why a route's patterns remain the operator's real control.
- The live list is only as trustworthy as the provider. Ids are vetted (printable ASCII, no quote, backslash or
  space, at most 255 bytes) before they are kept, and every string a catalog entry carries is JSON-escaped on the
  way out.

## Alternatives considered

- **Keep the compiled table and refresh it each release.** That is what 2.7.1 did, and it went stale between
  releases by construction — the incident above. A table stays, as the fallback; it is no longer the source.
- **Fetch on every `/v1/models/catalog` request.** One provider round-trip per picker open, on the request path,
  with the picker's short client deadline — and a slow provider would make the picker look broken. The snapshot
  answers in memory.
- **A separate banked catalog thread.** sigil now has 64 lanes, so a thread could hold one; but the pool already
  owns TLS for every other bankless background (the OTLP exporter enqueues to it), and one mechanism is easier to
  reason about than two.

## Addendum (2.10.2): tier, kind and tools

agnosai chooses a model per tier from the catalog, and could not say which model was which: it read a tier from prices, could
not keep an agent with tools off `gpt-6-astra`, and sent an embedder like `bge-m3` to chat.

**Every field is absent when nobody said.** A client must be able to tell "no" from "unknown". That rules out
`metadata_lookup` as the source: its defaults for a model with no row (STANDARD, tools on) suit the optimizer and would read as
facts on the wire. `metadata_lend` fills in only what the table holds.

**Whoever spoke first, wins.** An operator's block, then the provider, then the words, then the table's row.

| Field | `[[models]]` | The provider | The words | The table |
|---|---|---|---|---|
| `tier` | `tier` | nobody publishes one | | the row |
| `capabilities.tools` | `tools` | Ollama's `capabilities` | | the row |
| `capabilities.vision`, `context_window` | `vision`, `context` | as before, and Ollama's `capabilities` for vision | | the row, live entries too |
| `kind` | `kind` | Anthropic's list, Google's `generateContent`, Ollama's `details.family` and `capabilities` | a match | the row |

- **`kind` is one word**, `chat`, `embedding` or `other`, not an `endpoints` list. It answers what clients ask, and it can
  grow. `other` says "neither". A model that does both reads `chat`, since the failure to prevent is an embedder sent to chat.
- **The words never say chat.** A match means a model is not one, and a miss means nothing: `bge-m3` holds none of them. So
  an OpenAI-compatible model with no row has no `kind`, and that is the honest answer. The list stays filtered, and dropping is
  not how `bge-m3` was answered: a client that wants an embedder has to be able to find one.
- **A row lends only to a model of its own kind.** The lookup goes by longest prefix, so `gpt-4o` and `mistral` would otherwise
  describe `gpt-4o-transcribe` and `mistral-embed`.
- **Ollama costs one local `POST /api/show` per tag per refresh**, at most 256, because `/api/tags` says nothing about what a
  model does. A refused tag stays as it was, and a server that stops answering ends the round. `details.family` needs no call and
  answers for an embedder on an Ollama from before `capabilities`.
- **`gpt-6-sol` and `gpt-6-luna` read `tools: true`.** They take tools on Chat Completions only at reasoning effort `none`, and
  the column says what a request can do, as `meets_requirements` reads it. A client sending another effort with tools hears the
  provider's refusal. An operator who wants them off a picker sets `tools = false`.

Google's and OpenAI's embedders stay off the catalog's list. Listing them would hand a client that reads no `kind` a model it
cannot chat with.
