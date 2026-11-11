# ADR-013: What a request's memory belongs to

**Status:** Accepted (2.10.3), done for the gateway's request paths and bounded keepers in 2.11.2.
- 2.10.0 and 2.10.2 ship the fixes in their "Done" sections.
- 2.10.3 ships decision 1 for the chat path, with decision 2's boundary for the keepers it reaches.
- 2.11.2 ships the rest of decisions 1–3: every other path a chat reaches, and the recycling.
- What is left (the semantic index, the endpoints not yet measured) is on the roadmap.
**Date:** 2026-11-05

## Context

hoosh's allocator never frees. Everything `alloc()` hands out, and everything the stdlib allocates for it
(`str_builder`, `vec`, `map`, `bayan_json_parse`, `to_cstr`), lives until the process exits. The roadmap's 2.10.0
item said about 1 KiB per request still accumulated and called for a per-request arena. 2.10.0 measured before
designing, and the measurements changed the plan.

**How it was measured** (`scripts/memory-growth.py` repeats it). A real gateway with one Ollama route at a local
fake backend (`/api/chat`, a fixed reply), sent chat requests with unique prompts so that the response cache never answers. `VmRSS` came from
`/proc/<pid>/status` after 200 warm-up requests; a scratch build logged `alloc_used()` at each phase of a request,
sent one at a time.

**What 2.9.26 did, per request:**

| path | growth | where it went |
|---|---|---|
| buffered chat, local route | 139 KB | 128 KiB + 1 response buffer and 2 KiB header buffer per call, in `http_client.cyr` |
| streamed chat, local route | 79 KB | 64 KiB line buffer, 4 KiB read buffer and 4 KiB header buffer per stream |
| any insert into a full response cache | 524 KB and 4 ms | `cache_evict_expired` and `_cache_evict_lru` each built `map_keys()` of the whole map (a vector grown by doubling) and scanned it |

The cache fills at the default 10000 entries. With mostly new prompts at about 180 requests per second that took
about a minute, after which a 20000-request run reached 5.3 GB and slowed to a sixth of its throughput. The 4 ms
ran under `_chat_lock`, so it stalled every worker's chat.

## Done in 2.10.0

- **The cache evicts in O(1) and allocates nothing** beyond the new entry: each entry is on two lists, by access
  and by creation (`cache.cyr`). A full 10000-entry cache now takes 64 bytes and about 3 µs per insert.
- **The local client and local streams reuse per-thread scratch blocks** (`http_client.cyr`, thread-local slots 7
  and 8). A local response lives until the same thread's next local request; every caller reads it first.

After both: **5.6 KB per buffered request, 3.5–4.2 KB per stream**, steady across 10500 requests including the cache
filling and running full. That is still about 2 GB a day at 100 requests per second.

**What the 6.2 KB of a buffered request is** (`alloc_used()`, median of 200):

| phase | bytes | of which outlives the request |
|---|---|---|
| connection handling before the handler (trace context, auth, routing) | ~900 | none |
| prep: request parse and validation, cache key, routing, budget reservation | 1,208 | none |
| forward: provider body, response parse, text and token extraction | 1,392 | none |
| assemble: envelope, cost and metrics, event JSON, cache insert | 1,872 | the cached envelope, its key and entry; the event string in the ring |
| write: response head | 792 | none |

So two kinds of memory are mixed today:
1. **Request memory**, about 5.5 KB: dead once the response is written, and never reclaimed.
2. **Process memory with churn**, a few hundred bytes: the cache's values, keys and entries, the recent-events ring's
   strings, and, when enabled, audit entries past `max_entries` and OTLP spans. These are meant to outlive the
   request but are evicted or overwritten later, and then leak because nothing frees.

## Done in 2.10.2

The issue on `GET /v1/models/catalog` measured the listings: 413 KB kept per GET at 200 Claude-shaped entries, about 74 chats'
worth, for no provider tokens.

- **The catalog holds its body**, rebuilt when the router, the snapshot or `[[models]]` changes (`models_catalog_body`). A cache
  rather than an arena, because nothing on that path takes anything from the request, and a held body also saves the build.
- **`/v1/models` and `/api/tags` take decision 1 below**, ahead of its order: they reach no keeper, so there is no boundary to
  copy across. They use the thread's `_remote_arena()`. A builder there starts with 64 KiB of room in the arena, since
  `str_builder_add_cstr` and friends grow on the global heap, so an answer past that leaks as it did, and cannot corrupt.
- **A response head is on the stack**, so the write phase in the table above keeps nothing.

**After all three:** 176 bytes per catalog GET, 304 per `/v1/models` and 160 per `/api/tags`, all under the 1 KiB target at 200
entries. `scripts/memory-growth.py --get` repeats it, and reports `alloc_used()` from a scratch build that serves
`GET /v1/debug/alloc-used`, since VmRSS moves in whole chunks at these sizes.

## Done in 2.10.3

- **The request arena** is `src/lib/reqmem.cyr`: `req_arena()` per thread, released by `req_arena_begin()` when a job
  starts (`job_begin`, wq.cyr), as decision 1 has it. Slot 11. A thread that runs no jobs never releases its arena, and
  nothing on such a thread allocates from it per request.
- **The chat path builds there:** the prep record, the parsed request (`bayan_json_parse_a`), the cache key
  (`_cache_key_a`, with no copy of the body), the generation options, the provider body (`_build_chat_body_raw_x`, its
  Ollama translation included), a remote answer's copy, the forward's record, the completion body and its descriptor,
  and a stream's stop and usage chunks. A builder gets room in the arena (`sb_room_a`) and is read in place (`sb_cstr`).
- **The boundary**, for the keepers this path reaches: the cache copies its key and value (`cache_insert_kept`) and gives
  its key to the semantic index, and a batch item's answer is copied out (`_resp_keep`). The strings `audit_record` and
  the cost records may keep (`model_cstr`, `prompt_cstr`, the pool name) stay on the heap. The tests poison a reset.
- **A stream's chunks** are written from one buffer per thread, reused for every chunk, and its tokens read in place.
  This was not in the table above, which was measured with a one-line stream: a stream kept about 1 KB per token.
- **A request's trace context** is written into one block per thread, not two heap allocations.
- **Decision 3 for the event ring:** each slot is a buffer, copied into and reused as the ring wraps (`_ev_slot_put`),
  grown only for an event larger than any it has held. An event is built on the stack (`sb_on`), so building one takes
  nothing either. `GET /v1/events/recent` is built in the request arena, sized under the lock.

**After 2.10.3** (`scripts/memory-growth.py` against a `-D HOOSH_ALLOC_PROBE` build, which serves
`GET /v1/debug/alloc-used`):

| | 2.10.2 | 2.10.3 |
|---|---:|---:|
| buffered chat | 4,563 | 974, of which the cache keeps ~500 |
| stream, 1 token | 4,357 | 352 |
| stream, 1,000 tokens | 1,010,192 | 353 |
| `GET /v1/events/recent`, a full ring | 57,528 | 176 |

So request memory is about 480 bytes for a buffered chat, under the 1 KiB target. What remains, per phase (`alloc_used()`
marks in a scratch build, median of 300 unique chats; the rest of the 974 is the cache's map growing now and then):

| phase | bytes | what |
|---|---:|---|
| connection handling | 184 | the path's copy, the log line, auth |
| prep | 64 | `model_cstr` and the other heap strings, the message spans |
| forward | 144 | the local client's record (80), the reply text (56) and finish reason (8) copied out |
| events | 0 | the ring copies into recycled slots |
| the cache's copies | 496 | its key, value and entry: process memory, kept until evicted (decision 3) |
| assemble and write | 0 | |

## Done in 2.11.2

**Decision 1, the rest of the request:**
- **Connection handling:** the path's copy is in the arena, the request log line on the stack, and the accept loop's
  connection record is recycled (`wq.cyr`).
- **The forward:** the reply text is in the arena, the finish word on the stack, and the local client's response record
  in its per-thread scratch block, which also holds the body it points at.
- **Prep:** the spans, the model, the pool and the prompt, and compaction, compression and pruning.
- **The other paths:** every provider's body (Anthropic's and Gemini's kept about 1 KB a request), the JSON field
  readers' records, a remote request's URL and headers, the tool-call converters, embeddings, refusals and a stream's
  record.
- A per-event path (a stream's finish word, its error check, an Anthropic tool delta) uses the stack, so a long stream
  does not grow the arena.

**Decision 2:** `audit_record` copies the model, the one argument that comes from a request.

**Decision 3, recycling** (`src/lib/keep.cyr`: a size-class pool, four classes per power of two from 32 bytes to 1 MiB,
each block's class in an 8-byte header, poisoned on free in the tests):
- **The response cache.** Its entries go on a free list, and its key and value copies go back to its pool. Two rules
  keep that safe:
  - a hit is copied into the request arena under `_chat_lock` before the answer is written;
  - a removal drops every semantic entry under the key before the key is reused.
- **The audit chain.** An entry holds its three hashes, and evicted entries and model copies are reused. Hashing and
  verifying allocate nothing. The chain has a lock of its own, which `GET /v1/audit` takes: it read the chain unlocked,
  which was safe only while nothing was reused.
- **OTLP.** Its ring slots are reused as the event ring's are (`mem_slot_put`, shared in `reqmem.cyr`), and the exporter
  writes every batch into one buffer, with one https export in flight.

**After 2.11.2** (`scripts/memory-growth.py`, `-D HOOSH_ALLOC_PROBE` builds of both releases, same host, bytes kept per
request after 200 warm-up requests):

| | 2.11.1 | 2.11.2 |
|---|---:|---:|
| buffered chat, the cache filling | 974 | 622: the cached answer, held until evicted |
| buffered chat, the cache full (`--cache-max 100`) | 894 | 6 |
| a cache hit | 1,632 | 0 |
| a stream, 1 token / 1,000 tokens | 352 / 353 | 0 / 0 |
| an Anthropic / Gemini / OpenAI-compatible route, the cache full | 1,865 / 1,852 / 900 | 12 / 12 / 12 |
| the audit chain on and full, buffered / streamed | 2,318 / 1,776 | 6 / 0 |
| `GET /v1/audit`, 100 entries | 154,961 | 0 |
| OTLP export on | 2,875 | 84 |

- The 6 and 12 bytes are the pool allocating a block in a class it has not used yet: a full cache keeps 2 over 6,000
  requests.
- The 84 is the export's batch buffer growing once: 31 over 8,000 requests.
- While the cache fills, a cached answer takes about 90 bytes more than it did, for a 72-byte entry, the pool's headers
  and its rounding. The growth stops at `max_entries`.

The target this ADR set, request memory under 1 KiB per request, is met with room to spare: request memory is now 0 for
every shape measured, and what remains is process memory the keepers bound.

## Decision

Treat the two kinds separately.

### 1. Request memory: a per-thread request arena, threaded explicitly

- Each worker owns one growable arena (the 2.9.3 `_remote_arena()` generalized). It is **reset when the worker
  starts a job**: a connection, a batch item, an OTLP export or a catalog refresh. Not when a job ends, so a
  long stream keeps its memory until it is done.
- **The request path allocates from it explicitly.** It uses the stdlib's allocator-first variants
  (`str_builder_new_a`, `vec_new_a`, `map_new_a`, `str_new_a`, `bayan_json_parse_a`) and new `_a` forms of
  hoosh's own helpers (`to_cstr_a` first), handed the arena by `_chat_prep`, `_chat_forward`, `_chat_assemble`
  and `_resp_write`.
- **Anything not converted stays on the global heap.** A missed conversion leaks, as today; it cannot corrupt.
- **Order:** the chat path first (most traffic and most of the bytes), then embeddings, the hardware endpoints and the
  rest. The model listings went first, in 2.10.2.

### 2. The boundary: what crosses into process memory is copied

A request-arena pointer must never be stored where it outlives the job. The places that keep request data:

| keeper | what it keeps | rule |
|---|---|---|
| `cache_insert` | the response envelope and the cache key | copy both to process memory |
| semantic index (`semantic_insert_model`) | the key, the scope string and the embedding vector | copy |
| events (`event_emit`) | the event JSON in the recent-events ring | copy |
| audit (`audit_record`) | event, level, message, provider, model strings | copy (the model since 2.11.2; the rest are literals) |
| async batch (`batch_submit`) | the response descriptors a later `GET /v1/batch/{id}` reads | allocate from process memory |
| cost records (`_cost_find`) | the route's `base_url` | already process memory (the route's) |
| OTLP spans | span fields | copy |

**Enforcement:** in the tests, the arena is poisoned when it is reset: a hoosh-side reset fills the released
bytes. The suite runs requests through the real handlers' helpers and then reads the keepers, so a pointer that
crossed the boundary reads poison and fails an assertion, instead of passing on stale bytes.

### 3. Process memory with churn: recycle, do not leak

The keepers above get their own recycling.
- **Cache entries** (64 bytes) go on a free list, using `lib/freelist.cyr` or a per-cache list.
- **Cache values and keys** (variable size) go into size-class free lists, recycled on eviction.
- **The events ring** keeps fixed-size slots and copies into them instead of keeping a pointer to a new string.
- **Audit and OTLP** get the same treatment, bounded by their own maxima.

### Not chosen: a thread-local default allocator

A cyrius change could make `alloc()` and `default_alloc()` consult a per-thread scope, so that a request scope
catches every allocation, the stdlib's internal ones included, with no call-site changes. It is not chosen as the
first step, because it inverts the failure mode. Anything built lazily for the process during a request would land
in the arena and be reset under its readers:
- the pricing and metadata indexes;
- `keyindex` maps;
- thread-local scratch blocks;
- the stdlib's own lazy state.

That is silent corruption rather than a leak. 2.9.26's crash was exactly a lazily-built table read while half
made. It stays an option for later, behind an audit of every lazy initialization and the poison mode above.

## Consequences

- The request path's signatures grow an allocator argument. Most of the conversion is mechanical but large
  (`handlers.cyr` is about 2,500 lines), and it is done one path at a time, each measured with the harness above.
- **The target:** request memory under 1 KiB per request after the chat path, and the boundary covered by the
  poison tests.
- Until then, a long-lived instance under sustained load still grows by about 5.6 KB per buffered request and
  3.5 KB per stream, down from 139 KB and 79 KB, and with no cliff when the cache fills. Watch `/proc/<pid>/status`.

## References

- CHANGELOG 2.10.0 (the measurements), 2.10.3 (the arena, the stream, the boundary), 2.11.2 (the rest, and the
  recycling), 2.9.3 (`_remote_arena`), 2.5.11 (the per-thread request buffer).
- ADR 011 (the worker pool and its jobs).
