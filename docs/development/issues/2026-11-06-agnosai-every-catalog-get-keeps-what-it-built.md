# Every `GET /v1/models/catalog` keeps what it built, for good — OPEN

**Status:** 🟡 **OPEN** — measured 2026-11-06 on hoosh 2.10.1 (`ad87652`, the tip). ADR 013's arena has not reached this
path.
**Placement:** unpinned — backlog. ADR 013 puts the model listings third, after chat and embeddings.
**Discovered:** 2026-11-06 during agnosai 2.3.1 (its roadmap B33, ADR 030), when review asked what one re-read of the
catalog costs the gateway.
**Severity:** Medium — nothing fails, and memory only goes up.
**Affects:** hoosh 2.4.9 through 2.10.1, every version with the route. The figures are 2.10.1's.

## Summary

`handle_models_catalog` (`src/lib/handlers.cyr:224`) builds its answer on the global heap, which never frees, so all of it
stays once the response is written. This is ADR 013's *request memory*, on a path no request arena has reached.

| Route | Entries | Shape | Body | GETs | VmRSS (KiB) | Kept per GET |
|---|---:|---|---:|---:|---|---:|
| `/v1/models/catalog` | 9 | Claude-shaped | 3,524 B | 3,000 | 14,740 → 59,796 | **15.4 KB** |
| `/v1/models/catalog` | 200 | Claude-shaped | 81,643 B | 1,000 | 92,576 → 496,032 | **413 KB** |
| `/v1/models/catalog` | 9 | bare Ollama tags | 1,673 B | 3,000 | 12,696 → 41,368 | **9.8 KB** |
| `/v1/models/catalog` | 200 | bare Ollama tags | 36,626 B | 1,000 | 57,752 → 287,128 | **235 KB** |
| `/v1/models` | 9 | bare Ollama tags | 602 B | 3,000 | 12,688 → 24,976 | 4.2 KB |
| `/v1/models` | 200 | bare Ollama tags | 12,826 B | 1,000 | 22,928 → 82,320 | 60.8 KB |

KB is 1,000 bytes. *Claude-shaped* entries carry `display_name`, `context_window`, `max_output_tokens`, the `capabilities`
tree and `pricing`. *Bare* ones carry `routes` and `pricing`.

- **The cost is 4 to 6 times the body**, so it follows what a provider's list says as well as how long it is, and it grows
  a little faster than the list does.
- **A buffered chat keeps 5.6 KB** (ADR 013). At 200 Claude-shaped entries one GET keeps what about 74 chats do, and
  spends no provider tokens to do it.
- **Every picker open pays this.** thoth's model picker reads this route, and each process that reads it does.
- **One client, one GET at a time, grew the gateway by 403,456 KiB in 3.6 s** in the 200-entry run. The route is behind
  `auth_check`, so that takes a token, or nothing on a gateway with no tokens configured.

200 entries is a few live providers beside each other. `route_catalog_default` (`src/lib/route.cyr:74`) calls OpenRouter's
list alone several hundred.

## Reproduction

`scripts/memory-growth.py` with `chat(i)` replaced by a `GET /v1/models/catalog`, and its fake backend's `/api/tags`
widened to N tags, is the bare rows. For the Claude-shaped rows, point a `type = "anthropic"` route at a stub whose
`GET /v1/models` answers Anthropic's capability tree (`image_input`, `thinking.types`, `effort`) for each id. The 9-entry
rows use the current Claude ids. The 200-entry rows pad them with dated forms of one id, so a real list's body will
differ a little.

Each run was 200 warm-up GETs, then VmRSS before and after, one GET at a time, on x86_64 with `cyrius build` and no DCE.
The growth was identical in both runs of each catalog row. `/v1/models` was run once. RSS moves in whole MiB chunks here,
so read the 9-entry rows to about ±1 KB per GET. A build with `CYRIUS_DCE=1` was not measured.

## Root cause

Everything on the path takes from the global heap: the `str_builder_new()`, whose every doubling leaves the old buffer
behind (`lib/str.cyr:585`), then its build and `http_json_ok`'s copy; the `seen` vector; and for each entry the vector from
`_cat_routes_for`, `urls` and `alloc(16)`, plus `catalog_entry_new` and `alloc(40)` on the table path and `catalog_merge`
for a `[[models]]` block. That is from reading the code. The split between them was not measured.

None of it outlives the write. `/v1/models` has the same cause at a smaller size.

## Proposed fix

Either of these closes it, and which is hoosh's call. **(b) is the smaller change and need not wait for the chat path.**

**(a) The request arena** (ADR 013, decision 1). Thread the allocator through `handle_models_catalog`, `_cat_emit`,
`_cat_routes_for`, `_cat_table_entry`, `catalog_merge`, `catalog_emit_entry` and `http_json_ok`. Nothing on this path
reaches a keeper in ADR 013's table, so there is no boundary to copy across, which makes it a safe one to take ahead of
ADR 013's order. It would cover `/v1/models` and `/api/tags` with the same work.

**(b) A cached body.** `handle_models_catalog` takes nothing from the request, so one body serves every caller and a GET
allocates almost nothing. Its inputs are `catalog_snapshot()`, `_router` and `catalog_config()`. Hold the body with those
three pointers, read before building, and rebuild when any differs.

⚠ **"Rebuilt at each refresh" alone leaves a gap.** `/v1/admin/reload` replaces `_router` at once, and the old snapshot
keeps serving until `catalog_kick`'s refresh publishes (`src/main.cyr:229`). A body keyed on the snapshot alone would go on
listing a removed route. What (b) keeps is one build per change: at most four a day at the default `refresh_secs` of
21600, so about 1.7 MB a day at 200 Claude-shaped entries, plus any reload or `POST /v1/models/refresh`.

**Done means** `GET /v1/models/catalog` keeps under ADR 013's 1 KiB per request, at 200 entries.

## Consumer-side workaround

agnosai holds the catalog for its process and rations re-reads to one a minute (`_AGN_CAT_REREAD_MS`, ADR 030), and
this is the reason. One re-read per refused task would let anyone who can submit a crew grow the gateway by naming
models. **With this fixed agnosai can relax the ration**, which it keeps for hoosh's sake and not its own.

A picker has no workaround: each open pays.

Filed rather than patched: hoosh is a separate project.
