# The catalog cannot say a model's tier, whether it takes tools, or whether it chats — OPEN

**Status:** 🟡 **OPEN** — checked 2026-11-06 on hoosh 2.10.1 (`ad87652`, the tip) against stub providers.
**Placement:** unpinned — backlog. Three additive fields, each useful alone.
**Discovered:** 2026-11-06 during agnosai 2.3.1 (its roadmap B33, ADR 030), choosing a model per tier from
`GET /v1/models/catalog`.
**Severity:** Medium — a wrong model and a provider's 400 on a shipping consumer, with a per-tier override as the
workaround.
**Affects:** hoosh 2.8.0 through 2.10.1, the versions with the live catalog.

## Summary

An entry carries `id`, `owned_by`, `source`, `routes` and, when known, `display_name`, `context_window`,
`max_output_tokens`, `capabilities` (`vision`, `thinking`, `effort`) and `pricing`. A client choosing a model needs three
more things, and hoosh already holds most of them:

| A client needs | hoosh holds | The catalog says |
|---|---|---|
| How capable a model is | a tier on each row of `src/lib/metadata.cyr`, and `tier` in `[[models]]` | nothing |
| Whether it takes tools | a tools flag on each row, which `meets_requirements` (`src/lib/metadata.cyr:259`) applies to `/v1/cost/recommend` | nothing |
| Whether it chats | Google's `supportedGenerationMethods`, the word filter, three embedder rows | nothing |

**A live entry for `gpt-6-astra`**, from an OpenAI-compatible list:

```json
{"id":"gpt-6-astra","object":"model","owned_by":"openai","source":"live","routes":["…"],
 "pricing":{"input_micro_usd_per_1k":10000,"output_micro_usd_per_1k":50000}}
```

Its row says `1050000, TIER_PREMIUM`, vision 1, tools 0. ⚠ A live entry does not get even the two fields the table path
lends: the same id on a route that has not answered, `source: builtin`, carries `"context_window":1050000` and
`"capabilities":{"vision":true}`. `_cat_table_entry` (`src/lib/handlers.cyr:178`) copies those two, and only on that path.

**Ollama's tags, with embedders pulled:**

```json
{"id":"llama3.2:8b","object":"model","owned_by":"ollama","source":"live","routes":["…"],"pricing":{…}}
{"id":"nomic-embed-text:latest","object":"model","owned_by":"ollama","source":"live","routes":["…"],"pricing":{…}}
{"id":"bge-m3:latest","object":"model","owned_by":"ollama","source":"live","routes":["…"],"pricing":{…}}
```

All at price 0, and nothing tells them apart. `catalog_openai_nonchat` (`src/lib/catalog.cyr:322`) runs only inside
`catalog_parse_openai`, so `catalog_parse_ollama` (`src/lib/catalog.cyr:465`) passes every tag. The table path skips it
too, though the table's `listed = 0` covers most of that: a route of `gpt-*`, `text-embedding-*`, `whisper-*` and `nomic-*`
that has not answered lists 15 entries and no embedder. A literal pattern gets past `listed`:
`models = ["nomic-embed-text", "whisper-1", "llama3"]` lists all three as `source: builtin`, and a route has to match an
embedder for `/v1/embeddings` to reach it.

**What this costs agnosai** (its ADR 030):

- **A tier is read from prices and capabilities**, by four rules, each of them a wrong answer on a real price list first.
  It is still wrong in known cases. Only Anthropic declares `effort`, so beside OpenAI, Capable and Premium are both
  Anthropic's. An Anthropic route limited to Haiku makes Premium a Haiku. A list of prices alone makes Premium the
  dearest model, which can be an old one.
- **An agent with tools cannot be kept off `gpt-6-astra`.** On an OpenAI-only list, a `complexity: high` agent gets it and
  then a provider's 400, which no re-selection answers. This is the sharpest of these for agnostic, whose agents hold tools.
- **An embedder can stand for a tier.** On a box of local models the first listed is every tier, so a tag like `bge-m3`
  listed first is sent to chat and refused with a 400. Without a word list, an embedder is the cheapest thing on any list.

## Reproduction

Against any stub that answers the list, read `GET /v1/models/catalog`:

- **Live OpenAI-compatible:** `type = "openai"`, `models = ["*"]`, with `GET /v1/models` listing `gpt-6-astra`, `gpt-6-luna`,
  `gpt-4o`, `text-embedding-3-small`, `whisper-1` and `tts-1`. Three entries come back, with no `capabilities` on any.
- **Table path:** the same route pointed at a closed port. `gpt-6-astra` comes back `source: builtin`.
- **Ollama:** `type = "ollama"`, with `/api/tags` listing the three names above. All three come back.

## Root cause

`catalog_emit_entry` (`src/lib/catalog.cyr:1092`) and `_cat_add_caps` (`src/lib/catalog.cyr:1035`) write what an entry
holds. `CAT_E_TIER` is held, for `[[models]]` only, and is not written. An entry has no tools field at all, and
`[[models]]` has no `tools` key (`_config_model`, `src/lib/config.cyr`). `metadata_lookup` (`src/lib/metadata.cyr:241`) does
fold the table, the live list and `[[models]]`, for the cost optimizer, and the catalog does not use it.

## Proposed fix

```json
{"id":"gpt-6-astra", "…":"…", "tier":"premium", "kind":"chat",
 "capabilities":{"vision":true,"tools":false}, "pricing":{"…":"…"}}
```

All three are additive, and **all are absent when nobody said**, as `capabilities`' fields are today, so a client can tell
"no" from "unknown".

⚠ **`metadata_lookup` cannot be the source as it stands.** `_meta_from_entry` (`src/lib/metadata.cyr:222`) gives a model
with no row `TIER_STANDARD` and tools on. That is a fair default for the optimizer, and on the wire it would read as a fact.
`_meta_table_lookup` (`src/lib/metadata.cyr:209`) answers exact and then longest prefix, so a dated id finds its row, and
returns 0 for a model it does not know.

- **`tier`**: `economy`, `standard`, `premium` or `reasoning`, the words of `tier_name` and of `[[models]]`. `[[models]]`
  first, then the table. agnosai has three tiers to hoosh's four, and how it maps `reasoning` is agnosai's business.
- **`tools`** in `capabilities`: from the table, on live entries too. Add a `tools` key to `[[models]]` beside it, as whether
  `gpt-6-astra` takes tools is a provider's fact that can change, and today an operator cannot correct it. ⚠ The table's
  comment says `gpt-6-sol` and `gpt-6-luna` take tools on Chat Completions only at reasoning effort `none`, and both rows
  say 1. agnosai sends no `reasoning_effort`. Whether `true` is the right word there is hoosh's to decide.
- **`kind`**: `chat`, `embedding` and whatever else is worth naming, or an `endpoints` list (`["chat"]`, `["embeddings"]`)
  if hoosh wants room for a model that serves both. Either answers agnosai, and `kind` is the smaller. Google's list
  already says it, and the word filter's verdict is available wherever it runs. The table needs a column: `tools = 0` and
  `sys = 0` are no marker, as `starcoder2` has both and is listed. Ollama is the hard case, as `/api/tags` says nothing
  `catalog_parse_ollama` reads. From memory and not tested here, `details.family` in the tags and `capabilities` in
  `POST /api/show` may carry it.

⚠ **Please do not answer the third with a longer word list.** `bge-m3` holds none of the twelve words, and dropping an
embedder from the list takes it from a client that wants to find one. This needs a field, not a filter.

**What each one lets agnosai remove:**

| Field | Goes | Stays |
|---|---|---|
| `tier` | `_agnosai_model_pick_a`'s heuristics: `effort` over `thinking`, the median of distinct prices, and the cap of Capable under Premium | First listed within a tier, and free models off the ladder, which is a rule about spend and not a reading of a tier |
| `tools` | Nothing. It adds a check: a task's need for tools reaches the pick, and `tools: false` is passed over | |
| `kind` | `_agnosai_model_is_chat`'s twelve words, the one list of names ADR 030 keeps | Nothing, if `kind` covers Ollama's tags. If it does not, the words stay for what it leaves out |

⚠ All of it stays as the fallback for a gateway without these fields, until agnosai's floor passes the hoosh that ships them.

## Consumer-side workaround

The heuristics above, and `AGNOSAI_MODEL_FAST`, `AGNOSAI_MODEL_CAPABLE` and `AGNOSAI_MODEL_PREMIUM`, which an operator sets
to name a tier's model. They are per process and not per agent, so a crew mixing agents with and without tools cannot use
one to keep only the first off `gpt-6-astra`.

An operator can hide a tag with a `[[models]]` block of `id = "nomic-embed-text:latest"` and `listed = false` (tested).
That takes it from every client, per tag, per box.

Filed rather than patched: hoosh is a separate project, and this is a surface decision that is hoosh's to make.
