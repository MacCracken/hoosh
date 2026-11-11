# ADR-014: In-process ternary inference through tentib

**Status:** Accepted (2.11.0); updates for 2.11.1, 2026-11-10 and 2.11.2 below
**Date:** 2026-11-09

## Context

hoosh is local-first, and every local provider it has is a server on the same host: Ollama, llama.cpp, vLLM and
the rest are reached over a socket (`provider.cyr`'s raw-socket path). murti, AGNOS's own local inference engine,
is meant to sit behind hoosh as another of those, but it is still a Rust tree and has no Cyrius port to depend on.

tentib, AGNOS's ternary (1.58-bit) inference reference, has been a library since 1.3.0. It ships a distlib bundle
(`dist/tentib.cyr`) and, since 1.2.0, a packed-model file format: `tx_pack_serialize` writes a trained model,
`tx_pack_deserialize` loads it under a caller-set memory cap and returns a status code for any malformed byte, and
`tx_fwd_packed` serves it with a multiply-free integer kernel. There is no tentib server to forward to. The choice
was between building one and serving tentib in hoosh's own process.

## Decision

**A provider type `tentib` that serves a packed ternary model in hoosh's process** (`src/lib/tentib_local.cyr`),
from the file a `[[providers]]` block names:

```toml
[[providers]]
type = "tentib"
model_file = "/var/lib/hoosh/model.tpk"
models = ["tentib-hello"]
```

1. **The dependency.** `[deps.tentib]` with `modules = ["dist/tentib.cyr"]`, which brings rosnet and tyche with it
   (akshara is tentib's test-scope dependency and does not come). None of the three shares a function or global name
   with hoosh, its vendored bundles or its stdlib set; a build would warn of a duplicate.
2. **One model per process.** tentib keeps its model in module globals, so one tentib route serves: the first whose
   model loads, and any later tentib route is disabled with the reason logged. Every call into tentib holds
   `_tnt_lock`, **for one decode step**, not across a reply, so streams interleave and no lock is held across a
   socket write.
3. **Reloads.** The file is read at startup and on every reload (`config.cyr`). When it holds the bytes of the model
   serving (the same SHA-256), nothing is loaded and replies in progress go on: an operator reloads to change a key or
   a route far more often than to change the model. A load of another model bumps `_tnt_epoch` under the lock, and a
   reply that sees the epoch change stops: a buffered reply fails and is retried on the new model, and a stream ends
   with an error frame. Neither mixes two models.
4. **Loading treats the file as untrusted.** The read is capped (256 MiB), and the load's memory is capped (4 GiB,
   passed to `tx_pack_deserialize`, which computes the need and refuses before allocating, because a failed
   allocation would end the process). tentib validates every field. A refused load leaves the current model
   serving, and the route whose file was refused is disabled, with tentib's reason in words in the log. Every load
   reads into one buffer (`file_read_whole_into`), grown only for a file larger than any before, so a refused or
   unchanged reload keeps nothing.
5. **Bytes are tokens.** tentib's format v1 carries no vocabulary, so a served model uses the byte-identity one:
   V = 256, token id = byte value. A model with another V is refused at load, before tentib replaces the current
   model (the V header word is read first).
6. **tentib's `guard()` exits the process** on a precondition it refuses. Every value hoosh passes is in range by
   construction: tokens are bytes and V is 256, the window never exceeds the model's T, and nothing runs before a
   load succeeds.
7. **Decoding.** The prompt is every message's content, JSON-decoded, joined by `"\n"`. Decoding is greedy over a
   sliding window of the model's last T positions, to `max_tokens` (128 when unset, never more than 4096) or a 0
   byte. Output bytes become valid UTF-8, each invalid or unfinished sequence turning into U+FFFD, and are
   JSON-escaped. A stream sends each character once it is complete.
8. **What the route is to the rest of hoosh.** Local (DLP-confidential content may use it), healthy exactly when
   its model is loaded, with a static catalog (its patterns are its list), no embeddings (`/v1/embeddings` refuses it
   by name), no tools, and cost 0.

## Consequences

- **Positive.** No second server and no socket: a reply never leaves the process. The served logits are tentib's
  own, bit for bit (the same `tx_fwd_packed` tentib's suite gates). A bad model file cannot take the gateway down;
  it disables one route.
- **Negative.** One tentib model per process, and the worker pool shares one model lock. Each new token costs a full
  forward over the window (tentib has no incremental decode yet): about 395 µs for an 8-token reply on the bench's
  T = 25, C = 16 model. **A load of another model keeps the one it replaces in memory for the life of the process**:
  tentib allocates from the allocator that never frees, so each model loaded costs `tx_model_alloc_bytes(V, T, C, F)`
  for good (802,376 bytes at T = 30, C = 16; the load's log line states it). That figure includes tentib's training
  buffers, not just what serving reads. Greedy only: temperature and top_p are ignored. Today's tentib models are
  toy-scale.
- **Neutral.** The byte-vocabulary rule lasts until tentib's format carries a vocabulary. The memory a changed model
  keeps lasts until tentib has freeable model handles (its 2.0.0 backlog); its 1.4.x serving-only allocation shrinks it.

## Alternatives considered

- **A tentib HTTP server, reached like Ollama.** That needs a second binary, a protocol and its own failure modes,
  for a model hoosh can run itself. Rejected while tentib has no server.
- **Wait for murti.** murti is the intended general local engine, but it has no Cyrius port. When it has one,
  whether ternary models move behind it is a new decision. This ADR does not block it.
- **One model per route.** That would need tentib's state per instance, a breaking tentib change (its 2.0.0 backlog
  item, model handles).

## Update (2.11.1): tentib 1.4.0 — incremental decode, the header peek, same-shape reloads

tentib 1.4.0 shipped three of the things its first consumer asked for, and hoosh pins it. The decisions above stand
except where this says otherwise.

- **Decoding (7).** Until the window slides, a token is one tentib decode step against a cache of the window's earlier
  positions (`tx_dec_step`), not a full forward over the window. tentib's own gate is that the step's logits are the
  same bits as the full forward's last row. hoosh's suite checks it against a 2.11.0-style oracle, byte for byte: past
  the slide, from an empty prompt, from one longer than T, and with replies interleaved. Once the window slides, every
  position moves, so each token is a full forward again.
- **One cache, shared by turns.** tentib has one decode cache per process, and hoosh interleaves replies step by step
  (decision 2). The cache belongs to the reply that last reset it. Another reply's step takes it over: a reset, then that
  reply's window fed back in. That costs what the full forward it stands in for costs. So a lone reply decodes one
  position per token, and interleaved replies cost what they did in 2.11.0, never more. A per-reply cache is tentib's
  2.0.0 model-handle item. The suite pins the counts, a lone reply's (1 takeover, 22 cached steps, 17 full forwards) and
  an interleaved pair's.
- **No guard can fire.** A step never runs on a cache older than the model: a load clears the owner, and tentib would
  refuse otherwise. It never runs on a full cache either, since w − 1 < T. Decision 6 holds.
- **The vocabulary (5).** The header is read with `tx_pack_peek`, the load's own header checks with no state touched,
  instead of reading offset 24 by hand. A header tentib refuses is refused before tentib is called, and so is a V other
  than 256. A file may now declare the byte vocabulary (flags bit 0), and hoosh logs whether it is declared or assumed.
  A file that declares nothing is still served as bytes, as in 2.11.0. Lifting the byte rule needs a tokenizer in the
  file, which is tentib's 1.4.x token-table item.
- **Memory (3, and the consequences).** A load over the loaded model of the same V, T, C, F now allocates nothing:
  tentib overwrites its buffers (tentib's ADR 0003). The log says "(the replaced model's, reused)" or "(new)". A model
  of another shape still keeps its predecessor's memory, which tentib's serving-only allocation (its 1.5.x, renumbered
  from 1.4.x) shrinks and model handles end.
  - The decode cache, 16·T·C bytes (6,400 at T = 25, C = 16), is sized by the load of a model larger than any before,
    never by a reply.
  - The kept model path moved to a buffer reused by every load, like the file buffer. A reload to another file of the
    same shape therefore allocates nothing at all. Through 2.11.0, a changed `model_file` kept a copy of its path.

**Consequences.**
- **Positive:**
  - a 22-token reply on the bench's model takes about a tenth of 2.11.0's time (2.74 ms → 263 µs; see the
    CHANGELOG);
  - a same-shape reload keeps no memory;
  - the vocabulary check uses tentib's API instead of its byte layout.
- **Negative:**
  - hoosh now depends on tentib's decode cache being per process, and keeps an owner variable beside it;
  - a model whose dims match the live one's but which was built in-process (tests only) would make hoosh's "reused"
    log line wrong. Only the suite builds models in-process, and it writes every model before the first load.

## Update (2026-11-10): there is no murti port

The Context and the "Wait for murti" alternative describe murti as a Rust tree that hoosh would adopt once it had a
Cyrius port. No port is coming: the Rust repo (`MacCracken/murti` 0.1.0) was deleted on 2026-11-10. agnosticos
`docs/development/planning/murti.md` (2026-07-01) had already homed the model-load seam in hoosh: load a sovereign
weight file, place it, dispatch to rosnet or tentib, and keep the resident models. This provider is the first piece of
that seam. A murti repo comes back only if a second consumer needs the seam outside hoosh. Whether ternary models would
then move behind it is still a new decision. The decisions above stand.

## Update (2.11.2): tentib 1.5.1 — a load for serving only

hoosh pins tentib 1.5.1 and loads through `tx_pack_deserialize_serving` (tentib 1.5.0, its ADR 0004). The decisions
above stand.

- **What changes:** a load allocates `tx_serve_alloc_bytes(V, T, C, F)`: the packed store, the embeddings, gains and
  bias, and one forward's scratch. A full load also allocates the latent weights, gradients and training scratch, which
  a served model never reads.
  - At T = 25, C = 16, F = 32 that is 164,888 bytes, not 732,136.
  - At tentib's bench config it is 1,434,536, not 12,192,728.
- **What does not change:**
  - the checks and their order, the status codes, so `tnt_status_text` reads them as before;
  - the logits, bit for bit (tested against tentib's full load);
  - a same-shape reload, which allocates nothing (tentib's reuse rule covers serving loads).
- **What the "kept memory" consequence above now costs:** a change of shape keeps the replaced model's
  `tx_serve_alloc_bytes`: about a quarter of a full model at the suite's sizes, an eighth at tentib's bench config.
  tentib's 2.0.0 model handles still end it.
- **tentib 1.5.1's 2-bit weight store** is internal there and is not used. It would cut the weights to a quarter, at
  2–3× the decode cost until cyrius has a vector byte shuffle.
- **A model built in this process** (the tests only) is never reused by a serving load. So hoosh's "reused" log line
  would be wrong for one of the same dims, as this ADR already says for the full load.
